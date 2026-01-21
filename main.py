
import argparse
import json
import os
import sys
import concurrent.futures
import threading
from typing import List, Set

# Add the src directory to the python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import Transcript, AnalysisResult
from src.bedrock_client import BedrockClient
from src.agents import TimelineAgent, RootCauseAgent, AggregationAgent, NormalizationAgent
from src import config

# Thread-safe lock for writing checkpoints
checkpoint_lock = threading.Lock()

def load_transcripts(file_path: str) -> List[Transcript]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return [Transcript(**item) for item in data]

def load_processed_ids(checkpoint_path: str) -> Set[str]:
    ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    ids.add(record.get("transcript_id"))
                except:
                    pass
    return ids

def save_checkpoint(result: AnalysisResult, checkpoint_path: str):
    with checkpoint_lock:
        with open(checkpoint_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

def log_failure(transcript_id: str, error: str, failure_path: str):
    with checkpoint_lock:
        with open(failure_path, "a") as f:
            json.dump({"transcript_id": transcript_id, "error": str(error)}, f)
            f.write("\n")

def process_single_transcript(transcript: Transcript, timeline_agent: TimelineAgent, root_cause_agent: RootCauseAgent) -> AnalysisResult:
    # 1. Timeline
    timeline = timeline_agent.extract_timeline(transcript)
    if not timeline:
        raise Exception("Failed to extract timeline")
    
    # 2. Root Cause
    result = root_cause_agent.analyze_root_cause(transcript, timeline)
    return result

def main():
    parser = argparse.ArgumentParser(description="Roadside Assistance Root Cause Analysis (Production)")
    parser.add_argument("--input", required=True, help="Path to input transcripts JSON file")
    parser.add_argument("--output", required=True, help="Path to output report JSON file")
    args = parser.parse_args()

    # Ensure output dirs
    os.makedirs(os.path.dirname(config.CHECKPOINT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(config.FAILURES_FILE), exist_ok=True)

    # Initialize Bedrock
    try:
        bedrock = BedrockClient()
    except Exception as e:
        print(f"Failed to initialize AWS client: {e}")
        return

    # Initialize Agents
    timeline_agent = TimelineAgent(bedrock)
    root_cause_agent = RootCauseAgent(bedrock)
    normalization_agent = NormalizationAgent(bedrock)
    aggregation_agent = AggregationAgent(bedrock)

    # 1. Load Data
    print(f"Loading transcripts from {args.input}...")
    transcripts = load_transcripts(args.input)
    
    # 2. Filter Already Processed
    processed_ids = load_processed_ids(config.CHECKPOINT_FILE)
    to_process = [t for t in transcripts if t.id not in processed_ids]
    
    print(f"Total: {len(transcripts)}, Processed: {len(processed_ids)}, Remaining: {len(to_process)}")

    # 3. Parallel Analysis
    analysis_results = []
    
    # Helper for the thread executor
    def task_wrapper(t: Transcript):
        try:
            res = process_single_transcript(t, timeline_agent, root_cause_agent)
            save_checkpoint(res, config.CHECKPOINT_FILE)
            return res
        except Exception as e:
            print(f"Failed to process {t.id}: {e}")
            log_failure(t.id, str(e), config.FAILURES_FILE)
            return None

    if to_process:
        print(f"Starting analysis with {config.MAX_WORKERS} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures = {executor.submit(task_wrapper, t): t for t in to_process}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res:
                    analysis_results.append(res)
                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(to_process)} completed")

    # Post-Processing: Normalization and Aggregation
    print("Post-processing results for aggregation...")
    
    unique_categories = set()
    total_processed = 0
    
    # Pass 1: Collect Unique Categories
    if os.path.exists(config.CHECKPOINT_FILE):
        with open(config.CHECKPOINT_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    unique_categories.add(data.get("root_cause_category"))
                    total_processed += 1
                except:
                    pass
    
    # Normalize
    category_mapping = {}
    if unique_categories:
        print(f"Normalizing {len(unique_categories)} unique categories...")
        category_mapping = normalization_agent.normalize_categories(list(unique_categories))

    # Pass 2: Stream Aggregation
    root_cause_counts = {}
    sentiment_counts = {}
    sample_summaries = []
    
    print("Aggregating statistics...")
    if os.path.exists(config.CHECKPOINT_FILE):
        with open(config.CHECKPOINT_FILE, "r") as f:
            for line in f:
                try:
                    # Parse only what we need if possible, but for JSON we load all
                    data = json.loads(line)
                    result = AnalysisResult(**data)
                    
                    # Apply Normalization
                    cat = result.root_cause_category
                    if cat in category_mapping:
                        cat = category_mapping[cat]
                    
                    # Count
                    root_cause_counts[cat] = root_cause_counts.get(cat, 0) + 1
                    sentiment_counts[result.sentiment] = sentiment_counts.get(result.sentiment, 0) + 1
                    
                    # Sample (Reservoir sampling-ish or just first N)
                    if len(sample_summaries) < 40:
                        timeline_str = " -> ".join([t.description for t in result.timeline[:3]])
                        sample_summaries.append(f"- [{cat}] {result.root_cause}. Sequence: {timeline_str}...")
                        
                except Exception as e:
                    print(f"Skipping line during aggregation: {e}")

    # Final Report Generation
    if total_processed > 0:
        try:
            aggregate_report = aggregation_agent.aggregate_results(
                root_cause_counts=root_cause_counts,
                sentiment_counts=sentiment_counts,
                sample_summaries=sample_summaries,
                total=total_processed
            )
            
            # Save Summary Only (Detailed results are in checkpoint)
            # For the web UI, we might want to support pagination API instead of a huge JSON file,
            # but for now we will adhere to the interface.
            # To avoid writing 100k records to 'report.json', we might just save the summary 
            # and let the UI query the checkpoint or a DB. 
            # However, to keep backward compatibility with current UI:
            
            final_output = {
                "summary_report": aggregate_report.model_dump(),
                "detailed_results": [] # Optimization: Don't duplicate 100MB of data here. 
                                       # UI should eventually load from API which reads checkpoint line-by-line.
            }
            
            with open(args.output, "w") as f:
                json.dump(final_output, f, indent=2)
            print(f"Final summary saved to {args.output}")
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()
