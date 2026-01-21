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
from src.agents import TimelineAgent, RootCauseAgent, AggregationAgent
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

    # Reload all results from checkpoint for aggregation (including previously processed ones)
    print("Loading all results for aggregation...")
    all_results = []
    if os.path.exists(config.CHECKPOINT_FILE):
        with open(config.CHECKPOINT_FILE, "r") as f:
            for line in f:
                try:
                    all_results.append(AnalysisResult(**json.loads(line)))
                except Exception as e:
                    print(f"Skipping corrupt checkpoint line: {e}")

    # 4. Aggregation
    if all_results:
        print(f"Aggregating {len(all_results)} results...")
        try:
            aggregate_report = aggregation_agent.aggregate_results(all_results)
            
            final_output = {
                "summary_report": aggregate_report.model_dump(),
                "detailed_results": [r.model_dump() for r in all_results]
            }
            
            with open(args.output, "w") as f:
                json.dump(final_output, f, indent=2)
            print(f"Final report saved to {args.output}")
            
        except Exception as e:
            print(f"Error during aggregation: {e}")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()
