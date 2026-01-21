import json
from typing import List
from .models import Transcript, AnalysisResult, AggregateReport, TimelineEvent
from .bedrock_client import BedrockClient

class TimelineAgent:
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock_client = bedrock_client

    def extract_timeline(self, transcript: Transcript) -> List[TimelineEvent]:
        system_prompt = "You are an expert event analyst. Your job is to extract a chronological timeline of events from a customer service transcript, distinguishing between what the customer says happened and what happens during the call."
        
        prompt = f"""
        Read the following transcript between 'Speaker 1' and 'Speaker 2'. Identify the key events that have occurred leading up to this call and during this call.
        
        Transcript:
        {transcript.content}
        
        Extract a list of events in chronological order. For each event identify:
        - timestamp_order: A sequential integer (1, 2, 3...)
        - description: What happened (e.g., "Customer requested flatbed", "Driver arrived with wrong truck")
        - actor: Who did it (Customer, Agent, Provider, System)
        
        Return the response as a valid JSON list of objects.
        """
        
        response_text = self.bedrock_client.invoke_model(prompt, system_prompt=system_prompt)
        
        if not response_text:
             print(f"Timeline extraction failed for {transcript.id}")
             return []

        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            return [TimelineEvent(**item) for item in data]
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing timeline: {e}")
            return []

class RootCauseAgent:
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock_client = bedrock_client

    def analyze_root_cause(self, transcript: Transcript, timeline: List[TimelineEvent]) -> AnalysisResult:
        timeline_str = "\n".join([f"{e.timestamp_order}. [{e.actor}] {e.description}" for e in timeline])
        
        system_prompt = "You are an expert operational root cause analyst. You analyze event timelines to find the fundamental breakdown in the service."
        
        prompt = f"""
        Analyze the following timeline of events for a roadside assistance request.
        
        Timeline:
        {timeline_str}
        
        Determine the primary root cause of why the customer is calling back.
        
        Return valid JSON with:
        - root_cause: Specific description of the failure.
        - root_cause_category: The high-level category (e.g., ETA Missed, Equipment Mismatch, Service Change, Location Issue, Provider Cancellation).
        - sentiment: Customer sentiment (Positive, Neutral, Negative).
        - summary: A brief summary of the situation.
        - actionable_insight: A strategic recommendation to prevent this specific failure mode.
        """
        
        response_text = self.bedrock_client.invoke_model(prompt, system_prompt=system_prompt)
        
        # Default values in case of failure
        result = {
            "root_cause": "Unknown",
            "root_cause_category": "Other",
            "sentiment": "Neutral", 
            "summary": "Analysis failed",
            "actionable_insight": None
        }

        if response_text:
            try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean_text)
            except json.JSONDecodeError:
                print(f"Error parsing root cause analysis: {response_text}")

        return AnalysisResult(
            transcript_id=transcript.id,
            timeline=timeline,
            root_cause=result.get("root_cause", "Unknown"),
            root_cause_category=result.get("root_cause_category", "Other"),
            sentiment=result.get("sentiment", "Neutral"),
            summary=result.get("summary", "No summary"),
            actionable_insight=result.get("actionable_insight")
        )

class AggregationAgent:
    def __init__(self, bedrock_client: BedrockClient):
         self.bedrock_client = bedrock_client

    def aggregate_results(self, results: List[AnalysisResult]) -> AggregateReport:
        total = len(results)
        root_cause_counts = {}
        sentiment_counts = {}
        
        for r in results:
            root_cause_counts[r.root_cause_category] = root_cause_counts.get(r.root_cause_category, 0) + 1
            sentiment_counts[r.sentiment] = sentiment_counts.get(r.sentiment, 0) + 1

        # Synthesize findings based on richer data
        analysis_summaries = []
        for r in results:
            timeline_summary = " -> ".join([t.description for t in r.timeline[:3]]) + "..."
            analysis_summaries.append(f"- [{r.root_cause_category}] {r.root_cause}. Sequence: {timeline_summary}")

        formatted_summaries = "\n".join(analysis_summaries[:30]) # Limit context

        system_prompt = "You are a strategic operations director. Aggregate these findings into a strategic report."
        prompt = f"""
        Analyze {total} roadside assistance cases.
        
        Stats:
        Root Causes: {json.dumps(root_cause_counts)}
        Sentiments: {json.dumps(sentiment_counts)}
        
        Case Summaries:
        {formatted_summaries}
        
        Identify:
        1. Common Timeline Patterns: What sequences of events repeatedly lead to failure?
        2. Key Strategic Findings.
        3. Strategic Recommendations.
        
        Return JSON with keys: common_timeline_patterns, key_findings, recommendations.
        """
        
        response_text = self.bedrock_client.invoke_model(prompt, system_prompt=system_prompt)
        
        output = {"common_timeline_patterns": [], "key_findings": [], "recommendations": []}
        
        if response_text:
             try:
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                output = json.loads(clean_text)
             except json.JSONDecodeError:
                print("Failed to decode aggregation JSON")

        return AggregateReport(
            total_transcripts=total,
            root_cause_distribution=root_cause_counts,
            sentiment_distribution=sentiment_counts,
            common_timeline_patterns=output.get("common_timeline_patterns", []),
            key_findings=output.get("key_findings", []),
            recommendations=output.get("recommendations", [])
        )
