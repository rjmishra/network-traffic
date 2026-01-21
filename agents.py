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
        
        CRITICAL INSTRUCTION - ROOT CAUSE CATEGORIZATION:
        - Do NOT use generic categories like "Provider Issue" or "Customer Service". 
        - Use SPECIFIC, ACTIONABLE categories that describe the operational failure.
        - Normalize similar causes (e.g., "Driver late", "Tow truck delayed" -> "ETA Exceeded").
        
        Recommended Categories (expand if necessary, but keep granularity similar):
        - "Dispatch Accuracy - Wrong Equipment" (e.g., sent wheel lift instead of flatbed)
        - "Dispatch Accuracy - Wrong Location" (e.g., driver sent to wrong address)
        - "Customer Request - Change Drop-off Location" (e.g., user wants to tow to different shop)
        - "Customer Request - Change Pick-up Location" (e.g., user moved car or location was inaccurate)
        - "Customer Request - Service Upgrade" (e.g., jumpstart failed, needs tow)
        - "Provider Operations - ETA Exceeded" (e.g., driver significantly late)
        - "Provider Operations - No Show" (e.g., driver cancelled or never arrived)
        - "Provider Operations - Unprofessional Conduct"
        - "System/App - Payment Failure"
        - "System/App - Tracking Inaccuracy"
        
        Return valid JSON with:
        - root_cause: A concise, specific description of what went wrong (max 10 words).
        - root_cause_category: The specific category as defined above.
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
class NormalizationAgent:
    def __init__(self, bedrock_client: BedrockClient):
        self.bedrock_client = bedrock_client

    def normalize_categories(self, categories: List[str]) -> dict[str, str]:
        if not categories:
            return {}
            
        system_prompt = "You are a data cleanliness expert. Your job is to consolidate synonymous categories into a single canonical version."
        
        mapping = {}
        batch_size = 50
        
        for i in range(0, len(categories), batch_size):
            batch = categories[i:i + batch_size]
            
            prompt = f"""
            Review the following list of root cause categories extracted from customer service transcripts. 
            Many are duplicates or synonyms (e.g., "Cancel", "Cancellation", "User Cancel" -> "Customer Request - Cancellation").
            
            Input Categories:
            {json.dumps(batch, indent=2)}
            
            Task:
            1. Identify groups of synonymous categories.
            2. Choose (or create) a single Descriptive Canonical Category for each group.
            3. Map EVERY input category to its canonical version.
            
            Return a JSON object where keys are the input categories and values are the canonical categories.
            Example: {{"User Cancel": "Customer Request - Cancellation", "Cancellation": "Customer Request - Cancellation"}}
            """
            
            response_text = self.bedrock_client.invoke_model(prompt, system_prompt=system_prompt)
            
            if response_text:
                try:
                    clean_text = response_text.replace("```json", "").replace("```", "").strip()
                    batch_mapping = json.loads(clean_text)
                    mapping.update(batch_mapping)
                except json.JSONDecodeError:
                    print(f"Error parsing normalization mapping for batch {i}: {response_text}")
                    # Fallback for this batch
                    for c in batch:
                        mapping[c] = c
            else:
                 for c in batch:
                        mapping[c] = c
                        
        return mapping
class AggregationAgent:
    def __init__(self, bedrock_client: BedrockClient):
         self.bedrock_client = bedrock_client

    def aggregate_results(self, root_cause_counts: dict, sentiment_counts: dict, sample_summaries: List[str], total: int) -> AggregateReport:
        # Limit context
        formatted_summaries = "\n".join(sample_summaries[:30])

        system_prompt = "You are a strategic operations director. Aggregate these findings into a strategic report."
        prompt = f"""
        Analyze {total} roadside assistance cases.
        
        Stats:
        Root Causes: {json.dumps(root_cause_counts)}
        Sentiments: {json.dumps(sentiment_counts)}
        
        Case Summaries (Sample):
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
