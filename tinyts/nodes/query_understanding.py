"""Query Understanding Node - LLM parses user query into structured task plan.

Uses low temperature (routing_temperature) for deterministic intent extraction.
"""

import json
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from tinyts.config import settings, get_llm
from tinyts.nodes.base import BaseNode
from tinyts.state import AgentState, DataProfile, UserTaskPlan


AVAILABLE_MODELS = [
    "Naive", "SeasonalNaive", "ARIMA", "ETS",
    "RandomForest", "LightGBM", "N-BEATS", "TinyTimeMixer",
    "RandomForest", "LightGBM", "N-BEATS",
]

QUERY_PROMPT_TEMPLATE = """You are a time series analysis assistant. Parse the user's request.

DATASET:
- {n_rows} rows x {n_cols} columns
- Time column: {time_column}, Target: {target_column}
- Frequency: {frequency} ({steps_per_day} steps/day)
- Date range: {date_range}
- Trend: {has_trend}, Seasonality: {has_seasonality} (period={seasonal_period})
- Numeric columns: {numeric_columns}

USER QUERY: "{user_query}"

Return ONLY a valid JSON object (no markdown, no comments, no explanation) with these fields:
{{
  "user_query": "<the user's query>",
  "task_type": "forecast" or "anomaly" or "both",
  "is_multivariate": false,
  "target_column": "{target_column}",
  "feature_columns": [],
  "horizon": <integer number of time steps — if user says N days, multiply by {steps_per_day}>,
  "models_included": ["Naive", "ARIMA", "ETS"] (univariate) OR ["RandomForest", "LightGBM"] (multivariate),
  "models_excluded": [],
  "needs_cv": true,
  "needs_explanation": false,
  "needs_report": false,
  "needs_plots": true,
  "explanation": "<1 sentence>",
  "counterfactual_type": null,
  "counterfactual_changes": {{}},
  "counterfactual_target_value": null,
  "counterfactual_constraints": {{}}
}}

RULES:
- horizon MUST be in time steps, not days. {steps_per_day} steps = 1 day.
- Output raw JSON only. No ```json blocks. No comments. No trailing text.
- Only set is_multivariate=true if user explicitly asks for it.
- UNIVARIATE models: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS, TinyTimeMixer
- UNIVARIATE models: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS
- MULTIVARIATE models (require features): RandomForest, LightGBM
- If is_multivariate=true, ONLY include RandomForest/LightGBM in models_included
- If is_multivariate=false, ONLY include univariate models in models_included

COUNTERFACTUAL RULES:
- If user says "what if X drops/increases by N" or "if X changes to N":
  Set counterfactual_type="forward", counterfactual_changes={{"X": delta}} (negative for drops).
  MUST set is_multivariate=true with RandomForest/LightGBM. Include X in feature_columns.
- If user says "I want target to reach/go to X" or "reduce target to X":
  Set counterfactual_type="inverse", counterfactual_target_value=X.
  MUST set is_multivariate=true with RandomForest/LightGBM.
- If user specifies constraints like "temperature can't go below 25":
  Set counterfactual_constraints={{"temperature": [25, null]}} (null=unbounded).
- If no counterfactual intent, leave all counterfactual fields as null/empty."""


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from LLM output, handling markdown blocks and comments."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code blocks
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)

    # Remove single-line JS-style comments  (// ...)
    cleaned = re.sub(r"//[^\n]*", "", cleaned)

    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _infer_steps_per_day(frequency: Optional[str]) -> int:
    """Convert frequency string to steps per day."""
    if not frequency:
        return 1
    freq = frequency.upper().strip()
    mapping = {
        "H": 24, "1H": 24, "T": 1440, "1T": 1440, "15T": 96, "15MIN": 96,
        "30T": 48, "30MIN": 48, "D": 1, "1D": 1, "W": 1, "M": 1,
    }
    for key, val in mapping.items():
        if key in freq:
            return val
    # Try to detect from common patterns
    if "hour" in freq.lower():
        return 24
    if "min" in freq.lower():
        return 96  # assume 15min
    return 1


class QueryUnderstandingNode(BaseNode):
    """LLM-based query understanding node."""

    def __init__(self):
        super().__init__("QueryUnderstanding")
        self.prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)

    def execute(self, state: AgentState) -> AgentState:
        """Parse user query into structured task plan."""
        profile: DataProfile = state.get("data_profile")
        user_query = state.get("user_query", "")

        if not user_query:
            user_query = "Forecast the target variable"

        if profile is None:
            raise ValueError("Data profile not found in state. Run DataProfiler first.")

        steps_per_day = _infer_steps_per_day(profile.inferred_frequency)

        llm = get_llm(temperature=settings.routing_temperature)

        formatted_prompt = self.prompt.format_messages(
            n_rows=profile.shape[0],
            n_cols=profile.shape[1],
            time_column=profile.time_column,
            target_column=profile.target_column,
            numeric_columns=", ".join(profile.numeric_columns[:15]),
            frequency=profile.inferred_frequency or "Unknown",
            steps_per_day=steps_per_day,
            date_range=f"{profile.date_range[0]} to {profile.date_range[1]}",
            has_trend=profile.has_trend,
            has_seasonality=profile.has_seasonality,
            seasonal_period=profile.seasonal_period or "None",
            user_query=user_query,
        )

        self.logger.info("Parsing user query with LLM...")

        # Retry LLM call with backoff (handles 503 / transient errors)
        task_plan = None
        import time as _time

        for attempt in range(3):
            try:
                response = llm.invoke(formatted_prompt)
                parsed = _extract_json(response.content)
                if parsed:
                    try:
                        task_plan = UserTaskPlan(**parsed)
                        if not task_plan.target_column:
                            task_plan.target_column = profile.target_column
                    except Exception as e:
                        self.logger.warning(f"Pydantic validation failed: {e}")
                break  # Success — stop retrying
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    _time.sleep(2 ** attempt)  # 1s, 2s backoff

        if task_plan is None:
            self.logger.warning("LLM parse failed, using fallback heuristics")
            task_plan = self._create_fallback_plan(user_query, profile, steps_per_day)

        state["user_task_plan"] = task_plan
        self.logger.info(
            f"Query understood: task={task_plan.task_type}, "
            f"multivariate={task_plan.is_multivariate}, "
            f"horizon={task_plan.horizon}, "
            f"models={task_plan.models_included}"
        )

        return state

    def _create_fallback_plan(self, query: str, profile: DataProfile, steps_per_day: int) -> UserTaskPlan:
        """Create fallback plan using heuristics when LLM parsing fails."""
        query_lower = query.lower()

        # Detect task type
        if "anomal" in query_lower or "outlier" in query_lower or "detect" in query_lower:
            task_type = "anomaly"
        elif "both" in query_lower:
            task_type = "both"
        else:
            task_type = "forecast"

        # Detect multivariate intent
        is_multivariate = any(
            w in query_lower for w in ["multivariate", "features", "variables", "covariates"]
        )

        # Detect horizon — look for "N days" or "N hours" or bare number
        horizon = None
        import re as _re
        day_match = _re.search(r"(\d+)\s*day", query_lower)
        hour_match = _re.search(r"(\d+)\s*hour", query_lower)
        bare_match = _re.search(r"\b(\d+)\b", query_lower)

        if day_match:
            horizon = int(day_match.group(1)) * steps_per_day
        elif hour_match:
            horizon = int(hour_match.group(1))
        elif bare_match:
            horizon = int(bare_match.group(1)) * steps_per_day

        # Default model selection based on multivariate flag
        feature_cols = []
        if is_multivariate:
            feature_cols = [
                c for c in profile.numeric_columns
                if c != profile.target_column
            ]
            models = ["RandomForest", "LightGBM"]
        else:
            models = ["Naive", "ARIMA", "ETS"]
            if profile.has_seasonality:
                models.insert(1, "SeasonalNaive")

        # Detect counterfactual intent
        cf_type = None
        cf_changes = {}
        cf_target = None
        cf_constraints = {}
        if any(w in query_lower for w in ["what if", "what-if", "if the", "drops by", "increases by"]):
            cf_type = "forward"
            is_multivariate = True
            feature_cols = [c for c in profile.numeric_columns if c != profile.target_column]
            models = ["RandomForest", "LightGBM"]
        elif any(w in query_lower for w in ["want it to", "reach", "reduce to", "go down to", "go to"]):
            cf_type = "inverse"
            is_multivariate = True
            feature_cols = [c for c in profile.numeric_columns if c != profile.target_column]
            models = ["RandomForest", "LightGBM"]
            # Try to extract target value
            val_match = _re.search(r"to\s+(\d+\.?\d*)", query_lower)
            if val_match:
                cf_target = float(val_match.group(1))

        return UserTaskPlan(
            user_query=query,
            task_type=task_type,
            is_multivariate=is_multivariate,
            target_column=profile.target_column,
            feature_columns=feature_cols,
            horizon=horizon,
            models_included=models,
            models_excluded=[],
            needs_cv=True,
            needs_explanation="explain" in query_lower or "why" in query_lower,
            needs_report="report" in query_lower,
            needs_plots=True,
            explanation="Fallback plan: heuristic intent detection.",
            counterfactual_type=cf_type,
            counterfactual_changes=cf_changes,
            counterfactual_target_value=cf_target,
            counterfactual_constraints=cf_constraints,
        )
