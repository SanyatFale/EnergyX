"""Central LLM agent for TinyTS.

Replaces LangGraph DAG with a tool-calling agent loop.
Two phases:
  1. plan()   — profile dataset + LLM query understanding → UserTaskPlan
  2. execute() — LLM calls tools in a loop based on approved plan
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    HumanMessage, SystemMessage, ToolMessage,
)

from tinyts.agent_tools import create_agent_tools
from tinyts.config import get_llm, settings
from tinyts.state import UserTaskPlan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TinyTS-Scientist, an expert time series analysis agent.

You have these tools:
1. profile_dataset() — Profile the dataset (call FIRST, no args)
2. train_forecast_model(model_name, horizon) — Train one model with CV
3. train_and_explain_forecast(model_name, horizon) — Train + compute SHAP/FI/decomposition
4. select_ensemble_strategy() — Compute ensemble weights from all trained models
5. combine_forecasts() — Combine predictions using ensemble strategy
6. detect_anomalies() — Run 7-method anomaly ensemble
7. explain_anomalies() — Explain anomaly detection results
8. generate_report() — Generate analysis report

AVAILABLE MODELS:
- UNIVARIATE: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS, TinyTimeMixer
- MULTIVARIATE (requires features): RandomForest, LightGBM

CRITICAL TOOL-CALLING PROTOCOL:
1. Call EXACTLY ONE tool per response - NEVER multiple tools
2. Do NOT add ANY text when calling a tool - just the tool call
3. Do NOT add extra arguments - tools have fixed signatures
4. Only provide analysis AFTER all tools finish (no more tool_calls)

WORKFLOW:
- FORECASTING: train models → combine_forecasts() → (optional) generate_report()
- ANOMALY: detect_anomalies() → (optional) explain_anomalies() → (optional) generate_report()

FINAL SUMMARY (only when NO MORE TOOLS to call):
Provide a concise summary citing actual numbers from tool results.

1. **Model Performance**: Table of each model's MAPE. Which won and why.

2. **Ensemble Strategy Justification**: WHY weighted/best_model was chosen.
   Justify based on: error variance across models, relative stability (std of CV scores),
   model diversity (statistical vs tree vs neural), and whether combining helps.

3. **Lag Structure Analysis** (from lag_contributions):
   - Negative lag_1 → short-term mean reversion, volatile signal, naive forecast unstable
   - Positive lag_1 → persistent series, momentum
   - Weak seasonal lags (12, 24) → weak seasonality, tree models may struggle with cycles
   - Strong seasonal lags → clear periodic pattern models can exploit

4. **Decomposition Insights** (from trend_strength, seasonal_strength):
   - trend_strength near 1 → strong trend, differencing helps
   - seasonal_strength near 0 → weak seasonality, seasonal models add little
   - Relate to model results: did seasonal models actually help?

5. **Feature Reconciliation** (MUST reconcile FI, SHAP, and correlation together):
   - High SHAP + positive correlation → strong direct driver
   - Low correlation + moderate FI → non-linear effect the model captures
   - FI high but SHAP small → feature used often in splits but low marginal effect
   - Example interpretation: "air_temperature has FI=0.93 (dominant), SHAP=176
     (high magnitude), but correlation=0.29 (moderate). This suggests a strong
     but non-linear temperature-demand relationship."

6. **Statistical Context** (from recent_trend, recent_7d_avg, recent_1d_avg):
   - Is the series trending up/down recently?
   - How does recent behavior compare to training mean?

7. **Anomaly Summary** (if detect_anomalies was used):
   - Count, method agreement levels, what anomalous values look like
   - Which methods agreed most/least
"""


class TinyTSAgent:
    """Central agent that orchestrates TinyTS via tool calling."""

    def __init__(
        self,
        dataset_path: str,
        time_column: str,
        target_column: str,
        output_dir: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ):
        self.run_id = str(uuid.uuid4())[:8]
        if output_dir is None:
            output_dir = str(settings.output_dir / f"run_{self.run_id}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.dataset_path = dataset_path
        self.time_column = time_column
        self.target_column = target_column
        self.output_dir = output_dir
        self.feature_columns = feature_columns

        self.tools, self.session = create_agent_tools(
            dataset_path, time_column, target_column, output_dir, feature_columns,
        )
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------
    def plan(self, query: str) -> UserTaskPlan:
        """Profile the dataset and parse the user query into a plan.

        Returns a UserTaskPlan that the UI can display / let the user edit.
        """
        # Run profile tool
        profile_json = self.tool_map["profile_dataset"].invoke({})
        profile = self.session["profile"]

        # Query understanding (reuse existing prompt logic)
        from tinyts.nodes.query_understanding import (
            QUERY_PROMPT_TEMPLATE, _extract_json, _infer_steps_per_day,
        )
        from langchain_core.prompts import ChatPromptTemplate

        spd = _infer_steps_per_day(profile.inferred_frequency)
        prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        msgs = prompt.format_messages(
            n_rows=profile.shape[0],
            n_cols=profile.shape[1],
            time_column=self.time_column,
            target_column=self.target_column,
            numeric_columns=", ".join(profile.numeric_columns[:15]),
            frequency=profile.inferred_frequency or "Unknown",
            steps_per_day=spd,
            date_range=f"{profile.date_range[0]} to {profile.date_range[1]}",
            has_trend=profile.has_trend,
            has_seasonality=profile.has_seasonality,
            seasonal_period=profile.seasonal_period or "None",
            user_query=query,
        )

        llm = get_llm(temperature=settings.routing_temperature)
        task_plan = None

        for attempt in range(3):
            try:
                resp = llm.invoke(msgs)
                parsed = _extract_json(resp.content)
                if parsed:
                    task_plan = UserTaskPlan(**parsed)
                    if not task_plan.target_column:
                        task_plan.target_column = self.target_column
                break
            except Exception as e:
                logger.warning(f"Plan attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        if task_plan is None:
            task_plan = self._fallback_plan(query, profile, spd)

        return task_plan

    # ------------------------------------------------------------------
    # Phase 2: Execute
    # ------------------------------------------------------------------
    def execute(
        self,
        plan: UserTaskPlan,
        on_tool_call: Optional[Callable] = None,
        max_steps: int = 25,
    ) -> Dict[str, Any]:
        """Execute the approved plan via tool-calling loop.

        Args:
            plan: Approved UserTaskPlan
            on_tool_call: callback(tool_name, args, result_str) for progress
            max_steps: Safety limit on iterations

        Returns:
            Dict with response, session, messages, output_dir
        """
        plan_text = self._plan_to_prompt(plan)

        llm = get_llm(temperature=settings.routing_temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=plan_text),
        ]

        logger.info(f"Agent executing plan: {plan.task_type}")

        for step in range(max_steps):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"LLM call failed step {step}: {e}")
                time.sleep(2)
                try:
                    response = llm_with_tools.invoke(messages)
                except Exception as e2:
                    return self._make_result(f"Agent failed: {e2}", messages)

            messages.append(response)

            if not response.tool_calls:
                logger.info(f"Agent done in {step+1} steps")
                break

            for tc in response.tool_calls:
                name, args, tid = tc["name"], tc["args"], tc["id"]
                logger.info(f"  Step {step}: {name}({args})")

                fn = self.tool_map.get(name)
                if fn is None:
                    res = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    try:
                        res = fn.invoke(args)
                    except Exception as e:
                        res = json.dumps({"error": f"{name} failed: {e}"})

                messages.append(ToolMessage(content=res, tool_call_id=tid))

                if on_tool_call:
                    on_tool_call(name, args, res)

        final = ""
        if messages and hasattr(messages[-1], "content"):
            final = messages[-1].content or ""

        return self._make_result(final, messages)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _plan_to_prompt(self, plan: UserTaskPlan) -> str:
        lines = [
            "Execute this approved analysis plan:",
            f"- Task: {plan.task_type}",
            f"- Horizon: {plan.horizon}",
            f"- Multivariate: {plan.is_multivariate}",
        ]
        if plan.feature_columns:
            lines.append(f"- Feature columns: {', '.join(plan.feature_columns)}")
        if plan.models_included:
            lines.append(f"- Models to train: {', '.join(plan.models_included)}")
        lines.append(f"- Needs explanation: {plan.needs_explanation}")
        lines.append(f"- Needs report: {plan.needs_report}")
        lines.append("")
        lines.append("The dataset is already profiled. Start executing now.")
        return "\n".join(lines)

    def _fallback_plan(self, query, profile, steps_per_day) -> UserTaskPlan:
        """Heuristic fallback when LLM parsing fails."""
        from tinyts.nodes.query_understanding import QueryUnderstandingNode
        node = QueryUnderstandingNode()
        return node._create_fallback_plan(query, profile, steps_per_day)

    def _make_result(self, response_text, messages):
        return {
            "response": response_text,
            "session": self.session,
            "messages": messages,
            "output_dir": self.output_dir,
        }
