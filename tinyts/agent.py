"""Central LLM agent for TinyTS.

Replaces LangGraph DAG with a tool-calling agent loop.
Two phases:
  1. plan()   — profile dataset + LLM query understanding → UserTaskPlan
  2. execute() — LLM calls tools in a loop based on approved plan
"""

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage,
)

from tinyts.agent_tools import create_agent_tools
from tinyts.config import get_llm, settings
from tinyts.state import UserTaskPlan

logger = logging.getLogger(__name__)

BASE_PROMPT = """You are EnergyX, an expert time series analysis agent.

PROTOCOL:
- Call ONE tool per response. No text, just the tool call.
- Wait for the result, then decide the next tool.
- After ALL tools are done, provide your analysis.

TOOLS:
- profile_dataset() — Profile dataset (call first)
- train_forecast_model(model_name, horizon) — Train one model
- train_and_explain_forecast(model_name, horizon) — Train + SHAP/FI
- combine_forecasts() — Combine trained models (call after training all)
- detect_anomalies() — Anomaly ensemble (auto-uses multivariate when features configured)
- explain_anomalies() — Explain anomalies (after detect)
- generate_report() — Generate report (call last)
- counterfactual_forward(changes_json, horizon) — What-if forecast
- counterfactual_inverse(target_value, constraints_json) — Reach target

MODELS: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS (univariate) | RandomForest, LightGBM (multivariate)

RESILIENCE:
- If a tool returns "status": "warning", read the warnings and factor them into your analysis.
- If a tool returns an error, retry once. If it fails again, skip it and continue with remaining tasks.
- Report any skipped or failed tools in your final output so the user knows what was not completed.
"""

FORECAST_WORKFLOW = """
GOAL: Forecast the target variable for {horizon} steps.
Models to evaluate: {models}
Train each model, combine results into an ensemble, then present your analysis.

OUTPUT REQUIREMENTS:
1. A markdown table ALLthe  predicted values from the ensemble (step | value)
2. Each model's MAPE, ensemble weights, which model won and why.
"""

FORECAST_EXPLAIN_WORKFLOW = """
GOAL: Forecast the target variable for {horizon} steps with full explainability.
Models to evaluate: {models}
Train each model with explanations, combine results, then present grounded analysis.

OUTPUT REQUIREMENTS:
1. A markdown table of ALL predicted values from the ensemble (step | value).
2. Grounded analysis using ALL returned metrics:
- Model comparison (MAPE, std) — which won and why
- Feature drivers: cite exact FI% and SHAP% — explain what each feature DOES practically (e.g. "air_temperature FI=35% means temperature is the strongest predictor of energy use")
- Lag structure: interpret lags as real-world patterns:
  * lag_1 high → strong momentum / persistence (recent values predict next)
  * lag_24 high → daily repeating pattern (e.g. daily consumption cycle)
  * lag_168 high → weekly pattern (e.g. weekday vs weekend)
  * lag_1 negative → mean-reverting (spikes tend to correct)
- Decomposition: trend direction, seasonal strength, residual noise
- Correlations: which features directly drive target vs non-linear effects. If correlation=0 but FI>0, explain this means the feature matters but through complex interactions, not simple linear relationship
Write for both experts (cite numbers) and non-experts (explain what it means in plain language).
"""

ANOMALY_WORKFLOW = """
GOAL: Detect anomalies in the dataset.
Run anomaly detection and report results.
If feature columns are configured, detection automatically uses multivariate mode
(IsolationForest & DBSCAN on full feature matrix; Z-score/MAD/IQR/Rolling/STL per-channel with union voting).

OUTPUT REQUIREMENTS:
Total anomalies found, per-method counts, agreement level.
If multivariate: note which features contributed to detections.
"""

ANOMALY_EXPLAIN_WORKFLOW = """
GOAL: Detect and explain anomalies in the dataset.
Run anomaly detection, then compute explanations for the detected anomalies.
If feature columns are configured, detection automatically uses multivariate mode.

OUTPUT REQUIREMENTS:
Grounded analysis. For EACH metric, cite the value AND infer what it means:
- Method agreement: cite counts → infer if anomalies are clear-cut or borderline
- Z-scores: cite values → infer severity (mild fluctuation vs extreme event)
- IQR bounds: cite Q1/Q3/range → infer if anomalies are just outside normal or far beyond
- Percentiles: cite values → infer how rare these events are
- Context snapshots: compare target_window values before/during/after anomaly AND compare feature_window values → infer probable CAUSE (did a feature also spike/drop? or did only the target change while features stayed stable? if features changed too, which one correlates with the anomaly?)
For each anomaly snapshot, write one sentence: "Anomaly at index X: target dropped from A to B while [feature] also dropped from C to D, suggesting [cause]" or "target spiked but features were stable, suggesting measurement error or external event."
"""

COUNTERFACTUAL_FORWARD_WORKFLOW = """
GOAL: Evaluate a what-if scenario for {horizon} steps.
Feature changes to apply: {changes_json}
Models to evaluate: {models}
Train the multivariate models, then run the counterfactual scenario.

OUTPUT REQUIREMENTS:
- Baseline vs counterfactual forecast comparison (cite exact values)
- Mean and max impact of the intervention
- Practical interpretation: what this change means for the target variable
- Confidence: cite model MAPE to contextualize prediction reliability
"""

COUNTERFACTUAL_INVERSE_WORKFLOW = """
GOAL: Find what feature changes are needed to reach a target value of {target} over {horizon} steps.
Models to evaluate: {models}
Constraints: {constraints_json}
Train multivariate models with explanations, then run the inverse optimization.

OUTPUT REQUIREMENTS:
- Current vs target vs optimized prediction (cite exact values)
- Feature recommendations: current value, recommended value, change needed (cite %)
- Feasibility: are changes within historical bounds? did optimizer converge?
- Practical interpretation: what actions to take and expected outcome
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

        Profiles the FULL unfiltered dataset so the LLM can see all
        categorical values and resolve columns/filters from the query.
        After this returns, call ``reinitialize_with_plan(plan)`` to
        re-create tools with the resolved columns and filters applied.

        Returns a UserTaskPlan that the UI can display / let the user edit.
        """
        # Run profile tool (on the FULL unfiltered dataset)
        profile_json = self.tool_map["profile_dataset"].invoke({})
        profile = self.session["profile"]

        # Query understanding (reuse existing prompt logic)
        from tinyts.nodes.query_understanding import (
            QUERY_PROMPT_TEMPLATE, _extract_json, _infer_steps_per_day,
        )
        from langchain_core.prompts import ChatPromptTemplate

        spd = _infer_steps_per_day(profile.inferred_frequency)

        # Build enhanced context for column resolution
        all_columns_info = "; ".join(
            f"{c['name']} ({c['dtype']}, {c['unique_count']} unique)"
            for c in profile.columns[:30]
        )
        cat_vals_str = json.dumps(
            profile.categorical_values
            if profile.categorical_values else {}
        )

        prompt = ChatPromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)
        msgs = prompt.format_messages(
            n_rows=profile.shape[0],
            n_cols=profile.shape[1],
            time_column=self.time_column,
            default_target_column=self.target_column,
            numeric_columns=", ".join(profile.numeric_columns[:15]),
            categorical_columns=", ".join(profile.categorical_columns[:10]),
            categorical_values=cat_vals_str,
            all_columns_info=all_columns_info,
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

        # Try native structured output first (avoids fragile JSON parsing)
        try:
            structured_llm = llm.with_structured_output(UserTaskPlan)
            task_plan = structured_llm.invoke(msgs)
            if task_plan and not task_plan.target_column:
                task_plan.target_column = self.target_column
            logger.info("Query parsed via structured output")
        except Exception as e:
            logger.info(f"Structured output unavailable, falling back to JSON parsing: {e}")

        # Fallback: free-text JSON parsing with retry
        if task_plan is None:
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
    # Re-initialization with LLM-resolved plan
    # ------------------------------------------------------------------
    def reinitialize_with_plan(self, plan: UserTaskPlan):
        """Re-create tools with LLM-resolved columns and filters.

        Called after plan() returns and user approves the plan.
        Validates resolved columns, then re-loads data with filters applied.
        """
        import pandas as pd

        # Validate against actual dataset columns
        df_peek = pd.read_csv(self.dataset_path, nrows=1)
        all_cols = list(df_peek.columns)

        # Validate target_column
        if plan.target_column and plan.target_column in all_cols:
            self.target_column = plan.target_column
        else:
            if plan.target_column and plan.target_column not in all_cols:
                logger.warning(
                    f"LLM-resolved target '{plan.target_column}' not found, "
                    f"keeping '{self.target_column}'"
                )
            plan.target_column = self.target_column

        # Validate feature_columns
        valid_features = [c for c in (plan.feature_columns or []) if c in all_cols]
        if len(valid_features) < len(plan.feature_columns or []):
            invalid = set(plan.feature_columns or []) - set(valid_features)
            logger.warning(f"Removing invalid feature columns: {invalid}")
        plan.feature_columns = valid_features
        self.feature_columns = valid_features

        # Validate filter columns
        valid_filters = [
            f for f in (plan.data_filters or [])
            if f.get("column") in all_cols
        ]
        if len(valid_filters) < len(plan.data_filters or []):
            logger.warning("Some filter columns not found in dataset, removing invalid filters")
        plan.data_filters = valid_filters

        # Re-create tools with resolved config
        self.tools, self.session = create_agent_tools(
            self.dataset_path,
            self.time_column,
            self.target_column,
            self.output_dir,
            self.feature_columns,
            data_filters=plan.data_filters if plan.data_filters else None,
        )
        self.tool_map = {t.name: t for t in self.tools}
        logger.info(
            f"Agent reinitialized: target={self.target_column}, "
            f"features={self.feature_columns}, filters={plan.data_filters}"
        )

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
        # Build task-specific system prompt
        system_prompt = self._build_system_prompt(plan)
        plan_text = self._plan_to_prompt(plan)

        llm = get_llm(temperature=settings.routing_temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content=system_prompt),
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

            # --- Format adapter: Cerebras sometimes outputs tool calls as text ---
            tool_calls = response.tool_calls
            fake_id = None

            if not tool_calls and response.content:
                parsed = self._parse_text_tool_call(response.content)
                if parsed:
                    fake_id = f"text_{step}"
                    tool_calls = [{"name": parsed["name"], "args": parsed["args"], "id": fake_id}]
                    # Replace response with a proper AIMessage containing the tool call
                    response = AIMessage(
                        content="",
                        tool_calls=[{"name": parsed["name"], "args": parsed["args"], "id": fake_id}],
                    )

            messages.append(response)

            if not tool_calls:
                logger.info(f"Agent done in {step+1} steps")
                break

            for tc in tool_calls:
                name, args, tid = tc["name"], tc["args"], tc["id"]
                logger.info(f"  Step {step}: {name}({args})")

                fn = self.tool_map.get(name)
                if fn is None:
                    res = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    validation_error = self._validate_tool_args(name, args)
                    if validation_error:
                        res = json.dumps({"error": validation_error})
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
    # ------------------------------------------------------------------
    # Text-to-tool-call adapter
    # ------------------------------------------------------------------
    _TOOL_RE = re.compile(
        r'\b(profile_dataset|train_forecast_model|train_and_explain_forecast|'
        r'combine_forecasts|detect_anomalies|explain_anomalies|generate_report|'
        r'counterfactual_forward|counterfactual_inverse|select_ensemble_strategy'
        r')\s*\(([^)]*)\)'
    )
    _ARG_RE = re.compile(
        r"(\w+)\s*=\s*("
        r"'[^']*'"
        r'|"[^"]*"'
        r"|[^,)]+?"
        r")\s*(?:,|$)"
    )

    def _parse_text_tool_call(self, text: str) -> Optional[dict]:
        """Extract the first tool call from LLM text output."""
        m = self._TOOL_RE.search(text)
        if not m or m.group(1) not in self.tool_map:
            return None
        name = m.group(1)
        raw_args = m.group(2).strip()
        if not raw_args:
            return {"name": name, "args": {}}
        args = {}
        for am in self._ARG_RE.finditer(raw_args):
            k = am.group(1)
            v = am.group(2).strip()
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1]
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
            args[k] = v
        logger.info(f"  Parsed text tool call: {name}({args})")
        return {"name": name, "args": args}

    def _validate_tool_args(self, tool_name: str, args: dict) -> Optional[str]:
        """Validate tool arguments and return error message if invalid."""
        required_args = {
            "train_forecast_model": ["model_name", "horizon"],
            "train_and_explain_forecast": ["model_name", "horizon"],
            "counterfactual_forward": ["changes_json", "horizon"],
            "counterfactual_inverse": ["target_value", "constraints_json"],
        }

        if tool_name not in required_args:
            return None  # No validation needed

        missing = [arg for arg in required_args[tool_name] if arg not in args or args[arg] is None]
        if missing:
            return f"{tool_name} requires arguments: {required_args[tool_name]}. Missing: {missing}. Example: {tool_name}({', '.join(f'{k}=<value>' for k in required_args[tool_name])})"

        return None

    def _build_system_prompt(self, plan: UserTaskPlan) -> str:
        """Build task-specific system prompt."""
        models_str = ", ".join(plan.models_included)

        if plan.counterfactual_type == "forward":
            changes_str = json.dumps(plan.counterfactual_changes)
            workflow = COUNTERFACTUAL_FORWARD_WORKFLOW.format(
                models=models_str, horizon=plan.horizon,
                changes_json=changes_str,
            )
        elif plan.counterfactual_type == "inverse":
            constraints_str = json.dumps(plan.counterfactual_constraints) if plan.counterfactual_constraints else "{}"
            workflow = COUNTERFACTUAL_INVERSE_WORKFLOW.format(
                models=models_str, horizon=plan.horizon,
                target=plan.counterfactual_target_value,
                constraints_json=constraints_str,
            )
        elif plan.task_type == "anomaly":
            workflow = ANOMALY_EXPLAIN_WORKFLOW if plan.needs_explanation else ANOMALY_WORKFLOW
        else:
            if plan.needs_explanation:
                workflow = FORECAST_EXPLAIN_WORKFLOW.format(
                    models=models_str, horizon=plan.horizon,
                )
            else:
                workflow = FORECAST_WORKFLOW.format(
                    models=models_str,
                    horizon=plan.horizon,
                )

        return BASE_PROMPT + "\n" + workflow

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

        if plan.counterfactual_type == "forward":
            lines.append(f"- COUNTERFACTUAL FORWARD: changes={json.dumps(plan.counterfactual_changes)}")
            lines.append("  → Train multivariate models first, then call counterfactual_forward()")
        elif plan.counterfactual_type == "inverse":
            lines.append(f"- COUNTERFACTUAL INVERSE: target_value={plan.counterfactual_target_value}")
            if plan.counterfactual_constraints:
                lines.append(f"  constraints={json.dumps(plan.counterfactual_constraints)}")
            lines.append("  → Train multivariate models with explain first, then call counterfactual_inverse()")

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
