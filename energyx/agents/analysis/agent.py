"""Analysis Agent — heir to TinyTSAgent.

Preserves verbatim:
  - plan → approve → execute two-phase pattern
  - LangChain @tool JSON-in/JSON-out contract
  - Rolling-origin CV, 3 folds, max(50, 2*horizon) min training size
  - Inverse-MAPE ensemble weighting in combine_forecasts
  - Cerebras text-mode tool-call format adapter
  - Session-scoped shared metrics cache (_shared_explain)
  - Two-temperature LLM pattern (routing 0.1 / synthesis 0.7)

Changes from TinyTSAgent:
  - detect_anomalies REMOVED (moved to shared monitor core)
  - explain_anomalies RETAINED (triggered by monitoring events)
  - 10 new tools added (see new_tools.py)
  - BASE_PROMPT updated (no detect_anomalies reference)
"""

from __future__ import annotations

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
from energyx.agents.analysis.new_tools import create_analysis_extensions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts (detect_anomalies removed from TOOLS list)
# ---------------------------------------------------------------------------

BASE_PROMPT = """You are EnergyX Analysis Agent, an expert energy time-series agent.

PROTOCOL:
- Call ONE tool per response. No text, just the tool call.
- Wait for the result, then decide the next tool.
- After ALL tools are done, provide your analysis.

TOOLS (core):
- profile_dataset() — Profile dataset (call first)
- train_forecast_model(model_name, horizon) — Train one model
- train_and_explain_forecast(model_name, horizon) — Train + SHAP/FI
- combine_forecasts() — Combine trained models (call after training all)
- explain_anomalies() — Explain anomalies surfaced by monitoring events
- generate_report() — Generate report (call last)
- counterfactual_forward(changes_json, horizon) — What-if forecast
- counterfactual_inverse(target_value, constraints_json) — Reach target

TOOLS (new — energy-specific):
- predict_bill(horizon) — Forecast × active tariff → projected £ bill
- counterfactual_bill_forward(changes_json, horizon) — Bill-denominated what-if
- counterfactual_bill_inverse(target_delta_gbp, constraints_json) — Reduce bill by £X
- evaluate_tariff_switch(alternative_tariff_id) — Replay history on alternative tariff
- causal_attribution(period_a_start, period_a_end, period_b_start, period_b_end) — Decompose Δ
- schedule_flexible_loads(loads_json, horizon) — Optimal scheduling → HA JSON
- suggest_budget_corrections(monthly_cap_gbp) — Ranked corrective actions
- project_longhorizon(horizon_days) — Weather-normalised multi-year projection
- compute_elasticity() — GAM/boosted-tree partial dependence (scheduled/cached)
- update_degradation_baselines() — Nightly baseline refresh (scheduled)

MODELS: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS (univariate) | RandomForest, LightGBM (multivariate)

RESILIENCE:
- If a tool returns "status": "warning", read the warnings and factor them into your analysis.
- If a tool returns an error, retry once. If it fails again, skip it and continue.
- Report any skipped or failed tools in your final output.
"""

FORECAST_WORKFLOW = """
GOAL: Forecast the target variable for {horizon} steps.
Models to evaluate: {models}
Train each model, combine results into an ensemble, then present your analysis.

OUTPUT REQUIREMENTS:
1. A markdown table of ALL predicted values from the ensemble (step | value)
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
- Feature drivers: cite exact FI% and SHAP%
- Lag structure: interpret as real-world patterns
- Decomposition: trend direction, seasonal strength
- Correlations: direct drivers vs non-linear effects
Write for both experts (cite numbers) and non-experts (plain language).
"""

ANOMALY_EXPLAIN_WORKFLOW = """
GOAL: Explain the anomalies that have been detected by the monitoring system.
Call explain_anomalies() then provide grounded causal interpretation.

OUTPUT REQUIREMENTS:
Grounded analysis. For EACH metric, cite the value AND infer what it means.
For each anomaly snapshot write: "Anomaly at index X: target dropped from A to B
while [feature] also dropped from C to D, suggesting [cause]."
"""

BILL_FORECAST_WORKFLOW = """
GOAL: Predict the upcoming energy bill for {horizon} steps.
Models to evaluate: {models}
Train models, combine forecasts, then call predict_bill() to project cost.
Include tariff structure in your explanation.
"""

COUNTERFACTUAL_FORWARD_WORKFLOW = """
GOAL: Evaluate a what-if scenario for {horizon} steps.
Feature changes: {changes_json}  Models: {models}
Train multivariate models, then run counterfactual_forward() and optionally counterfactual_bill_forward().

OUTPUT: Baseline vs counterfactual forecast + bill impact. Cite model MAPE.
"""

COUNTERFACTUAL_INVERSE_WORKFLOW = """
GOAL: Find what changes are needed to reach target value {target} over {horizon} steps.
Constraints: {constraints_json}  Models: {models}
Train models with explain, then run counterfactual_inverse() or counterfactual_bill_inverse().

OUTPUT: Feature recommendations, feasibility, expected £ saving if applicable.
"""

CAUSAL_ATTRIBUTION_WORKFLOW = """
GOAL: Decompose the consumption change between two periods.
Call causal_attribution() with the two date ranges, then explain the drivers.

OUTPUT: % attribution to weather, occupancy, behavioural, degradation factors.
"""

SCHEDULING_WORKFLOW = """
GOAL: Find optimal run times for flexible loads.
Loads: {loads_json}  Horizon: {horizon}
Call schedule_flexible_loads() to get HA service-call JSON.

OUTPUT: Schedule with justification (tariff windows, carbon intensity, occupancy).
In offline mode this is advisory only — no action will be dispatched.
"""


class AnalysisAgent:
    """Analysis Agent — the LLM agentic loop for time-series analysis.

    Preserves the TinyTSAgent plan/execute structure verbatim.
    detect_anomalies removed; 10 new tools added via create_analysis_extensions().
    """

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

        # Core tools from tinyts (preserved verbatim, minus detect_anomalies)
        all_tools, self.session = create_agent_tools(
            dataset_path, time_column, target_column, output_dir, feature_columns,
        )
        # Remove detect_anomalies — it now lives in shared monitor core
        core_tools = [t for t in all_tools if t.name != "detect_anomalies"]

        # New energy-specific tools
        ext_tools = create_analysis_extensions(self.session, dataset_path, time_column, target_column, feature_columns)

        self.tools = core_tools + ext_tools
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Phase 1: Plan (preserved verbatim from TinyTSAgent)
    # ------------------------------------------------------------------

    def plan(self, query: str) -> UserTaskPlan:
        profile_json = self.tool_map["profile_dataset"].invoke({})
        profile = self.session["profile"]

        from tinyts.nodes.query_understanding import (
            QUERY_PROMPT_TEMPLATE, _extract_json, _infer_steps_per_day,
        )
        from langchain_core.prompts import ChatPromptTemplate

        spd = _infer_steps_per_day(profile.inferred_frequency)
        all_columns_info = "; ".join(
            f"{c['name']} ({c['dtype']}, {c['unique_count']} unique)"
            for c in profile.columns[:30]
        )
        cat_vals_str = json.dumps(profile.categorical_values if profile.categorical_values else {})

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

        try:
            structured_llm = llm.with_structured_output(UserTaskPlan)
            task_plan = structured_llm.invoke(msgs)
            if task_plan and not task_plan.target_column:
                task_plan.target_column = self.target_column
        except Exception as e:
            logger.info(f"Structured output unavailable: {e}")

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

    def reinitialize_with_plan(self, plan: UserTaskPlan):
        """Re-create tools with LLM-resolved columns and filters (preserved verbatim)."""
        import pandas as pd
        df_peek = pd.read_csv(self.dataset_path, nrows=1)
        all_cols = list(df_peek.columns)

        if plan.target_column and plan.target_column in all_cols:
            self.target_column = plan.target_column
        else:
            if plan.target_column and plan.target_column not in all_cols:
                logger.warning(f"LLM target '{plan.target_column}' not found, keeping '{self.target_column}'")
            plan.target_column = self.target_column

        valid_features = [c for c in (plan.feature_columns or []) if c in all_cols]
        plan.feature_columns = valid_features
        self.feature_columns = valid_features

        valid_filters = [f for f in (plan.data_filters or []) if f.get("column") in all_cols]
        plan.data_filters = valid_filters

        all_tools, self.session = create_agent_tools(
            self.dataset_path, self.time_column, self.target_column,
            self.output_dir, self.feature_columns,
            data_filters=plan.data_filters if plan.data_filters else None,
        )
        core_tools = [t for t in all_tools if t.name != "detect_anomalies"]
        ext_tools = create_analysis_extensions(
            self.session, self.dataset_path, self.time_column,
            self.target_column, self.feature_columns,
        )
        self.tools = core_tools + ext_tools
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Phase 2: Execute (preserved verbatim from TinyTSAgent)
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: UserTaskPlan,
        on_tool_call: Optional[Callable] = None,
        max_steps: int = 25,
    ) -> Dict[str, Any]:
        system_prompt = self._build_system_prompt(plan)
        plan_text = self._plan_to_prompt(plan)

        llm = get_llm(temperature=settings.routing_temperature)
        llm_with_tools = llm.bind_tools(self.tools)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=plan_text),
        ]

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

            tool_calls = response.tool_calls
            if not tool_calls and response.content:
                parsed = self._parse_text_tool_call(response.content)
                if parsed:
                    fid = f"text_{step}"
                    tool_calls = [{"name": parsed["name"], "args": parsed["args"], "id": fid}]
                    response = AIMessage(
                        content="",
                        tool_calls=[{"name": parsed["name"], "args": parsed["args"], "id": fid}],
                    )

            messages.append(response)

            if not tool_calls:
                logger.info(f"Agent done in {step+1} steps")
                break

            for tc in tool_calls:
                name, args, tid = tc["name"], tc["args"], tc["id"]
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
    # Helpers (preserved verbatim from TinyTSAgent)
    # ------------------------------------------------------------------

    # Format adapter — updated to include new tools; detect_anomalies removed
    _TOOL_RE = re.compile(
        r'\b(profile_dataset|train_forecast_model|train_and_explain_forecast|'
        r'combine_forecasts|explain_anomalies|generate_report|'
        r'counterfactual_forward|counterfactual_inverse|select_ensemble_strategy|'
        r'predict_bill|counterfactual_bill_forward|counterfactual_bill_inverse|'
        r'evaluate_tariff_switch|causal_attribution|schedule_flexible_loads|'
        r'suggest_budget_corrections|project_longhorizon|compute_elasticity|'
        r'update_degradation_baselines'
        r')\s*\(([^)]*)\)'
    )
    _ARG_RE = re.compile(
        r"(\w+)\s*=\s*('(?:[^']*)'|\"(?:[^\"]*)\"|[^,)]+?)\s*(?:,|$)"
    )

    def _parse_text_tool_call(self, text: str) -> Optional[dict]:
        m = self._TOOL_RE.search(text)
        if not m or m.group(1) not in self.tool_map:
            return None
        name = m.group(1)
        raw = m.group(2).strip()
        if not raw:
            return {"name": name, "args": {}}
        args = {}
        for am in self._ARG_RE.finditer(raw):
            k, v = am.group(1), am.group(2).strip()
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
        return {"name": name, "args": args}

    def _validate_tool_args(self, tool_name: str, args: dict) -> Optional[str]:
        required = {
            "train_forecast_model": ["model_name", "horizon"],
            "train_and_explain_forecast": ["model_name", "horizon"],
            "counterfactual_forward": ["changes_json", "horizon"],
            "counterfactual_inverse": ["target_value", "constraints_json"],
        }
        if tool_name not in required:
            return None
        missing = [a for a in required[tool_name] if a not in args or args[a] is None]
        if missing:
            return (f"{tool_name} requires: {required[tool_name]}. "
                    f"Missing: {missing}.")
        return None

    def _build_system_prompt(self, plan: UserTaskPlan) -> str:
        models_str = ", ".join(plan.models_included)
        if plan.counterfactual_type == "forward":
            changes_str = json.dumps(plan.counterfactual_changes)
            workflow = COUNTERFACTUAL_FORWARD_WORKFLOW.format(
                models=models_str, horizon=plan.horizon, changes_json=changes_str,
            )
        elif plan.counterfactual_type == "inverse":
            constraints_str = json.dumps(plan.counterfactual_constraints) if plan.counterfactual_constraints else "{}"
            workflow = COUNTERFACTUAL_INVERSE_WORKFLOW.format(
                models=models_str, horizon=plan.horizon,
                target=plan.counterfactual_target_value, constraints_json=constraints_str,
            )
        elif plan.task_type == "anomaly":
            workflow = ANOMALY_EXPLAIN_WORKFLOW
        elif getattr(plan, "task_type", "") == "bill":
            workflow = BILL_FORECAST_WORKFLOW.format(models=models_str, horizon=plan.horizon)
        else:
            if plan.needs_explanation:
                workflow = FORECAST_EXPLAIN_WORKFLOW.format(models=models_str, horizon=plan.horizon)
            else:
                workflow = FORECAST_WORKFLOW.format(models=models_str, horizon=plan.horizon)
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
        elif plan.counterfactual_type == "inverse":
            lines.append(f"- COUNTERFACTUAL INVERSE: target_value={plan.counterfactual_target_value}")
        lines.append("")
        lines.append("The dataset is already profiled. Start executing now.")
        return "\n".join(lines)

    def _fallback_plan(self, query, profile, steps_per_day) -> UserTaskPlan:
        from tinyts.nodes.query_understanding import QueryUnderstandingNode
        return QueryUnderstandingNode()._create_fallback_plan(query, profile, steps_per_day)

    def _make_result(self, response_text, messages):
        return {
            "response": response_text,
            "session": self.session,
            "messages": messages,
            "output_dir": self.output_dir,
        }
