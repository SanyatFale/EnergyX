"""Analysis Agent.

14-tool redesign:
  forecast_univariate, forecast_multivariate, explain_forecast,
  detect_anomalies, explain_anomalies, counterfactual_fwd, counterfactual_inv,
  get_cost_analysis, get_consumption_analysis, make_budget,
  weather_impact_analysis, suggest_budget_correction,
  evaluate_tariff_switch, generate_report

Internal tinyts tools are passed into create_analysis_extensions but
NOT exposed to the LLM.

HIL tools (require human parameter input before/during execution):
  forecast_univariate, forecast_multivariate, detect_anomalies, make_budget
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tinyts.agent_tools import create_agent_tools
from tinyts.config import get_llm, settings
from tinyts.state import UserTaskPlan
from energyx.agents.analysis.new_tools import create_analysis_extensions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

BASE_PROMPT = """You are EnergyX Analysis Agent — an expert home energy analyst.

PROTOCOL:
- Call ONE tool per response. No text, just the tool call.
- Wait for the result before calling the next tool.
- After ALL tools complete, provide your analysis in plain English.

TOOLS:
  forecast_univariate(horizon, models)       — Best univariate model by MAPE
  forecast_multivariate(horizon, models)     — Best multivariate model with features
  explain_forecast(model_name)               — SHAP, lags, STL, causal attribution
  detect_anomalies(methods, level)           — 7-method anomaly ensemble
  explain_anomalies()                        — Z-score, IQR, STL residuals for anomalies
  counterfactual_fwd(changes_json, horizon, level)  — What-if scenario
  counterfactual_inv(target, level, constraints_json) — Reach a target value
  get_cost_analysis()                        — Historical cost breakdown by tariff
  get_consumption_analysis(level)            — Patterns at home/room/appliance/all level
  make_budget(budget_gbp)                    — Budget analysis + suggestions
  weather_impact_analysis()                  — Weather↔consumption correlation
  suggest_budget_correction()                — Appliance actions to hit budget
  evaluate_tariff_switch(alternative_tariff_id) — Compare tariffs
  generate_report()                          — Write markdown report (call last)

RESILIENCE:
- On "status":"warning", read warnings and factor into analysis.
- On error, retry once. If it fails again, skip and report it.
"""

FORECAST_WORKFLOW = """
GOAL: Forecast electricity consumption for {horizon} steps.

STEPS:
1. Call forecast_{mode}(horizon={horizon}, models="{models}")
2. Call explain_forecast() to understand what drives the forecast.
3. Call generate_report() last.

OUTPUT: Forecast table (step | predicted W), model comparison, top 3 causal drivers.
"""

ANOMALY_WORKFLOW = """
GOAL: Detect and explain anomalies in home energy data.

STEPS:
1. Call detect_anomalies(methods="{methods}", level="{level}")
2. Call explain_anomalies() immediately after.
3. Optionally call generate_report() last.

OUTPUT: Number of anomalies, normal range, top 5 anomalous events with timestamp
+ likely cause. Recommend: data quality issue, behavioural pattern, or equipment fault?
"""

COUNTERFACTUAL_FWD_WORKFLOW = """
GOAL: Estimate impact of scenario: {changes_desc} over {horizon} steps.

STEPS:
1. Call counterfactual_fwd(changes_json='{changes_json}', horizon={horizon}, level="{level}")

OUTPUT: Baseline vs scenario — watts change + bill change (if level=bill). Cite model MAPE.
"""

COUNTERFACTUAL_INV_WORKFLOW = """
GOAL: Find what changes are needed to reach target {target} ({level} level).

STEPS:
1. Call counterfactual_inv(target={target}, level="{level}", constraints_json="{constraints}")

OUTPUT: Recommended feature changes, feasibility, expected saving.
"""

COST_WORKFLOW = """
GOAL: Analyse historical electricity cost.

STEPS:
1. Call get_cost_analysis() — returns daily/monthly/annual cost breakdown.

OUTPUT: Cost table by period, tariff details, appliance cost breakdown.
"""

CONSUMPTION_WORKFLOW = """
GOAL: Analyse consumption patterns at {level} level.

STEPS:
1. Call get_consumption_analysis(level="{level}")

OUTPUT: Statistical summary, hourly/daily patterns, natural-language narrative.
"""

BUDGET_WORKFLOW = """
GOAL: Create or check household energy budget.

STEPS:
1. Call make_budget() — phase 1: shows consumption analysis, awaits user input.
2. After user sets budget: call make_budget(budget_gbp=<user_amount>)
3. Call suggest_budget_correction() for appliance-level actions.

OUTPUT: Overshoot/undershoot vs budget, ranked appliance actions with savings estimates.
"""

WEATHER_WORKFLOW = """
GOAL: Quantify weather impact on home energy consumption.

STEPS:
1. Call weather_impact_analysis()

OUTPUT: Correlation table, regression slopes (W/°C), per-temperature-band consumption,
appliance-level weather correlations.
"""

TARIFF_WORKFLOW = """
GOAL: Compare current tariff against alternative.

STEPS:
1. Call evaluate_tariff_switch(alternative_tariff_id="{tariff_id}")

OUTPUT: Annual cost comparison, projected saving, recommendation.
"""


class AnalysisAgent:
    """Analysis Agent with 14-tool redesign and HIL parameter support."""

    # HIL_TOOLS: these tools benefit from user parameter input before execution
    HIL_TOOLS = frozenset(["forecast_univariate", "forecast_multivariate",
                            "detect_anomalies", "make_budget"])

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

        self.dataset_path   = dataset_path
        self.time_column    = time_column
        self.target_column  = target_column
        self.output_dir     = output_dir
        self.feature_columns = feature_columns

        # Internal tinyts tools (not shown to LLM)
        internal_list, self.session = create_agent_tools(
            dataset_path, time_column, target_column, output_dir, feature_columns,
        )
        self._internal_map = {t.name: t for t in internal_list}

        # High-level tools exposed to LLM
        self.tools = create_analysis_extensions(
            session=self.session,
            dataset_path=dataset_path,
            time_column=time_column,
            target_column=target_column,
            feature_columns=feature_columns,
            internal_tools=self._internal_map,
        )
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------

    def plan(self, query: str) -> UserTaskPlan:
        profile_json = self._internal_map["profile_dataset"].invoke({})
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
            n_rows=profile.shape[0], n_cols=profile.shape[1],
            time_column=self.time_column, default_target_column=self.target_column,
            numeric_columns=", ".join(profile.numeric_columns[:15]),
            categorical_columns=", ".join(profile.categorical_columns[:10]),
            categorical_values=cat_vals_str,
            all_columns_info=all_columns_info,
            frequency=profile.inferred_frequency or "Unknown",
            steps_per_day=spd,
            date_range=f"{profile.date_range[0]} to {profile.date_range[1]}",
            has_trend=profile.has_trend, has_seasonality=profile.has_seasonality,
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
        """Re-create tools with LLM-resolved columns and filters."""
        import pandas as pd
        df_peek = pd.read_csv(self.dataset_path, nrows=1)
        all_cols = list(df_peek.columns)

        if plan.target_column and plan.target_column in all_cols:
            self.target_column = plan.target_column
        else:
            plan.target_column = self.target_column

        valid_features = [c for c in (plan.feature_columns or []) if c in all_cols]
        plan.feature_columns = valid_features
        self.feature_columns = valid_features

        valid_filters = [f for f in (plan.data_filters or []) if f.get("column") in all_cols]
        plan.data_filters = valid_filters

        internal_list, self.session = create_agent_tools(
            self.dataset_path, self.time_column, self.target_column,
            self.output_dir, self.feature_columns,
            data_filters=plan.data_filters if plan.data_filters else None,
        )
        self._internal_map = {t.name: t for t in internal_list}
        self.tools = create_analysis_extensions(
            session=self.session,
            dataset_path=self.dataset_path,
            time_column=self.time_column,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            internal_tools=self._internal_map,
        )
        self.tool_map = {t.name: t for t in self.tools}

    # ------------------------------------------------------------------
    # Phase 2: Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        plan: UserTaskPlan,
        on_tool_call: Optional[Callable] = None,
        max_steps: int = 20,
        hil_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the plan. hil_params injects user-confirmed parameters from HIL forms."""

        # Short-circuit for single-tool queries
        direct = self._try_direct_call(plan, on_tool_call, hil_params)
        if direct is not None:
            return direct

        system_prompt = self._build_system_prompt(plan, hil_params)
        plan_text     = self._plan_to_prompt(plan, hil_params)

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
            clean_content = self._strip_thinking(response.content or "")
            if not tool_calls and clean_content:
                parsed = self._parse_text_tool_call(clean_content)
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

                # Inject HIL params for matching tool
                if hil_params and name in hil_params:
                    args = {**args, **hil_params[name]}

                fn = self.tool_map.get(name)
                res = (json.dumps({"error": f"Unknown tool: {name}"}) if fn is None
                       else self._safe_invoke(fn, args))

                messages.append(ToolMessage(content=res, tool_call_id=tid))
                if on_tool_call:
                    on_tool_call(name, args, res)

        final = ""
        if messages and hasattr(messages[-1], "content"):
            final = self._strip_thinking(messages[-1].content or "")

        return self._make_result(final, messages)

    def _safe_invoke(self, fn, args: dict) -> str:
        try:
            return fn.invoke(args)
        except Exception as e:
            return json.dumps({"error": f"{fn.name} failed: {e}"})

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    _FORECAST_RE   = re.compile(r'\b(forecast|predict|predict\w*|project|next \d+|future)\b', re.I)
    _ANOMALY_RE    = re.compile(r'\b(anomal|unusual|spike|outlier|fault|weird|strange|detect)\b', re.I)
    _COST_RE       = re.compile(r'\b(cost|bill|spend|tariff|£|price|rate|how much)\b', re.I)
    _BUDGET_RE     = re.compile(r'\b(budget|monthly cap|on track|afford)\b', re.I)
    _CONSUMPTION_RE= re.compile(r'\b(consumption|usage|pattern|how much.*use|appliance|room|breakdown)\b', re.I)
    _WEATHER_RE    = re.compile(r'\b(weather|temperature.*impact|outdoor|does.*cold|weather.*energy)\b', re.I)
    _CF_FWD_RE     = re.compile(r'\b(what if|if.*\d+%|heating.*\+|scenario|increase|decrease.*consump)\b', re.I)
    _CF_INV_RE     = re.compile(r'\b(reduce.*bill|reach.*target|get.*to \d+|target.*watt|save.*£\d+)\b', re.I)
    _TARIFF_RE     = re.compile(r'\b(tariff switch|compare.*tariff|alternative tariff|economy7|economy 7)\b', re.I)

    def _build_system_prompt(
        self, plan: UserTaskPlan, hil_params: Optional[Dict[str, Any]] = None
    ) -> str:
        q       = plan.user_query or ""
        hil     = hil_params or {}
        horizon = hil.get("horizon", plan.horizon or 24)
        models  = hil.get("models", ", ".join(plan.models_included or []) or "all")
        methods = hil.get("methods", "all")
        level   = hil.get("level", "home")

        if self._ANOMALY_RE.search(q):
            return BASE_PROMPT + "\n" + ANOMALY_WORKFLOW.format(
                methods=methods, level=level)

        if self._BUDGET_RE.search(q):
            return BASE_PROMPT + "\n" + BUDGET_WORKFLOW

        if self._CF_INV_RE.search(q):
            target = self._extract_amount(q, 40.0)
            return BASE_PROMPT + "\n" + COUNTERFACTUAL_INV_WORKFLOW.format(
                target=target, level="bill" if "£" in q else "consumption",
                constraints="{}",
            )

        if self._CF_FWD_RE.search(q):
            changes = self._extract_changes(q)
            return BASE_PROMPT + "\n" + COUNTERFACTUAL_FWD_WORKFLOW.format(
                changes_desc=str(changes), changes_json=json.dumps(changes),
                horizon=horizon, level="bill" if "bill" in q.lower() else "consumption",
            )

        if self._TARIFF_RE.search(q):
            tariff_id = self._extract_tariff_id(q)
            return BASE_PROMPT + "\n" + TARIFF_WORKFLOW.format(tariff_id=tariff_id)

        if self._WEATHER_RE.search(q):
            return BASE_PROMPT + "\n" + WEATHER_WORKFLOW

        if self._CONSUMPTION_RE.search(q) and not self._FORECAST_RE.search(q):
            cons_level = ("appliance" if "appliance" in q.lower()
                          else "room" if "room" in q.lower() else "all")
            return BASE_PROMPT + "\n" + CONSUMPTION_WORKFLOW.format(level=cons_level)

        if self._COST_RE.search(q) and not self._FORECAST_RE.search(q):
            return BASE_PROMPT + "\n" + COST_WORKFLOW

        # Default: forecast
        mode = "multivariate" if (plan.is_multivariate and self.feature_columns) else "univariate"
        return BASE_PROMPT + "\n" + FORECAST_WORKFLOW.format(
            horizon=horizon, models=models, mode=mode,
        )

    def _plan_to_prompt(
        self, plan: UserTaskPlan, hil_params: Optional[Dict[str, Any]] = None
    ) -> str:
        hil     = hil_params or {}
        horizon = hil.get("horizon", plan.horizon or 24)
        models  = hil.get("models", ", ".join(plan.models_included or []) or "all")
        methods = hil.get("methods", "all")
        level   = hil.get("level", "home")

        lines = [
            "Execute this approved analysis plan:",
            f"- Task: {plan.task_type}",
            f"- Horizon: {horizon}",
            f"- Is multivariate: {plan.is_multivariate}",
        ]
        if plan.feature_columns:
            lines.append(f"- Feature columns: {', '.join(plan.feature_columns)}")
        if models and models != "all":
            lines.append(f"- Models: {models}")
        if methods and methods != "all":
            lines.append(f"- Anomaly methods: {methods}")
        lines.append(f"- Level: {level}")
        lines.append(f"- Needs explanation: {plan.needs_explanation}")
        lines.append("")
        if plan.counterfactual_type == "forward":
            lines.append(f"- COUNTERFACTUAL FORWARD: changes={json.dumps(plan.counterfactual_changes)}")
        elif plan.counterfactual_type == "inverse":
            lines.append(f"- COUNTERFACTUAL INVERSE: target={plan.counterfactual_target_value}")
        lines.append("The dataset is already profiled. Start executing now.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Direct call sequences (bypass LLM for deterministic queries)
    # ------------------------------------------------------------------

    def _try_direct_call(
        self,
        plan: UserTaskPlan,
        on_tool_call: Optional[Callable],
        hil_params: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        q   = plan.user_query or ""
        hil = hil_params or {}

        if self._TARIFF_RE.search(q):
            return self._direct_seq(on_tool_call, [
                ("evaluate_tariff_switch",
                 {"alternative_tariff_id": self._extract_tariff_id(q)}),
            ], self._fmt_tariff)

        if self._BUDGET_RE.search(q):
            budget = self._extract_amount(q, None)
            steps = [("make_budget", {"budget_gbp": budget} if budget else {})]
            if budget:
                steps.append(("suggest_budget_correction", {}))
            return self._direct_seq(on_tool_call, steps, self._fmt_budget)

        if self._CF_INV_RE.search(q):
            delta = self._extract_amount(q, 40.0)
            return self._direct_seq(on_tool_call, [
                ("counterfactual_inv",
                 {"target": delta, "level": "bill", "constraints_json": "{}"}),
            ], self._fmt_cf_inv)

        if self._CF_FWD_RE.search(q):
            changes = self._extract_changes(q)
            horizon = hil.get("horizon", plan.horizon or 24)
            return self._direct_seq(on_tool_call, [
                ("counterfactual_fwd",
                 {"changes_json": json.dumps(changes), "horizon": horizon,
                  "level": "bill" if "bill" in q.lower() else "consumption"}),
            ], self._fmt_cf_fwd)

        if self._ANOMALY_RE.search(q):
            return self._direct_seq(on_tool_call, [
                ("detect_anomalies",  {"methods": hil.get("methods", "all"),
                                       "level":   hil.get("level", "home")}),
                ("explain_anomalies", {}),
            ], self._fmt_anomaly)

        if self._FORECAST_RE.search(q):
            horizon   = int(hil.get("horizon", plan.horizon or 24))
            models    = hil.get("models", ", ".join(plan.models_included or []) or "all")
            is_mv     = plan.is_multivariate and bool(self.feature_columns)
            tool_name = "forecast_multivariate" if is_mv else "forecast_univariate"
            steps = [(tool_name, {"horizon": horizon, "models": models})]
            if plan.needs_explanation:
                steps.append(("explain_forecast", {"model_name": "best"}))
            if plan.needs_report:
                steps.append(("generate_report", {}))
            return self._direct_seq(on_tool_call, steps, self._fmt_forecast)

        if self._COST_RE.search(q) and not self._FORECAST_RE.search(q):
            return self._direct_seq(on_tool_call,
                                    [("get_cost_analysis", {})], self._fmt_cost)

        if self._WEATHER_RE.search(q):
            return self._direct_seq(on_tool_call,
                                    [("weather_impact_analysis", {})], self._fmt_weather)

        if (self._CONSUMPTION_RE.search(q)
                and not self._FORECAST_RE.search(q)
                and not self._ANOMALY_RE.search(q)):
            lvl = ("appliance" if "appliance" in q.lower()
                   else "room" if "room" in q.lower() else "all")
            return self._direct_seq(on_tool_call,
                                    [("get_consumption_analysis", {"level": lvl})],
                                    self._fmt_consumption)

        return None

    def _direct_seq(
        self,
        on_tool_call: Optional[Callable],
        steps: List,
        formatter,
    ) -> Dict[str, Any]:
        merged: dict = {}
        for name, args in steps:
            fn = self.tool_map.get(name)
            if fn is None:
                return self._make_result(f"Tool '{name}' not available.", [])
            raw = self._safe_invoke(fn, args)
            if on_tool_call:
                on_tool_call(name, args, raw)
            try:
                result = json.loads(raw)
                merged.update(result)  # accumulate — later tools add to, not replace
            except Exception:
                merged["raw"] = raw
        return self._make_result(formatter(merged), [])

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_forecast(d: dict) -> str:
        if d.get("error"):
            return f"Forecast error: {d['error']}"
        best    = d.get("best_model", "?")
        mape    = d.get("best_mape_pct", 0)
        preds   = d.get("predictions", [])
        horizon = d.get("horizon", len(preds))
        compare = d.get("model_comparison", {})

        lines = [f"**Best model: {best}** (MAPE = {mape:.2f}%)\n"]

        if compare and len(compare) > 1:
            lines.append("| Model | MAPE % | Std % |")
            lines.append("|---|---|---|")
            for mn, ms in sorted(compare.items(), key=lambda x: x[1].get("mape", 999)):
                lines.append(f"| {mn} | {ms.get('mape',0):.3f} | {ms.get('std',0):.3f} |")
            lines.append("")

        if preds:
            lines.append(f"**Forecast ({horizon} steps):**\n")
            lines.append("| Step | Predicted (W) |")
            lines.append("|---|---|")
            for i, v in enumerate(preds[:horizon], 1):
                lines.append(f"| {i} | {v:.1f} |")

        attr = d.get("attribution", {})
        if attr:
            lines.append("\n**Causal attribution:**")
            for k, v in attr.items():
                if v > 0:
                    lines.append(f"- {k.replace('_', ' ').title()}: {v:.1f}%")
        hints = d.get("lag_hints", [])
        if hints:
            lines.append("\n**Temporal patterns:** " + " · ".join(hints))

        return "\n".join(lines)

    @staticmethod
    def _fmt_anomaly(d: dict) -> str:
        if d.get("error"):
            return f"Anomaly detection error: {d['error']}"
        n       = d.get("n_anomalies", 0)
        total   = d.get("total_data_points", 0)
        agree   = d.get("average_agreement", 0)
        counts  = d.get("method_counts", {})
        rng     = d.get("normal_range_watts", "?")
        samples = d.get("anomaly_samples", [])
        appl    = d.get("appliance_anomalies", {})

        lines = [f"**{n} anomalies** detected in {total} data points "
                 f"(avg method agreement: {agree:.1f}).\n"]
        lines.append(f"Normal range: **{rng} W**\n")

        if counts:
            lines.append("| Method | Flags |")
            lines.append("|---|---|")
            for m, c in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {m} | {c} |")
            lines.append("")

        if samples:
            lines.append("**Top anomalous events:**")
            for s in samples[:5]:
                lines.append(
                    f"- `{s['timestamp']}` — **{s['value']:.0f} W** "
                    f"(z={s['z_score']:+.1f}, p{s['percentile']:.0f}) "
                    f"→ {s['likely_cause']}"
                )

        if appl:
            lines.append("\n**Appliance spikes:**")
            for name, info in appl.items():
                lines.append(
                    f"- {name}: {info['n_spikes']} spike(s) above {info['threshold_watts']:.0f} W "
                    f"(max {info['max_watts']:.0f} W)"
                )

        return "\n".join(lines)

    @staticmethod
    def _fmt_tariff(d: dict) -> str:
        if d.get("error"):
            avail = d.get("available_tariffs", [])
            return (f"Tariff not found. Available: {', '.join(avail)}. "
                    f"Try: evaluate_tariff_switch(alternative_tariff_id='economy7')")
        curr   = d.get("current_tariff", "current")
        alt    = d.get("alternative_tariff", "alternative")
        curr_c = d.get("current_annual_cost_gbp", 0)
        alt_c  = d.get("alternative_annual_cost_gbp", 0)
        saving = d.get("projected_annual_saving_gbp", 0)
        rec    = d.get("recommendation", "stay")
        kwh    = d.get("annual_kwh_estimate", 0)
        verdict = (f"**Switch to {alt}** — saves £{saving:.2f}/year."
                   if rec == "switch" else
                   f"**Stay on {curr}** — switching costs £{abs(saving):.2f}/year more.")
        return (
            f"Based on ~{kwh:.0f} kWh/year estimated consumption:\n\n"
            f"| Tariff | Annual cost |\n|---|---|\n"
            f"| {curr} (current) | £{curr_c:.2f} |\n"
            f"| {alt} | £{alt_c:.2f} |\n\n{verdict}"
        )

    @staticmethod
    def _fmt_budget(d: dict) -> str:
        if d.get("error"):
            return f"Budget error: {d['error']}"
        if d.get("phase") == 1 or d.get("awaiting_budget_input"):
            monthly = d.get("current_monthly_total_gbp", 0)
            yearly  = d.get("current_yearly_estimate_gbp", 0)
            return (
                f"**Current spend**: £{monthly:.2f}/month (£{yearly:.2f}/year estimated).\n\n"
                f"Appliance breakdown saved. Please enter your monthly budget target."
            )
        proj   = d.get("projected_monthly_gbp", 0)
        cap    = d.get("monthly_budget_gbp", 0)
        over   = d.get("overshoot_gbp", 0)
        header = (f"**On track.** Projected: £{proj:.2f} vs budget £{cap:.2f}/month."
                  if over <= 0 else
                  f"**Over budget** by £{over:.2f}. Projected: £{proj:.2f}, budget: £{cap:.2f}/month.")
        lines = [header]
        for s in d.get("suggestions", []):
            lines.append(f"- **{s.get('action', '')}** — save ~£{s.get('potential_saving_gbp', 0):.2f}/month")
        return "\n".join(lines)

    @staticmethod
    def _fmt_cost(d: dict) -> str:
        if d.get("error"):
            return f"Cost analysis error: {d['error']}"
        return (
            f"**Cost breakdown** ({d.get('tariff', 'standard')} tariff, "
            f"{d.get('unit_rate_gbp_per_kwh', 0):.4f} £/kWh):\n\n"
            f"| Period | Energy | Standing | Total |\n|---|---|---|---|\n"
            f"| Daily | £{d.get('daily_cost_gbp',0):.4f} | £{d.get('standing_charge_gbp_per_day',0):.4f} "
            f"| £{d.get('daily_cost_gbp',0)+d.get('standing_charge_gbp_per_day',0):.4f} |\n"
            f"| Monthly | £{d.get('monthly_energy_cost_gbp',0):.4f} | "
            f"£{d.get('monthly_standing_charge_gbp',0):.4f} | **£{d.get('monthly_total_gbp',0):.4f}** |\n"
            f"| Annual est. | — | — | **£{d.get('annual_estimate_gbp',0):.2f}** |\n\n"
            + (("\n**Appliance costs:**\n" +
                "\n".join(f"- {a['appliance']}: £{a['monthly_cost_gbp']:.4f}/month "
                          f"({a['pct_of_total']:.1f}%)"
                          for a in d.get("appliance_cost_breakdown", [])[:5]))
               if d.get("appliance_cost_breakdown") else "")
        )

    @staticmethod
    def _fmt_weather(d: dict) -> str:
        if d.get("error"):
            return f"Weather analysis error: {d['error']}"
        drivers = d.get("weather_drivers", [])
        lines = ["**Weather impact on energy consumption:**\n"]
        for drv in drivers:
            corr  = drv.get("pearson_corr_with_electricity", 0)
            slope = drv.get("regression_slope_watts_per_unit", 0)
            lines.append(
                f"- **{drv['label']}**: corr={corr:+.3f}, "
                f"{slope:+.1f} W per unit change\n  {drv['interpretation']}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fmt_consumption(d: dict) -> str:
        if d.get("error"):
            return f"Consumption analysis error: {d['error']}"
        parts = []
        if d.get("home"):
            h = d["home"]
            parts.append(
                f"**Home**: {h.get('mean_watts',0):.0f}W avg, "
                f"peak at {h.get('peak_hour',0)}:00, "
                f"{h.get('monthly_kwh_estimate',0):.0f} kWh/month.\n"
                f"{h.get('narrative','')}"
            )
        if d.get("appliance", {}).get("appliances"):
            appl = d["appliance"]["appliances"]
            parts.append("**Top appliances by cost:**\n" +
                         "\n".join(f"- {a['appliance']}: "
                                   f"£{a['monthly_cost_gbp']:.4f}/month "
                                   f"({a['pct_of_home']:.1f}% of home)"
                                   for a in appl[:5]))
        if d.get("room", {}).get("rooms"):
            rooms = d["room"]["rooms"]
            parts.append("**Room comfort:**\n" +
                         "\n".join(f"- {r['sensor']}: {r['mean_temp_c']}°C avg, "
                                   f"{r['in_comfort_band_pct']:.0f}% in 18–22°C band"
                                   for r in rooms[:4]))
        return "\n\n".join(parts) if parts else "No consumption data available."

    @staticmethod
    def _fmt_cf_fwd(d: dict) -> str:
        if d.get("error"):
            return f"Counterfactual error: {d['error']}"
        base  = d.get("baseline_mean_watts", 0)
        cf    = d.get("counterfactual_mean_watts", 0)
        delta = d.get("mean_impact_watts", cf - base)
        lines = [
            f"**Scenario**: {d.get('changes_applied', {})}",
            f"Baseline: {base:.0f} W avg → Counterfactual: {cf:.0f} W avg "
            f"(**{delta:+.0f} W, {delta/base*100:+.1f}%**)" if base > 0 else "",
        ]
        if d.get("baseline_bill_gbp"):
            lines.append(
                f"Bill impact: £{d.get('baseline_bill_gbp',0):.4f} → "
                f"£{d.get('counterfactual_bill_gbp',0):.4f} "
                f"(**{d.get('bill_change_gbp',0):+.4f}** / "
                f"{d.get('pct_change',0):+.1f}%)"
            )
        return "\n".join(l for l in lines if l)

    @staticmethod
    def _fmt_cf_inv(d: dict) -> str:
        if d.get("error"):
            return f"Counterfactual inverse error: {d['error']}"
        recs = d.get("feature_recommendations", {})
        lines = [
            f"To reach target **{d.get('target')}** "
            f"(currently {d.get('current_predicted', '?')} → optimised to {d.get('optimized_predicted', '?')}):",
        ]
        for feat, info in recs.items():
            lines.append(
                f"- **{feat}**: {info['current']} → {info['recommended']} "
                f"({info['change']:+.2f}, {info['change_pct']:+.1f}%)"
            )
        if not d.get("converged"):
            lines.append("⚠ Optimisation did not fully converge — these are approximate recommendations.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_amount(query: str, default) -> Optional[float]:
        m = re.search(r'£\s*(\d+(?:\.\d+)?)', query)
        if m:
            return float(m.group(1))
        m = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:pounds?|gbp)\b', query, re.I)
        return float(m.group(1)) if m else default

    @staticmethod
    def _extract_changes(query: str) -> dict:
        changes: dict = {}
        for m in re.finditer(r'(\w+)\s*([+-]?\d+(?:\.\d+)?)\s*%', query, re.I):
            changes[m.group(1).lower()] = float(m.group(2)) / 100.0
        return changes or {"consumption": 0.1}

    @staticmethod
    def _extract_tariff_id(query: str) -> str:
        q = query.lower()
        if "economy 7" in q or "economy7" in q:
            return "economy7"
        if "agile" in q:
            return "agile_octopus"
        if "go" in q and "octopus" in q:
            return "octopus_go"
        return "economy7"

    # ------------------------------------------------------------------
    # Text-mode tool-call parser (Cerebras / DeepSeek compat)
    # ------------------------------------------------------------------

    _TOOL_RE = re.compile(
        r'\b(forecast_univariate|forecast_multivariate|explain_forecast|'
        r'detect_anomalies|explain_anomalies|counterfactual_fwd|counterfactual_inv|'
        r'get_cost_analysis|get_consumption_analysis|make_budget|'
        r'weather_impact_analysis|suggest_budget_correction|'
        r'evaluate_tariff_switch|generate_report'
        r')\s*\(([^)]*)\)'
    )
    _ARG_RE = re.compile(
        r"(\w+)\s*=\s*('(?:[^']*)'|\"(?:[^\"]*)\"|[^,)]+?)\s*(?:,|$)"
    )

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _parse_text_tool_call(self, text: str) -> Optional[dict]:
        text = self._strip_thinking(text)

        # ── Format 1: JSON {"type":"function","name":"...","parameters":{...}} ──
        try:
            start = text.index("{")
            depth, end = 0, start
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            obj = json.loads(text[start:end + 1])
            name = obj.get("name") or obj.get("function", {}).get("name", "")
            params = (obj.get("parameters") or obj.get("arguments")
                      or obj.get("function", {}).get("arguments") or {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {}
            if name and name in self.tool_map:
                return {"name": name, "args": params}
        except Exception:
            pass

        # ── Format 2: tool_name(arg=val, ...) ──
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

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

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
