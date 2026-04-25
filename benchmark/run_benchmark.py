"""EnergyX LLM Benchmark — three-phase evaluation harness.

Follows AgentBench (Liu et al., 2023) and ToolBench (Qin et al., 2023) conventions.

Phases
------
1. routing     — classify_query() accuracy across 75 labelled queries
2. tool_call   — first-tool-selection accuracy against 40 gold plans
3. synthesis   — RAG answer faithfulness across 25 context/query pairs

Token budget (per call, kept deliberately tight):
  routing   →   10 tokens  (single intent word)
  tool_call →  200 tokens  (tool name + JSON args, first call only)
  synthesis →  150 tokens  (2-sentence grounded answer)
  judge     →   20 tokens  (YES/NO + ≤15-word reason)

Prompt-level logging
--------------------
Every result record stores a `prompt_log` dict containing:
  - prompt_text       : exact string sent to the model (full, untruncated)
  - raw_response      : exact string returned by the model
  - prompt_chars      : character count of prompt_text
  - response_chars    : character count of raw_response
  - prompt_tokens_est : rough token estimate (chars / 4)
  - response_tokens_est
  - model_name        : model string passed to the API
  - temperature       : temperature used for this call
  - max_tokens        : token cap applied
  - timestamp_utc     : ISO-8601 call timestamp

This allows full reproducibility audit and cost estimation post-hoc.

Usage
-----
  python -m benchmark.run_benchmark --phases routing tool_call synthesis
  python -m benchmark.run_benchmark --phases routing --models cerebras/llama3.1-8b ollama/llama3.2:3b
  python -m benchmark.run_benchmark --stats-only   # re-run stats on existing results
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.models import ALL_MODELS, ModelConfig, get_model, get_llm_for_model, strip_thinking
from benchmark.metrics import (
    arg_accuracy, bonferroni_correction, cohens_kappa, first_tool_correct,
    hallucination_rate, latency_stats, macro_f1, mcnemars_test, rouge_l,
    routing_accuracy, step_efficiency, tool_f1, wilcoxon_signed_rank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR  = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Prompt-level logger
# ---------------------------------------------------------------------------

def _prompt_log(
    prompt_text: str,
    raw_response: Optional[str],
    model_cfg: ModelConfig,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build a fully reproducible audit record for a single LLM call.

    Stored verbatim in every per_item result so any call can be replayed
    or inspected without re-running the benchmark.
    """
    prompt_chars   = len(prompt_text)
    response_chars = len(raw_response) if raw_response else 0
    return {
        "prompt_text":         prompt_text,
        "raw_response":        raw_response if raw_response is not None else "",
        "prompt_chars":        prompt_chars,
        "response_chars":      response_chars,
        "prompt_tokens_est":   prompt_chars  // 4,
        "response_tokens_est": response_chars // 4,
        "model_id":            model_cfg.model_id,
        "model_name":          model_cfg.model_name,
        "provider":            model_cfg.provider,
        "temperature":         temperature,
        "max_tokens":          max_tokens,
        "timestamp_utc":       datetime.datetime.utcnow().isoformat() + "Z",
    }

# Valid tool names (mirrors tinyts/agent.py BASE_PROMPT)
VALID_TOOLS = [
    "train_forecast_model",
    "train_and_explain_forecast",
    "combine_forecasts",
    "detect_anomalies",
    "explain_anomalies",
    "generate_report",
    "counterfactual_forward",
    "counterfactual_inverse",
]

# ---------------------------------------------------------------------------
# Stub tools for tool-calling phase (no real computation — schema only)
# ---------------------------------------------------------------------------

def _build_stub_tools():
    """Execution-phase tools only — profile_dataset is intentionally excluded.

    In the real agent, profiling happens during plan() before execute() is called.
    Including it here caused small models to always call it first (literal reading of
    the BASE_PROMPT instruction), collapsing tool-calling accuracy to 0 for all cases.
    This is a real finding (logged in EXPT-001) but not what we are measuring here.
    The benchmark tests execution-phase tool selection: given a resolved plan, call
    the correct first execution tool.
    """
    from langchain_core.tools import tool

    @tool
    def train_forecast_model(model_name: str, horizon: int) -> str:
        """Train one forecast model.
        model_name: Naive | SeasonalNaive | ARIMA | ETS | N-BEATS | RandomForest | LightGBM
        horizon: number of steps to forecast ahead."""
        return "{}"

    @tool
    def train_and_explain_forecast(model_name: str, horizon: int) -> str:
        """Train a forecast model and compute SHAP feature importance.
        model_name: Naive | SeasonalNaive | ARIMA | ETS | N-BEATS | RandomForest | LightGBM
        horizon: number of steps to forecast ahead."""
        return "{}"

    @tool
    def combine_forecasts() -> str:
        """Combine all trained models into an inverse-SMAPE weighted ensemble. Call after all train calls."""
        return "{}"

    @tool
    def detect_anomalies() -> str:
        """Run 7-method anomaly detection ensemble (ZScore, MAD, IQR, Rolling, IsolationForest, STL, DBSCAN)."""
        return "{}"

    @tool
    def explain_anomalies() -> str:
        """Explain detected anomalies with context snapshots. Call after detect_anomalies."""
        return "{}"

    @tool
    def generate_report() -> str:
        """Generate final markdown analysis report. Call last."""
        return "{}"

    @tool
    def counterfactual_forward(changes_json: str, horizon: int) -> str:
        """Run a what-if scenario. changes_json: JSON dict of feature→delta. horizon: steps."""
        return "{}"

    @tool
    def counterfactual_inverse(target_value: float, constraints_json: str) -> str:
        """Find feature changes needed to reach target_value. constraints_json: JSON dict."""
        return "{}"

    return [
        train_forecast_model, train_and_explain_forecast,
        combine_forecasts, detect_anomalies, explain_anomalies, generate_report,
        counterfactual_forward, counterfactual_inverse,
    ]


# Benchmark-specific system prompt for the tool-calling phase.
# Differs from tinyts BASE_PROMPT in two ways:
#   1. profile_dataset is removed from tool list (already done in plan())
#   2. Opens with "Profiling complete" so there is no competing "call first" instruction
_TC_SYSTEM_PROMPT = """You are EnergyX, an expert time series analysis agent.
Dataset profiling is already complete. Do NOT call profile_dataset.

PROTOCOL:
- Call ONE tool per response. No text, just the tool call.
- Wait for the result, then decide the next tool.
- After ALL tools are done, provide your analysis.

AVAILABLE TOOLS:
- train_forecast_model(model_name, horizon) — Train one forecast model
- train_and_explain_forecast(model_name, horizon) — Train + SHAP feature importance
- combine_forecasts() — Combine trained models into ensemble (call after all train calls)
- detect_anomalies() — Run 7-method anomaly detection ensemble
- explain_anomalies() — Explain anomalies (call after detect_anomalies)
- generate_report() — Generate final report (call last if required)
- counterfactual_forward(changes_json, horizon) — What-if scenario
- counterfactual_inverse(target_value, constraints_json) — Reach target value

MODELS: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS (univariate) | RandomForest, LightGBM (multivariate)

RESILIENCE:
- If a tool returns an error, retry once. If it fails again, skip and continue.
- Report any skipped tools in your final output.
"""

# Text-based tool-calling fallback for models that don't support native bind_tools.
# Used when Ollama returns HTTP 400 "does not support tools".
_TC_TEXT_FALLBACK_PROMPT = """You are EnergyX, an expert time series analysis agent.
Dataset profiling is already complete.

AVAILABLE TOOLS (choose exactly one):
- train_forecast_model(model_name: str, horizon: int)
- train_and_explain_forecast(model_name: str, horizon: int)
- combine_forecasts()
- detect_anomalies()
- explain_anomalies()
- generate_report()
- counterfactual_forward(changes_json: str, horizon: int)
- counterfactual_inverse(target_value: float, constraints_json: str)

MODELS: Naive, SeasonalNaive, ARIMA, ETS, N-BEATS (univariate) | RandomForest, LightGBM (multivariate)

Given the plan below, respond with ONLY a JSON object for the FIRST tool to call.
Format: {{"tool": "<tool_name>", "args": {{<arg_key>: <value>, ...}}}}
No explanation, no text — just the JSON object.

{plan_text}"""

_TC_TEXT_TOOL_RE = __import__("re").compile(
    r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}', __import__("re").DOTALL
)


def _parse_text_tool_call(text: str) -> tuple:
    """Extract (tool_name, args_dict) from a text JSON tool-call response.

    Strategy:
      1. Try json.loads on the whole cleaned text (model output is often exactly the JSON)
      2. Find the outermost {...} via balanced-brace scan and parse that
      3. Fall back to regex for the tool name only
    """
    import json as _json, re

    text = text.strip()

    # 1. Direct parse — handles the common case cleanly
    try:
        obj = _json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj["tool"], obj.get("args", {})
    except Exception:
        pass

    # 2. Balanced-brace extraction (handles nested args dicts)
    start = text.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                obj = _json.loads(text[start:end+1])
                if isinstance(obj, dict) and "tool" in obj:
                    return obj["tool"], obj.get("args", {})
            except Exception:
                pass

    # 3. Regex fallback — at least recover the tool name
    m = re.search(r'"tool"\s*:\s*"([^"]+)"', text)
    if m:
        args_m = re.search(r'"args"\s*:\s*(\{[^}]*\})', text)
        args = {}
        if args_m:
            try:
                args = _json.loads(args_m.group(1))
            except Exception:
                pass
        return m.group(1).rstrip("()"), args

    return None, {}


def _normalise_tool_name(name: Optional[str]) -> Optional[str]:
    """Strip trailing () that some models append to tool names in text output."""
    return name.rstrip("()").strip() if name else None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_results(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved → {path}")


def _call_with_retry(
    llm,
    prompt: str,
    model_cfg: ModelConfig,
    temperature: float,
    max_tokens: int,
    retries: int = 2,
) -> Tuple[Optional[str], float, Dict]:
    """Invoke LLM; return (content, latency_ms, prompt_log).

    prompt_log is populated even on failure (raw_response will be empty).
    """
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            resp = llm.invoke(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = (resp.content if hasattr(resp, "content") else str(resp)).strip()
            # Strip <think>...</think> for reasoning models; evaluation uses cleaned content
            content = strip_thinking(raw) if model_cfg.uses_thinking else raw
            log = _prompt_log(prompt, raw, model_cfg, temperature, max_tokens)
            log["response_cleaned"] = content
            log["had_thinking_block"] = raw != content
            think_chars = len(raw) - len(content)
            log["thinking_chars"] = think_chars if raw != content else 0
            return content, round(latency_ms, 1), log
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt+1}/{retries+1} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)

    log = _prompt_log(prompt, None, model_cfg, temperature, max_tokens)
    return None, -1.0, log


# ---------------------------------------------------------------------------
# Phase 1: Routing
# ---------------------------------------------------------------------------

_ROUTING_PROMPT = """Classify the following user query into exactly one of these intents:

- knowledge   : Questions about UK energy regulations, schemes (ECO4, BUS, SEG), tariffs, EPC, suppliers.
- analysis    : Forecasting, anomaly detection, what-if scenarios, bill prediction, reports, budget tracking.
- control     : Direct device commands — turn on/off, set temperature, schedule appliance, defer load.
- status      : Current/live readings, real-time cost, recent events, monitoring status.

Query: {query}

Respond with ONLY the intent word (knowledge, analysis, control, or status)."""


def run_routing_phase(
    models: List[ModelConfig],
    dataset: List[Dict],
    seeds: int = 3,
    output_dir: Path = RESULTS_DIR,
) -> Dict:
    """Evaluate routing classification.

    For each model × seed, classify all queries and compute accuracy + macro F1.
    Token cap: 10 per call.
    """
    logger.info(f"=== ROUTING PHASE | {len(dataset)} queries | {len(models)} models | {seeds} seeds ===")
    phase_results = {}

    _ROUTING_TEMP = {0: 0.0, 1: 0.1, 2: 0.1}  # seed → temperature
    _ROUTING_MAX_TOKENS = 10

    for model_cfg in models:
        logger.info(f"  Model: {model_cfg.model_id}")
        model_runs = []

        for seed in range(seeds):
            temperature = _ROUTING_TEMP.get(seed, 0.1)
            logger.info(f"    Seed {seed+1}/{seeds} (temp={temperature})")
            llm = get_llm_for_model(model_cfg, temperature=temperature, max_tokens=_ROUTING_MAX_TOKENS)
            preds, labels, latencies, per_item = [], [], [], []

            for item in dataset:
                prompt = _ROUTING_PROMPT.format(query=item["query"])
                content, lat, plog = _call_with_retry(
                    llm, prompt, model_cfg, temperature, _ROUTING_MAX_TOKENS
                )
                if content is None:
                    pred = "error"
                else:
                    pred = "unknown"
                    for intent in ("knowledge", "analysis", "control", "status"):
                        if intent in content.lower():
                            pred = intent
                            break

                correct = pred == item["label"]
                preds.append(pred)
                labels.append(item["label"])
                latencies.append(lat)
                per_item.append({
                    "id": item["id"],
                    "query": item["query"],
                    "label": item["label"],
                    "pred": pred,
                    "correct": correct,
                    "latency_ms": lat,
                    "difficulty": item.get("difficulty"),
                    "prompt_log": plog,
                })

            valid_preds = [p for p in preds if p != "error"]
            valid_labels = [l for p, l in zip(preds, labels) if p != "error"]

            run_metrics = {
                "seed": seed,
                "accuracy": routing_accuracy(valid_preds, valid_labels) if valid_preds else 0.0,
                "per_class": macro_f1(valid_preds, valid_labels) if valid_preds else {},
                "latency": latency_stats([l for l in latencies if l >= 0]),
                "error_rate": round(sum(p == "error" for p in preds) / len(preds), 4),
                "per_item": per_item,
            }
            model_runs.append(run_metrics)

        # Aggregate across seeds
        accs = [r["accuracy"] for r in model_runs]
        phase_results[model_cfg.model_id] = {
            "model": model_cfg.display_name,
            "tier": model_cfg.tier,
            "provider": model_cfg.provider,
            "accuracy_mean": round(sum(accs) / len(accs), 4),
            "accuracy_std":  round(_std(accs), 4),
            "macro_f1": model_runs[0]["per_class"].get("macro_f1", 0.0),  # seed-0 detail
            "per_class_seed0": model_runs[0]["per_class"],
            "latency_seed0": model_runs[0]["latency"],
            "runs": model_runs,
        }

    _save_results(phase_results, output_dir / "routing_results.json")
    _print_routing_summary(phase_results)
    return phase_results


def _print_routing_summary(results: Dict) -> None:
    print("\n--- ROUTING SUMMARY ---")
    print(f"{'Model':<40} {'Acc (mean±std)':<18} {'Macro F1':<10} {'p50 ms'}")
    print("-" * 82)
    for mid, r in sorted(results.items(), key=lambda x: -x[1]["accuracy_mean"]):
        lat = r["latency_seed0"].get("p50_ms", "-")
        print(
            f"{r['model']:<40} "
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}   "
            f"{r['macro_f1']:<10.4f} "
            f"{lat}"
        )


# ---------------------------------------------------------------------------
# Phase 2: Tool-calling (first-call accuracy)
# ---------------------------------------------------------------------------

def _build_tool_call_prompt(plan: Dict) -> str:
    """Build system+human prompt for the tool-calling benchmark.

    Uses _TC_SYSTEM_PROMPT (no profile_dataset, opens with profiling-complete).
    Human turn is explicit about what to do next so there is no competing instruction.
    """
    task = plan.get("task_type", "forecast")
    horizon = plan.get("horizon", 24)
    models = plan.get("models_included", ["Naive", "ARIMA", "ETS"])
    models_str = ", ".join(models)
    needs_expl = plan.get("needs_explanation", False)
    needs_report = plan.get("needs_report", False)
    is_mv = plan.get("is_multivariate", False)
    features = plan.get("feature_columns", [])
    cf_type = plan.get("counterfactual_type", None)
    cf_changes = plan.get("counterfactual_changes", {})
    cf_target = plan.get("counterfactual_target_value", None)
    cf_constraints = plan.get("counterfactual_constraints", {})

    human_lines = [
        "Profiling is complete. Execute this approved analysis plan by calling the first tool now.",
        "",
        f"Task:             {task}",
        f"Horizon:          {horizon} steps",
        f"Multivariate:     {is_mv}",
    ]
    if features:
        human_lines.append(f"Feature columns:  {', '.join(features)}")
    if models:
        human_lines.append(f"Models to train:  {models_str}")
    human_lines.append(f"Needs explanation:{needs_expl}")
    human_lines.append(f"Needs report:     {needs_report}")

    if cf_type == "forward":
        human_lines.append("")
        human_lines.append(f"Counterfactual forward — feature changes: {json.dumps(cf_changes)}")
        human_lines.append("Step 1: train the multivariate model(s). Step 2: call counterfactual_forward().")
    elif cf_type == "inverse":
        human_lines.append("")
        human_lines.append(f"Counterfactual inverse — target value: {cf_target}")
        if cf_constraints:
            human_lines.append(f"Constraints: {json.dumps(cf_constraints)}")
        human_lines.append("Step 1: train multivariate model(s) with explanation. Step 2: call counterfactual_inverse().")

    human_lines.append("")
    human_lines.append("Call the first tool now.")

    return _TC_SYSTEM_PROMPT + "\n\nHuman: " + "\n".join(human_lines)


def run_tool_calling_phase(
    models: List[ModelConfig],
    dataset: List[Dict],
    output_dir: Path = RESULTS_DIR,
) -> Dict:
    """Evaluate first-tool-call accuracy against gold plans.

    Token cap: 200 per call (enough for one tool call + args JSON).
    No repeated seeds — tool calls are largely deterministic at temp=0.
    """
    logger.info(f"=== TOOL-CALLING PHASE | {len(dataset)} cases | {len(models)} models ===")
    stub_tools = _build_stub_tools()
    phase_results = {}

    _TC_TEMP = 0.0
    _TC_MAX_TOKENS = 200

    for model_cfg in models:
        logger.info(f"  Model: {model_cfg.model_id}")
        llm = get_llm_for_model(model_cfg, temperature=_TC_TEMP, max_tokens=_TC_MAX_TOKENS)

        # Probe whether this model supports native tool calling
        use_text_fallback = False
        try:
            llm_with_tools = llm.bind_tools(stub_tools)
            # Do a cheap probe call to confirm tools work at runtime
            probe_prompt = _build_tool_call_prompt(
                {"task_type":"forecast","horizon":24,"models_included":["Naive"],
                 "needs_explanation":False,"needs_report":False,"is_multivariate":False,"feature_columns":[]}
            )
            probe_resp = llm_with_tools.invoke(probe_prompt)
            # If we get a 400-style error wrapped in content, treat as unsupported
            probe_content = getattr(probe_resp, "content", "") or ""
            if "does not support tools" in str(probe_content).lower():
                raise ValueError("tools not supported (detected in response)")
        except Exception as e:
            if "does not support tools" in str(e).lower() or "400" in str(e):
                logger.warning(f"    {model_cfg.model_id} does not support native tools — using text-based fallback")
                use_text_fallback = True
            else:
                logger.warning(f"    bind_tools/probe failed for {model_cfg.model_id}: {e}. Skipping.")
                phase_results[model_cfg.model_id] = {"error": str(e), "model": model_cfg.display_name}
                continue

        if use_text_fallback:
            # Plain LLM — no bind_tools; tool selection via JSON in text.
            # Extra tokens needed: thinking budget + longer JSON output for complex plans.
            llm_tc = get_llm_for_model(model_cfg, temperature=_TC_TEMP, max_tokens=350)
        else:
            llm_tc = llm_with_tools

        per_item = []
        first_tool_hits = []
        arg_scores = []
        all_latencies = []

        for item in dataset:
            if use_text_fallback:
                plan_text = _build_tool_call_prompt(item["plan"]).split("Human: ", 1)[-1]
                prompt = _TC_TEXT_FALLBACK_PROMPT.format(plan_text=plan_text)
            else:
                prompt = _build_tool_call_prompt(item["plan"])

            t0 = time.perf_counter()
            try:
                response = llm_tc.invoke(prompt)
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            except Exception as e:
                logger.warning(f"    {item['id']} failed: {e}")
                plog = _prompt_log(prompt, None, model_cfg, _TC_TEMP, _TC_MAX_TOKENS)
                per_item.append({"id": item["id"], "error": str(e), "prompt_log": plog})
                continue

            # Extract tool call — native or text-based
            if use_text_fallback:
                raw = (response.content if hasattr(response, "content") else str(response)).strip()
                cleaned = strip_thinking(raw) if model_cfg.uses_thinking else raw
                pred_tool_raw, pred_args = _parse_text_tool_call(cleaned)
                pred_tool = _normalise_tool_name(pred_tool_raw)
                raw_resp_str = raw
                tool_calls = [{"name": pred_tool, "args": pred_args}] if pred_tool else []
            else:
                tool_calls = getattr(response, "tool_calls", []) or []
                pred_tool = tool_calls[0]["name"] if tool_calls else None
                pred_args  = tool_calls[0].get("args", {}) if tool_calls else {}
                raw_resp_str = (
                    json.dumps({"tool": pred_tool, "args": pred_args})
                    if pred_tool else
                    (response.content if hasattr(response, "content") else str(response))
                )

            plog = _prompt_log(prompt, raw_resp_str, model_cfg, _TC_TEMP, _TC_MAX_TOKENS)
            plog["tool_call_method"] = "text_fallback" if use_text_fallback else "native"
            if model_cfg.uses_thinking:
                plog["had_thinking_block"] = raw_resp_str != (strip_thinking(raw_resp_str) if use_text_fallback else raw_resp_str)
                plog["thinking_chars"] = len(raw_resp_str) - len(strip_thinking(raw_resp_str))

            gold_tool = item["expected_first_tool"]
            gold_args = item.get("expected_first_args", {})

            ftc = first_tool_correct(pred_tool, gold_tool)
            aa  = arg_accuracy(pred_args, gold_args) if gold_args else None
            hr  = hallucination_rate([pred_tool] if pred_tool else [], VALID_TOOLS)

            first_tool_hits.append(ftc)
            if aa is not None:
                arg_scores.append(aa)
            all_latencies.append(latency_ms)

            per_item.append({
                "id": item["id"],
                "description": item.get("description", ""),
                "workflow_type": item.get("workflow_type"),
                "difficulty": item.get("difficulty"),
                "gold_tool": gold_tool,
                "pred_tool": pred_tool,
                "pred_args": pred_args,
                "gold_args": gold_args,
                "first_tool_correct": ftc,
                "arg_accuracy": aa,
                "hallucination_rate": hr,
                "latency_ms": latency_ms,
                "n_tool_calls_returned": len(tool_calls),
                "prompt_log": plog,
            })

        # Difficulty breakdown
        easy   = [x for x in per_item if x.get("difficulty") == "easy"   and "error" not in x]
        medium = [x for x in per_item if x.get("difficulty") == "medium" and "error" not in x]
        hard   = [x for x in per_item if x.get("difficulty") == "hard"   and "error" not in x]

        phase_results[model_cfg.model_id] = {
            "model": model_cfg.display_name,
            "tier": model_cfg.tier,
            "provider": model_cfg.provider,
            "first_tool_accuracy": round(sum(first_tool_hits) / len(first_tool_hits), 4) if first_tool_hits else 0.0,
            "arg_accuracy_mean": round(sum(arg_scores) / len(arg_scores), 4) if arg_scores else None,
            "by_difficulty": {
                "easy":   round(sum(x["first_tool_correct"] for x in easy)   / len(easy),   4) if easy   else None,
                "medium": round(sum(x["first_tool_correct"] for x in medium) / len(medium), 4) if medium else None,
                "hard":   round(sum(x["first_tool_correct"] for x in hard)   / len(hard),   4) if hard   else None,
            },
            "latency": latency_stats(all_latencies) if all_latencies else {},
            "n_items": len(per_item),
            "per_item": per_item,
        }

    _save_results(phase_results, output_dir / "tool_calling_results.json")
    _print_tool_calling_summary(phase_results)
    return phase_results


def _print_tool_calling_summary(results: Dict) -> None:
    print("\n--- TOOL-CALLING SUMMARY ---")
    print(f"{'Model':<40} {'FTC Acc':<10} {'Arg Acc':<10} {'Easy':<8} {'Med':<8} {'Hard':<8}")
    print("-" * 84)
    for mid, r in sorted(results.items(), key=lambda x: -(x[1].get("first_tool_accuracy") or 0)):
        if "error" in r:
            print(f"{r.get('model', mid):<40} ERROR: {r['error']}")
            continue
        bd = r.get("by_difficulty", {})
        print(
            f"{r['model']:<40} "
            f"{r['first_tool_accuracy']:<10.4f} "
            f"{str(r['arg_accuracy_mean']):<10} "
            f"{str(bd.get('easy', '-')):<8} "
            f"{str(bd.get('medium', '-')):<8} "
            f"{str(bd.get('hard', '-')):<8}"
        )


# ---------------------------------------------------------------------------
# Phase 3: Synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
Answer the question below using ONLY the information in the provided context. Be concise — answer in 2 sentences maximum.

Context: {context}

Question: {query}

Answer:"""


def run_synthesis_phase(
    models: List[ModelConfig],
    dataset: List[Dict],
    seeds: int = 3,
    output_dir: Path = RESULTS_DIR,
    run_judge: bool = True,
) -> Dict:
    """Evaluate RAG synthesis quality.

    Metrics: ROUGE-L vs reference, faithfulness (judge YES/NO), completeness (judge YES/NO).
    Token cap: 150 per generation call, 20 per judge call.
    """
    logger.info(f"=== SYNTHESIS PHASE | {len(dataset)} items | {len(models)} models | {seeds} seeds ===")

    judge_llm = None
    if run_judge:
        try:
            from benchmark.judge import faithfulness_score, completeness_score
            judge_llm = True  # signal that judge is available; lazy-init inside judge.py
        except Exception as e:
            logger.warning(f"Judge unavailable: {e}. Skipping faithfulness scoring.")
            judge_llm = None

    phase_results = {}

    for model_cfg in models:
        logger.info(f"  Model: {model_cfg.model_id}")
        model_runs = []

        _SYN_TEMP_MAP = {0: 0.0, 1: 0.3, 2: 0.3}
        _SYN_MAX_TOKENS = 150

        for seed in range(seeds):
            temperature = _SYN_TEMP_MAP.get(seed, 0.3)
            logger.info(f"    Seed {seed+1}/{seeds} (temp={temperature})")
            llm = get_llm_for_model(model_cfg, temperature=temperature, max_tokens=_SYN_MAX_TOKENS)
            per_item = []

            for item in dataset:
                prompt = _SYNTHESIS_PROMPT.format(context=item["context"], query=item["query"])
                response, latency_ms, plog = _call_with_retry(
                    llm, prompt, model_cfg, temperature, _SYN_MAX_TOKENS
                )
                if response is None:
                    plog["error"] = "llm_failed"
                    per_item.append({"id": item["id"], "error": "llm_failed", "prompt_log": plog})
                    continue

                rl = rouge_l(response, item["reference_answer"])
                faith = {"score": -1, "raw": "skipped"}
                comp  = {"score": -1, "raw": "skipped"}

                if judge_llm:
                    from benchmark.judge import faithfulness_score, completeness_score
                    faith = faithfulness_score(response, item["context"], item["query"])
                    comp  = completeness_score(response, item["query"])

                per_item.append({
                    "id": item["id"],
                    "domain": item.get("domain"),
                    "difficulty": item.get("difficulty"),
                    "query": item["query"],
                    "response": response,
                    "reference": item["reference_answer"],
                    "rouge_l": rl,
                    "faithfulness": faith["score"],
                    "faithfulness_raw": faith.get("raw", ""),
                    "completeness": comp["score"],
                    "completeness_raw": comp.get("raw", ""),
                    "latency_ms": latency_ms,
                    "prompt_log": plog,
                })

            valid = [x for x in per_item if "error" not in x]
            rouge_scores = [x["rouge_l"] for x in valid]
            faith_scores = [x["faithfulness"] for x in valid if x["faithfulness"] >= 0]
            comp_scores  = [x["completeness"] for x in valid if x["completeness"] >= 0]

            model_runs.append({
                "seed": seed,
                "rouge_l_mean": round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else 0.0,
                "faithfulness_rate": round(sum(faith_scores) / len(faith_scores), 4) if faith_scores else None,
                "completeness_rate": round(sum(comp_scores) / len(comp_scores), 4) if comp_scores else None,
                "per_item": per_item,
            })

        rouge_means = [r["rouge_l_mean"] for r in model_runs]
        phase_results[model_cfg.model_id] = {
            "model": model_cfg.display_name,
            "tier": model_cfg.tier,
            "provider": model_cfg.provider,
            "rouge_l_mean": round(sum(rouge_means) / len(rouge_means), 4),
            "rouge_l_std":  round(_std(rouge_means), 4),
            "faithfulness_rate_seed0": model_runs[0]["faithfulness_rate"],
            "completeness_rate_seed0": model_runs[0]["completeness_rate"],
            "runs": model_runs,
        }

    _save_results(phase_results, output_dir / "synthesis_results.json")
    _print_synthesis_summary(phase_results)
    return phase_results


def _print_synthesis_summary(results: Dict) -> None:
    print("\n--- SYNTHESIS SUMMARY ---")
    print(f"{'Model':<40} {'ROUGE-L (±std)':<20} {'Faithful':<12} {'Complete'}")
    print("-" * 80)
    for mid, r in sorted(results.items(), key=lambda x: -(x[1].get("rouge_l_mean") or 0)):
        print(
            f"{r['model']:<40} "
            f"{r['rouge_l_mean']:.4f} ± {r['rouge_l_std']:.4f}    "
            f"{str(r['faithfulness_rate_seed0']):<12} "
            f"{r['completeness_rate_seed0']}"
        )


# ---------------------------------------------------------------------------
# Statistical analysis (run across all saved results)
# ---------------------------------------------------------------------------

def run_statistical_analysis(output_dir: Path = RESULTS_DIR) -> None:
    """Load saved phase results, run McNemar + Wilcoxon, apply Bonferroni."""
    print("\n=== STATISTICAL ANALYSIS ===")

    for phase_file, metric_key, is_binary in [
        ("routing_results.json",     "accuracy_mean",          False),
        ("tool_calling_results.json","first_tool_accuracy",    False),
        ("synthesis_results.json",   "rouge_l_mean",           False),
    ]:
        path = output_dir / phase_file
        if not path.exists():
            logger.info(f"Skipping {phase_file} (not found)")
            continue

        with open(path) as f:
            results = json.load(f)

        model_ids = [mid for mid, r in results.items() if "error" not in r]
        if len(model_ids) < 2:
            continue

        print(f"\n--- {phase_file} ---")

        # Pairwise Wilcoxon on per-item continuous scores
        pairs = [(a, b) for i, a in enumerate(model_ids) for b in model_ids[i+1:]]
        p_values = []

        for id_a, id_b in pairs:
            ra = results[id_a]
            rb = results[id_b]
            # Extract per-item scores from seed-0
            scores_a = _extract_item_scores(ra, phase_file)
            scores_b = _extract_item_scores(rb, phase_file)
            if not scores_a or not scores_b or len(scores_a) != len(scores_b):
                continue

            stat = wilcoxon_signed_rank(scores_a, scores_b)
            p_values.append(stat["p_value"])
            print(f"  {id_a} vs {id_b}: W={stat['W']}, p={stat['p_value']}")

        if p_values and isinstance(p_values[0], float):
            corrected = bonferroni_correction(p_values)
            print(f"  Bonferroni-corrected p-values: {corrected}")

    print("\nDone. Interpret p < 0.05 (after Bonferroni) as significant.")


def _extract_item_scores(result: Dict, phase_file: str) -> List[float]:
    if "routing" in phase_file:
        runs = result.get("runs", [])
        if not runs:
            return []
        seed0 = runs[0]
        return [float(x["correct"]) for x in seed0.get("per_item", []) if "correct" in x]
    elif "tool_calling" in phase_file:
        return [float(x["first_tool_correct"]) for x in result.get("per_item", []) if "first_tool_correct" in x]
    elif "synthesis" in phase_file:
        runs = result.get("runs", [])
        if not runs:
            return []
        return [x["rouge_l"] for x in runs[0].get("per_item", []) if "rouge_l" in x]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EnergyX LLM Benchmark — AgentBench/ToolBench-style",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phases", nargs="+",
        choices=["routing", "tool_call", "synthesis", "stats"],
        default=["routing", "tool_call", "synthesis"],
        help="Phases to run (default: all three evaluation phases)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=(
            "Model IDs to include (default: all). Examples:\n"
            "  cerebras/llama3.1-8b\n"
            "  ollama/mistral:7b\n"
            "Run without --models to use all registered models."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds for routing + synthesis phases (default: 3)",
    )
    parser.add_argument(
        "--output-dir", default=str(RESULTS_DIR),
        help="Directory to write results JSON files (default: benchmark/results/)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-judge calls in synthesis phase (saves tokens)",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only run statistical analysis on existing results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stats_only:
        run_statistical_analysis(output_dir)
        return

    # Resolve model list
    if args.models:
        selected = []
        for mid in args.models:
            m = get_model(mid)
            if m is None:
                logger.warning(f"Unknown model ID '{mid}' — skipping. Available: {[m.model_id for m in ALL_MODELS]}")
            else:
                selected.append(m)
    else:
        selected = ALL_MODELS

    if not selected:
        logger.error("No valid models selected. Exiting.")
        sys.exit(1)

    logger.info(f"Running benchmark: phases={args.phases}, models={[m.model_id for m in selected]}, seeds={args.seeds}")

    if "routing" in args.phases:
        dataset = _load_jsonl(DATASETS_DIR / "routing_test.jsonl")
        run_routing_phase(selected, dataset, seeds=args.seeds, output_dir=output_dir)

    if "tool_call" in args.phases:
        dataset = _load_jsonl(DATASETS_DIR / "tool_calling_test.jsonl")
        run_tool_calling_phase(selected, dataset, output_dir=output_dir)

    if "synthesis" in args.phases:
        dataset = _load_jsonl(DATASETS_DIR / "synthesis_test.jsonl")
        run_synthesis_phase(
            selected, dataset,
            seeds=args.seeds,
            output_dir=output_dir,
            run_judge=not args.no_judge,
        )

    if "stats" in args.phases or len(args.phases) > 1:
        run_statistical_analysis(output_dir)


if __name__ == "__main__":
    main()
