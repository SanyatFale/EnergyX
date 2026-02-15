"""TinyTS Query Success Rate Benchmark.

Two-phase design:
  1. --collect : Call LLM via agent.plan() and dump raw plans to JSON.
                 Agent is created ONCE per dataset (single CSV load + profile).
  2. --evaluate: Pure offline comparison of collected plans vs ground truth.
                 Zero ML imports, zero compute — just dict matching.

Usage:
    python benchmark.py --collect                           # collect all 50
    python benchmark.py --collect --category "Filtered Query"
    python benchmark.py --collect --max-queries 10
    python benchmark.py --collect --parallel 3 --api-keys "k1,k2,k3"

    python benchmark.py --evaluate                          # evaluate collected
    python benchmark.py --evaluate --input my_results.json

    python benchmark.py --dry-run                           # print query set
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ground truth: query definitions with expected plan fields + tool order
# ---------------------------------------------------------------------------
#
# Horizon values are EXACT (steps_per_day * days).
#   ETTh1:            frequency=H -> 24 steps/day
#   building_energy:   frequency=H -> 24 steps/day
#
# Tool sequences: profile_dataset is OPTIONAL, not listed.
# Wildcards: "tool_name*" = one or more calls.

QUERY_SET: Dict[str, List[Dict[str, Any]]] = {
    # ------------------------------------------------------------------
    # 1. Simple Univariate Forecast
    # ------------------------------------------------------------------
    "Univariate Forecast": [
        {
            "query": "Forecast OT for the next 7 days",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": False,
                "models_subset": ["Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS"],
                "horizon": 168,
                "needs_explanation": False,
                "counterfactual_type": None,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Predict the next 24 hours of OT",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": False,
                "horizon": 24,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "What will the target look like over the next 3 days?",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": False,
                "horizon": 72,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Give me a 48-step forecast of OT",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": False,
                "horizon": 48,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Forecast the next week of data",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": False,
                "horizon": 168,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
    ],

    # ------------------------------------------------------------------
    # 2. Multivariate Forecast
    # ------------------------------------------------------------------
    "Multivariate Forecast": [
        {
            "query": "Forecast meter reading using air temperature and wind speed as features for 3 days",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": True,
                "models_subset": ["RandomForest", "LightGBM"],
                "horizon": 72,
                "feature_columns_include": ["air_temperature", "wind_speed"],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Predict OT using HUFL, HULL, MUFL as covariates for 24 steps",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": True,
                "models_subset": ["RandomForest", "LightGBM"],
                "horizon": 24,
                "feature_columns_include": ["HUFL", "HULL", "MUFL"],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Use all weather variables to forecast meter reading for 7 days",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": True,
                "models_subset": ["RandomForest", "LightGBM"],
                "horizon": 168,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Multivariate forecast of energy usage with temperature and humidity for 48 hours",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": True,
                "horizon": 48,
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Predict next 5 days of meter reading considering dew temperature and cloud coverage",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast",
                "is_multivariate": True,
                "horizon": 120,
                "feature_columns_include": ["dew_temperature", "cloud_coverage"],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
    ],

    # ------------------------------------------------------------------
    # 3. Anomaly Detection
    # ------------------------------------------------------------------
    "Anomaly Detection": [
        {
            "query": "Detect anomalies in the dataset",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "anomaly",
                "needs_explanation": False,
                "counterfactual_type": None,
            },
            "expect_tools": ["detect_anomalies"],
        },
        {
            "query": "Find outliers in the meter reading data",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "anomaly"},
            "expect_tools": ["detect_anomalies"],
        },
        {
            "query": "Are there any anomalous patterns in OT?",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly"},
            "expect_tools": ["detect_anomalies"],
        },
        {
            "query": "Check for unusual spikes in energy consumption",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "anomaly"},
            "expect_tools": ["detect_anomalies"],
        },
        {
            "query": "Run anomaly detection on this time series",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly"},
            "expect_tools": ["detect_anomalies"],
        },
    ],

    # ------------------------------------------------------------------
    # 4. Anomaly + Explanation
    # ------------------------------------------------------------------
    "Anomaly Explained": [
        {
            "query": "Detect and explain anomalies in meter reading",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
        {
            "query": "Find outliers and tell me why they happened",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
        {
            "query": "Detect anomalies in OT and explain their probable causes",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
        {
            "query": "Are there anomalies? If so, explain what caused them",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
        {
            "query": "Run anomaly detection with full explanation",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
    ],

    # ------------------------------------------------------------------
    # 5. Forecast + Explanation
    # ------------------------------------------------------------------
    "Forecast Explained": [
        {
            "query": "Forecast OT for 7 days and explain what drives the predictions",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "forecast", "needs_explanation": True, "horizon": 168},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts"],
        },
        {
            "query": "Predict meter reading for 3 days with SHAP explanations",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "forecast", "needs_explanation": True, "horizon": 72},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts"],
        },
        {
            "query": "Give me a 24-step forecast with feature importance analysis",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "forecast", "needs_explanation": True, "horizon": 24},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts"],
        },
        {
            "query": "Forecast the next week and explain why each model performs differently",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "forecast", "needs_explanation": True, "horizon": 168},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts"],
        },
        {
            "query": "Predict next 48 hours and tell me which features matter most",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "forecast", "needs_explanation": True, "horizon": 48},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts"],
        },
    ],

    # ------------------------------------------------------------------
    # 6. Counterfactual Forward
    # ------------------------------------------------------------------
    "Counterfactual Forward": [
        {
            "query": "What if air temperature drops by 5 degrees, how does meter reading change over 3 days?",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast", "is_multivariate": True,
                "counterfactual_type": "forward",
                "models_subset": ["RandomForest", "LightGBM"], "horizon": 72,
            },
            "expect_tools": ["train_forecast_model*", "counterfactual_forward"],
        },
        {
            "query": "What happens to OT if HUFL increases by 2 for the next 24 hours?",
            "dataset": "ETTh1",
            "expect_plan": {
                "task_type": "forecast", "is_multivariate": True,
                "counterfactual_type": "forward", "horizon": 24,
            },
            "expect_tools": ["train_forecast_model*", "counterfactual_forward"],
        },
        {
            "query": "Simulate meter reading if wind speed doubles for 7 days",
            "dataset": "building_energy",
            "expect_plan": {"is_multivariate": True, "counterfactual_type": "forward", "horizon": 168},
            "expect_tools": ["train_forecast_model*", "counterfactual_forward"],
        },
        {
            "query": "What if cloud coverage drops to 0, forecast energy for 48 hours",
            "dataset": "building_energy",
            "expect_plan": {"is_multivariate": True, "counterfactual_type": "forward", "horizon": 48},
            "expect_tools": ["train_forecast_model*", "counterfactual_forward"],
        },
        {
            "query": "How would meter reading change if dew temperature increases by 3 over the next day?",
            "dataset": "building_energy",
            "expect_plan": {"is_multivariate": True, "counterfactual_type": "forward", "horizon": 24},
            "expect_tools": ["train_forecast_model*", "counterfactual_forward"],
        },
    ],

    # ------------------------------------------------------------------
    # 7. Counterfactual Inverse
    # ------------------------------------------------------------------
    "Counterfactual Inverse": [
        {
            "query": "I want meter reading to reach 500, what features need to change?",
            "dataset": "building_energy",
            "expect_plan": {
                "is_multivariate": True, "counterfactual_type": "inverse",
                "counterfactual_target_value": 500.0,
            },
            "expect_tools": ["train_and_explain_forecast*", "counterfactual_inverse"],
        },
        {
            "query": "How can I reduce OT to 20 over the next 24 hours?",
            "dataset": "ETTh1",
            "expect_plan": {
                "is_multivariate": True, "counterfactual_type": "inverse",
                "counterfactual_target_value": 20.0,
            },
            "expect_tools": ["train_and_explain_forecast*", "counterfactual_inverse"],
        },
        {
            "query": "What changes are needed to bring energy consumption down to 100?",
            "dataset": "building_energy",
            "expect_plan": {
                "is_multivariate": True, "counterfactual_type": "inverse",
                "counterfactual_target_value": 100.0,
            },
            "expect_tools": ["train_and_explain_forecast*", "counterfactual_inverse"],
        },
        {
            "query": "I want the target to go to 300, what should I adjust?",
            "dataset": "building_energy",
            "expect_plan": {
                "is_multivariate": True, "counterfactual_type": "inverse",
                "counterfactual_target_value": 300.0,
            },
            "expect_tools": ["train_and_explain_forecast*", "counterfactual_inverse"],
        },
        {
            "query": "Reduce meter reading to 50, temperature can't go below 10",
            "dataset": "building_energy",
            "expect_plan": {
                "is_multivariate": True, "counterfactual_type": "inverse",
                "counterfactual_target_value": 50.0,
            },
            "expect_tools": ["train_and_explain_forecast*", "counterfactual_inverse"],
        },
    ],

    # ------------------------------------------------------------------
    # 8. Filtered Query
    # ------------------------------------------------------------------
    "Filtered Query": [
        {
            "query": "Forecast electricity meter reading for 7 days",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast", "horizon": 168,
                "data_filters_include": [{"column": "meter_type", "value": "electricity"}],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Detect anomalies in chilledwater consumption",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "anomaly",
                "data_filters_include": [{"column": "meter_type", "value": "chilledwater"}],
            },
            "expect_tools": ["detect_anomalies"],
        },
        {
            "query": "Predict steam usage for the next 3 days",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast", "horizon": 72,
                "data_filters_include": [{"column": "meter_type", "value": "steam"}],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "What will electricity consumption look like next week?",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "forecast", "horizon": 168,
                "data_filters_include": [{"column": "meter_type", "value": "electricity"}],
            },
            "expect_tools": ["train_forecast_model*", "combine_forecasts"],
        },
        {
            "query": "Find anomalies in the electricity meter data and explain them",
            "dataset": "building_energy",
            "expect_plan": {
                "task_type": "anomaly", "needs_explanation": True,
                "data_filters_include": [{"column": "meter_type", "value": "electricity"}],
            },
            "expect_tools": ["detect_anomalies", "explain_anomalies"],
        },
    ],

    # ------------------------------------------------------------------
    # 9. Both (Forecast + Anomaly)
    # ------------------------------------------------------------------
    "Both Forecast+Anomaly": [
        {
            "query": "Forecast meter reading for 7 days and also detect anomalies",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "both", "horizon": 168},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "detect_anomalies"],
        },
        {
            "query": "Run both forecasting and anomaly detection on this data",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "both"},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "detect_anomalies"],
        },
        {
            "query": "Predict the next 3 days and check for outliers",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "both", "horizon": 72},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "detect_anomalies"],
        },
        {
            "query": "I want a full analysis: forecast 24 steps and detect anomalies",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "both", "horizon": 24},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "detect_anomalies"],
        },
        {
            "query": "Do both forecast and anomaly detection on OT",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "both"},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "detect_anomalies"],
        },
    ],

    # ------------------------------------------------------------------
    # 10. Report Generation
    # ------------------------------------------------------------------
    "Report Generation": [
        {
            "query": "Forecast OT for 7 days and generate a full report",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "forecast", "needs_report": True, "horizon": 168},
            "expect_tools": ["train_forecast_model*", "combine_forecasts", "generate_report"],
        },
        {
            "query": "Run anomaly detection and write a report",
            "dataset": "ETTh1",
            "expect_plan": {"task_type": "anomaly", "needs_report": True},
            "expect_tools": ["detect_anomalies", "generate_report"],
        },
        {
            "query": "Complete analysis with forecast, explanation, and report for 3 days",
            "dataset": "ETTh1",
            "expect_plan": {"needs_explanation": True, "needs_report": True, "horizon": 72},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts", "generate_report"],
        },
        {
            "query": "Predict next week with explanations and generate a summary report",
            "dataset": "ETTh1",
            "expect_plan": {"needs_explanation": True, "needs_report": True, "horizon": 168},
            "expect_tools": ["train_and_explain_forecast*", "combine_forecasts", "generate_report"],
        },
        {
            "query": "Detect anomalies, explain them, and produce a report",
            "dataset": "building_energy",
            "expect_plan": {"task_type": "anomaly", "needs_explanation": True, "needs_report": True},
            "expect_tools": ["detect_anomalies", "explain_anomalies", "generate_report"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _get_dataset_config(name: str) -> Dict[str, Any]:
    base = Path(__file__).parent / "data"
    configs = {
        "ETTh1": {
            "path": str(base / "test" / "ETTh1.csv"),
            "time_column": "date",
            "target_column": "OT",
            "feature_columns": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        },
        "building_energy": {
            "path": str(base / "raw" / "building_energy_data.csv"),
            "time_column": "timestamp",
            "target_column": "meter_reading",
            "feature_columns": [
                "air_temperature", "cloud_coverage", "dew_temperature",
                "wind_speed", "wind_direction", "sea_level_pressure",
            ],
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(configs)}")
    return configs[name]


# ---------------------------------------------------------------------------
# Flatten helper
# ---------------------------------------------------------------------------

def _flatten_queries(
    categories: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    qset = {k: v for k, v in QUERY_SET.items() if k in categories} if categories else QUERY_SET
    flat = [(cat, q) for cat, queries in qset.items() for q in queries]
    return flat[:max_queries] if max_queries else flat


# ===================================================================
#  PHASE 1: COLLECT  — LLM calls, dump raw plans to JSON
# ===================================================================

def run_collect(
    categories: Optional[List[str]] = None,
    max_queries: Optional[int] = None,
    output_path: str = "benchmark_results.json",
    parallel: int = 1,
    api_keys: Optional[List[str]] = None,
):
    """Call agent.plan() for each query and save raw UserTaskPlan fields to JSON.

    Creates ONE TinyTSAgent per dataset and reuses it for all queries
    on that dataset — single CSV load + single profile per dataset.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from tinyts.agent import TinyTSAgent

    all_queries = _flatten_queries(categories, max_queries)
    total = len(all_queries)

    print(f"\n{'='*80}")
    print(f"  COLLECT PHASE — {total} queries")
    if parallel > 1:
        print(f"  Workers: {parallel}, API keys: {len(api_keys or [])}")
    print(f"{'='*80}\n")

    # Pre-create agents (one per dataset) to avoid repeated CSV load + profiling
    needed_datasets = set(q["dataset"] for _, q in all_queries)
    agents: Dict[str, Any] = {}
    for ds_name in needed_datasets:
        ds = _get_dataset_config(ds_name)
        print(f"  Loading dataset '{ds_name}'...", end=" ", flush=True)
        agent = TinyTSAgent(
            dataset_path=ds["path"],
            time_column=ds["time_column"],
            target_column=ds["target_column"],
            feature_columns=ds["feature_columns"],
        )
        # Pre-run profile so it's cached in agent.session["profile"]
        agent.tool_map["profile_dataset"].invoke({})
        agents[ds_name] = agent
        print("done.")

    print()

    # Worker function
    def _collect_one(idx, cat, qdef, key=None):
        if key:
            os.environ["CEREBRAS_API_KEY"] = key

        query = qdef["query"]
        ds_name = qdef["dataset"]
        entry = {
            "category": cat,
            "query": query,
            "dataset": ds_name,
            "plan": None,
            "error": None,
            "elapsed": 0.0,
        }

        try:
            # Reuse pre-loaded agent — plan() will skip profiling since
            # session["profile"] already exists, it just calls the LLM.
            agent = agents[ds_name]
            t0 = time.time()
            plan = agent.plan(query)
            entry["elapsed"] = round(time.time() - t0, 2)

            # Serialize plan to plain dict
            entry["plan"] = plan.model_dump()
            print(f"  [{idx}/{total}] OK ({entry['elapsed']}s) [{cat}] {query}")

        except Exception as e:
            entry["error"] = str(e)
            print(f"  [{idx}/{total}] ERROR [{cat}] {query}: {e}")

        return entry

    # Run
    collected = []
    if parallel > 1 and api_keys and len(api_keys) > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for idx, (cat, qdef) in enumerate(all_queries, 1):
                key = api_keys[(idx - 1) % len(api_keys)]
                f = executor.submit(_collect_one, idx, cat, qdef, key)
                futures[f] = idx
            for f in as_completed(futures):
                try:
                    collected.append(f.result())
                except Exception as e:
                    print(f"  CRASHED: {e}")
    else:
        for idx, (cat, qdef) in enumerate(all_queries, 1):
            collected.append(_collect_one(idx, cat, qdef))

    # Save
    with open(output_path, "w") as f:
        json.dump(collected, f, indent=2)

    ok = sum(1 for c in collected if c["plan"] is not None)
    err = sum(1 for c in collected if c["error"] is not None)
    times = [c["elapsed"] for c in collected if c["elapsed"] > 0]
    avg_t = sum(times) / len(times) if times else 0

    print(f"\n  Collected: {ok}/{total} plans ({err} errors)")
    print(f"  Avg LLM time: {avg_t:.1f}s")
    print(f"  Saved to: {output_path}\n")


# ===================================================================
#  PHASE 2: EVALUATE  — pure offline comparison, zero imports
# ===================================================================

def _check_plan(plan: dict, expect: dict) -> List[dict]:
    """Compare one collected plan dict against expected fields. Returns list of checks."""
    checks = []

    for key, expected in expect.items():
        if key == "horizon":
            actual = plan.get("horizon")
            ok = actual is not None and actual == expected or actual*24 == expected
            checks.append({"field": "horizon", "expected": expected, "actual": actual, "passed": ok,
                           "note": "" if ok else f"{actual} != {expected}"})

        elif key == "models_subset":
            actual = plan.get("models_included", [])
            ok = bool(set(actual) & set(expected))
            checks.append({"field": "models_subset", "expected": expected, "actual": actual, "passed": ok,
                           "note": "" if ok else f"no overlap with {expected}"})

        elif key == "feature_columns_include":
            actual = plan.get("feature_columns", [])
            missing = [c for c in expected if c not in actual]
            ok = len(missing) == 0
            checks.append({"field": "feature_columns", "expected": expected, "actual": actual, "passed": ok,
                           "note": "" if ok else f"missing: {missing}"})

        elif key == "data_filters_include":
            actual_filters = plan.get("data_filters", [])
            ok = True
            for ef in expected:
                found = any(
                    f.get("column") == ef["column"] and f.get("value") == ef["value"]
                    for f in actual_filters
                )
                if not found:
                    ok = False
            checks.append({"field": "data_filters", "expected": expected, "actual": actual_filters,
                           "passed": ok, "note": "" if ok else "filter not found"})

        elif key == "counterfactual_target_value":
            actual = plan.get("counterfactual_target_value")
            if expected is None:
                ok = actual is None
            elif actual is None:
                ok = False
            else:
                ok = abs(actual - expected) / max(abs(expected), 1e-9) < 0.1
            checks.append({"field": key, "expected": expected, "actual": actual, "passed": ok, "note": ""})

        else:
            actual = plan.get(key, "MISSING")
            ok = actual == expected
            checks.append({"field": key, "expected": expected, "actual": actual, "passed": ok,
                           "note": "" if ok else f"{actual} != {expected}"})

    return checks


def _check_tool_sequence(actual_calls: List[str], expected_pattern: List[str]) -> Tuple[str, str]:
    """Match tool call sequence. Returns (verdict, note)."""
    filtered = [t for t in actual_calls if t != "profile_dataset"]
    if not filtered:
        return "FAILURE", "no tool calls (excluding profile_dataset)"

    pattern = []
    for entry in expected_pattern:
        if entry.endswith("*"):
            pattern.append((entry[:-1], True))
        else:
            pattern.append((entry, False))

    pi, ai, matched = 0, 0, 0
    while pi < len(pattern) and ai < len(filtered):
        pname, repeatable = pattern[pi]
        if filtered[ai] == pname:
            matched += 1
            if repeatable:
                while ai < len(filtered) and filtered[ai] == pname:
                    ai += 1
            else:
                ai += 1
            pi += 1
        else:
            ai += 1

    total_groups = len(pattern)
    if pi < total_groups:
        missing = [p[0] for p in pattern[pi:]]
        ratio = matched / total_groups if total_groups else 0
        if ratio >= 0.6:
            return "PARTIAL", f"missing: {missing}"
        return "FAILURE", f"missing: {missing}"
    return "SUCCESS", ""


def run_evaluate(input_path: str = "benchmark_results.json"):
    """Load collected plans from JSON and compare against ground truth.

    This is pure Python dict comparison — no ML imports, no compute.
    """
    with open(input_path) as f:
        collected = json.load(f)

    # Build lookup: (category, query) -> ground truth
    gt_lookup = {}
    for cat, queries in QUERY_SET.items():
        for q in queries:
            gt_lookup[(cat, q["query"])] = q

    total = len(collected)
    results = []

    print(f"\n{'='*80}")
    print(f"  EVALUATE PHASE — {total} collected plans")
    print(f"{'='*80}\n")

    for entry in collected:
        cat = entry["category"]
        query = entry["query"]
        plan = entry.get("plan")
        error = entry.get("error")

        gt = gt_lookup.get((cat, query))
        if gt is None:
            print(f"  SKIP: no ground truth for [{cat}] {query}")
            continue

        expect_plan = gt["expect_plan"]
        expect_tools = gt["expect_tools"]

        r = {
            "category": cat,
            "query": query,
            "dataset": entry.get("dataset", ""),
            "elapsed": entry.get("elapsed", 0),
            "plan_checks": [],
            "plan_verdict": "FAILURE",
            "exec_verdict": "NOT_RUN",
            "overall_verdict": "FAILURE",
            "needs_human": False,
            "human_reason": "",
            "plan_error": error or "",
            "tool_calls_actual": entry.get("tool_calls", []),
        }

        if plan is None:
            r["plan_verdict"] = "FAILURE"
            r["plan_error"] = error or "no plan returned"
        else:
            checks = _check_plan(plan, expect_plan)
            r["plan_checks"] = checks
            n_pass = sum(1 for c in checks if c["passed"])
            n_total = len(checks)
            if n_total == 0:
                r["plan_verdict"] = "SUCCESS"
            else:
                ratio = n_pass / n_total
                r["plan_verdict"] = "SUCCESS" if ratio == 1.0 else ("PARTIAL" if ratio >= 0.5 else "FAILURE")

        # If tool_calls were recorded (from execution runs), evaluate them too
        tool_calls = entry.get("tool_calls", [])
        if tool_calls:
            r["exec_verdict"], _ = _check_tool_sequence(tool_calls, expect_tools)

        # Overall
        verdicts = [r["plan_verdict"], r["exec_verdict"]]
        if "FAILURE" in verdicts:
            r["overall_verdict"] = "FAILURE"
        elif "PARTIAL" in verdicts:
            r["overall_verdict"] = "PARTIAL"
        elif r["exec_verdict"] == "NOT_RUN":
            r["overall_verdict"] = r["plan_verdict"]
        else:
            r["overall_verdict"] = "SUCCESS"

        if r["overall_verdict"] != "SUCCESS":
            r["needs_human"] = True
            reasons = []
            if r["plan_verdict"] != "SUCCESS":
                failed = [c["field"] for c in r["plan_checks"] if not c["passed"]]
                reasons.append(f"plan: {failed}")
            if r["exec_verdict"] not in ("SUCCESS", "NOT_RUN"):
                reasons.append("tool sequence")
            r["human_reason"] = "; ".join(reasons)

        # Print per-query
        status_icon = {"SUCCESS": "PASS", "PARTIAL": "PART", "FAILURE": "FAIL"}
        print(f"  [{status_icon.get(r['plan_verdict'], '????')}] [{cat}] {query}")
        for c in r["plan_checks"]:
            mark = "+" if c["passed"] else "x"
            print(f"    {mark} {c['field']}: expected={c['expected']}, got={c['actual']}", end="")
            if c["note"]:
                print(f"  ({c['note']})", end="")
            print()
        if r["plan_error"] and not plan:
            print(f"    ERROR: {r['plan_error']}")

        results.append(r)

    # ---- Summary ----
    success = sum(1 for r in results if r["overall_verdict"] == "SUCCESS")
    partial = sum(1 for r in results if r["overall_verdict"] == "PARTIAL")
    failure = sum(1 for r in results if r["overall_verdict"] == "FAILURE")
    human_needed = sum(1 for r in results if r["needs_human"])

    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  Total:     {total}")
    print(f"  SUCCESS:   {success:3d}  ({100*success/total:.1f}%)")
    print(f"  PARTIAL:   {partial:3d}  ({100*partial/total:.1f}%)")
    print(f"  FAILURE:   {failure:3d}  ({100*failure/total:.1f}%)")
    print(f"  Human:     {human_needed:3d}  ({100*human_needed/total:.1f}%)")
    print()

    # Per-category
    cat_stats: Dict[str, Dict] = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "success": 0, "partial": 0, "failure": 0}
        cat_stats[c]["total"] += 1
        cat_stats[c][r["overall_verdict"].lower()] += 1

    print(f"  {'Category':<28s} {'N':>3s} {'OK':>3s} {'PT':>3s} {'FL':>3s} {'Rate':>6s}")
    print(f"  {'-'*28} {'-'*3} {'-'*3} {'-'*3} {'-'*3} {'-'*6}")
    for cat, s in cat_stats.items():
        rate = 100 * s["success"] / s["total"]
        print(f"  {cat:<28s} {s['total']:>3d} {s['success']:>3d} {s['partial']:>3d} {s['failure']:>3d} {rate:>5.1f}%")
    print()

    # Problem queries
    problems = [r for r in results if r["overall_verdict"] != "SUCCESS"]
    if problems:
        print(f"  NEEDS HUMAN INTERVENTION ({len(problems)}):")
        print(f"  {'-'*60}")
        for r in problems:
            print(f"  [{r['category']}] {r['query']}")
            print(f"    verdict={r['plan_verdict']}, reason={r['human_reason']}")
            if r["plan_error"]:
                print(f"    error: {r['plan_error']}")
        print()

    # Timing
    times = [r["elapsed"] for r in results if r["elapsed"] > 0]
    if times:
        print(f"  Timing (from collect):")
        print(f"    avg={sum(times)/len(times):.1f}s  min={min(times):.1f}s  max={max(times):.1f}s")
        print()

    # Save evaluated results
    eval_path = input_path.replace(".json", "_evaluated.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Evaluated results saved to: {eval_path}")

    return results


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="TinyTS Query Success Rate Benchmark")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true", help="Phase 1: call LLM and save plans")
    mode.add_argument("--evaluate", action="store_true", help="Phase 2: offline comparison")
    mode.add_argument("--dry-run", action="store_true", help="Print query set")

    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--input", type=str, default="benchmark_results.json")
    parser.add_argument("--log-level", type=str, default="WARNING")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--api-keys", type=str, default=None,
                        help="Comma-separated Cerebras API keys")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    categories = [args.category] if args.category else None

    if args.dry_run:
        flat = _flatten_queries(categories, args.max_queries)
        print(f"\n  DRY RUN — {len(flat)} queries\n")
        for cat, q in flat:
            print(f"  [{cat}] {q['query']}  (dataset={q['dataset']})")
        print()
        return

    if args.collect:
        api_keys = None
        if args.api_keys:
            api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
        elif args.parallel > 1:
            api_keys = []
            for i in range(1, args.parallel + 1):
                k = os.environ.get(f"CEREBRAS_API_KEY_{i}")
                if k:
                    api_keys.append(k)
            if not api_keys:
                single = os.environ.get("CEREBRAS_API_KEY", "")
                if single:
                    api_keys = [single]
                    print(f"  WARNING: --parallel={args.parallel} but only 1 API key.\n")

        run_collect(
            categories=categories,
            max_queries=args.max_queries,
            output_path=args.output,
            parallel=args.parallel,
            api_keys=api_keys,
        )

    elif args.evaluate:
        run_evaluate(input_path=args.input)


if __name__ == "__main__":
    main()
