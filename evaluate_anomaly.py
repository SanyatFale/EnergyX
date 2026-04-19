#!/usr/bin/env python3
"""
evaluate_anomaly.py — Synthetic anomaly injection benchmark for EnergyX.

Injects three anomaly types into ASHRAE and ETT test sets, runs all 7 individual
detection methods + consensus ensemble, and reports Precision / Recall / F1 / AUROC
per method and per anomaly type.  Also analyses ensemble calibration across vote
thresholds (2/7 … 7/7) to show that high-agreement detections are more precise.

This directly calls tinyts/tools/anomaly.py (unchanged) via .invoke().

Usage
-----
  python evaluate_anomaly.py                  # ASHRAE + ETTh1/h2
  python evaluate_anomaly.py --ashrae-only    # ASHRAE datasets only
  python evaluate_anomaly.py --seed 99        # different injection seed

Results saved to:
  outputs/benchmark/anomaly_evaluation_results.csv
  outputs/benchmark/anomaly_per_type_results.csv
  outputs/benchmark/anomaly_calibration_results.csv
"""

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tinyts.tools.anomaly import (
    run_statistical_anomaly_detection,   # ZScore
    run_modified_zscore,                 # MAD
    run_rolling_anomaly,                 # RollingStats
    run_iqr_anomaly,                     # IQR
    run_stl_anomaly,                     # STL
    run_isolation_forest,                # IsolationForest
    run_dbscan_anomaly,                  # DBSCAN
    run_anomaly_ensemble,                # Consensus ensemble (3/7 default)
)

# ── Constants ──────────────────────────────────────────────────────────────────

SEED        = 42
TRAIN_RATIO = 0.8

# Injection rates
SPIKE_RATE      = 0.02   # fraction of n → point spikes
SHIFT_COUNT     = 2      # number of level shift windows
SHIFT_LEN       = 24     # length of each shift window (hours)
CONTEXTUAL_RATE = 0.01   # fraction of n → contextual anomalies

ASHRAE_DATASETS = {
    "ASHRAE_elec":    ROOT / "data/raw/building_energy_data (copy 1)_elec.csv",
    "ASHRAE_chilled": ROOT / "data/raw/building_energy_data (copy 1)_chilled.csv",
}
ETT_DATASETS = {
    "ETTh1": ROOT / "data/test/ETTh1.csv",
    "ETTh2": ROOT / "data/test/ETTh2.csv",
}

ASHRAE_TARGET = "meter_reading"
ETT_TARGET    = "OT"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_ashrae_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=[ASHRAE_TARGET])
    return df[ASHRAE_TARGET].values.astype(float)


def load_ett_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("date").reset_index(drop=True)
    return df[ETT_TARGET].values.astype(float)


# ── Anomaly injection ──────────────────────────────────────────────────────────

def inject_anomalies(
    y: np.ndarray,
    seasonal_period: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Inject three types of anomalies into a clean series.

    Types
    -----
    point_spike    — single-point deviations at ±5σ from the global mean.
                     Detectable by all global statistical methods (Z-score, IQR, etc.)

    level_shift    — a sustained +3σ shift over a SHIFT_LEN-hour window.
                     Best detected by Rolling Stats, STL (temporal context methods).
                     Harder for point-wise global methods.

    contextual     — a value that sits within the global 2σ bounds (not a global
                     spike) but is far above the *local* rolling mean, making it
                     anomalous in context.  Injected during "quiet" periods.
                     Best detected by Rolling Stats / STL; global methods
                     (Z-score, IQR, MAD) typically miss these.

    Returns
    -------
    y_contaminated  : np.ndarray — series with injections
    gt_labels       : dict mapping type → bool array (True = anomaly position)
    """
    n      = len(y)
    y_mean = float(np.mean(y))
    y_std  = float(np.std(y))

    y_contaminated = y.copy()
    gt_spike       = np.zeros(n, dtype=bool)
    gt_shift       = np.zeros(n, dtype=bool)
    gt_contextual  = np.zeros(n, dtype=bool)

    # ── 1. Point spikes ───────────────────────────────────────────────────────
    n_spikes = max(5, int(SPIKE_RATE * n))
    # Avoid the last 2*SHIFT_LEN positions (reserved for level shifts)
    spike_candidates = np.arange(n - SHIFT_LEN * 2)
    spike_idx = rng.choice(spike_candidates, size=n_spikes, replace=False)
    for i, idx in enumerate(spike_idx):
        sign = 1 if i % 2 == 0 else -1
        y_contaminated[idx] = y_mean + sign * 5.0 * y_std
    gt_spike[spike_idx] = True

    # ── 2. Level shifts ───────────────────────────────────────────────────────
    # Place SHIFT_COUNT non-overlapping windows in the middle 50% of the series
    quarter     = n // 4
    shift_starts: List[int] = []
    attempts    = 0
    used_ranges: set = set()
    while len(shift_starts) < SHIFT_COUNT and attempts < 200:
        start = rng.randint(quarter, 3 * quarter - SHIFT_LEN)
        new_r = set(range(start, start + SHIFT_LEN))
        if not new_r & used_ranges:
            shift_starts.append(start)
            used_ranges |= new_r
        attempts += 1
    for start in shift_starts:
        y_contaminated[start:start + SHIFT_LEN] += 3.0 * y_std
        gt_shift[start:start + SHIFT_LEN] = True

    # ── 3. Contextual anomalies ───────────────────────────────────────────────
    # Injection value: global_mean + 1.5σ  (within the global 3σ window).
    # Injection sites: "quiet" periods where local rolling mean < P10 of all
    # rolling means, so the injected value is ≥ 4σ_local above the local mean.
    local_window = max(6, seasonal_period // 4)
    rolling_mean = (
        pd.Series(y)
        .rolling(window=local_window, min_periods=1)
        .mean()
        .values
    )
    quiet_thr = float(np.percentile(rolling_mean, 10))
    quiet_mask = rolling_mean <= quiet_thr

    # Exclude positions already marked anomalous
    already = gt_spike | gt_shift
    ctx_candidates = np.where(quiet_mask & ~already)[0]

    injection_value = y_mean + 1.5 * y_std  # within global 2.5σ
    n_contextual = max(3, int(CONTEXTUAL_RATE * n))

    if len(ctx_candidates) >= n_contextual:
        ctx_idx = rng.choice(ctx_candidates, size=n_contextual, replace=False)
        y_contaminated[ctx_idx] = injection_value
        gt_contextual[ctx_idx] = True

    return y_contaminated, {
        "point_spike": gt_spike,
        "level_shift": gt_shift,
        "contextual":  gt_contextual,
    }


# ── Method execution ───────────────────────────────────────────────────────────

INDIVIDUAL_METHODS = [
    ("ZScore",          run_statistical_anomaly_detection, {}),
    ("MAD",             run_modified_zscore,               {}),
    ("RollingStats",    run_rolling_anomaly,               {}),
    ("IQR",             run_iqr_anomaly,                   {}),
    ("STL",             run_stl_anomaly,                   {}),
    ("IsolationForest", run_isolation_forest,              {}),
    ("DBSCAN",          run_dbscan_anomaly,                {}),
]


def _call_tool(tool, y_json: str, extra: dict) -> Dict:
    """Invoke an anomaly tool and return the parsed result dict."""
    n = len(json.loads(y_json))
    try:
        raw = tool.invoke({"data": y_json, **extra})
        return json.loads(raw)
    except Exception as e:
        return {
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
            "n_anomalies": 0,
            "metadata": {"error": str(e)},
        }


def _call_ensemble(y_json: str, min_votes: int = 3) -> Dict:
    n = len(json.loads(y_json))
    try:
        raw = run_anomaly_ensemble.invoke({"data": y_json, "min_votes": min_votes})
        return json.loads(raw)
    except Exception as e:
        return {
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
            "n_anomalies": 0,
            "method_counts": {},
            "metadata": {"error": str(e)},
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_labels: np.ndarray,   # -1 = anomaly, 1 = normal
    pred_scores: np.ndarray,   # continuous scores (higher = more anomalous)
    gt_binary: np.ndarray,     # 0/1 ground truth (1 = anomaly)
) -> Dict[str, float]:
    if gt_binary.sum() == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "auroc": np.nan}

    pred_binary = (pred_labels == -1).astype(int)
    prec = float(precision_score(gt_binary, pred_binary, zero_division=0))
    rec  = float(recall_score(gt_binary, pred_binary, zero_division=0))
    f1   = float(f1_score(gt_binary, pred_binary, zero_division=0))

    try:
        s_min, s_max = pred_scores.min(), pred_scores.max()
        scores_norm = (
            (pred_scores - s_min) / (s_max - s_min)
            if s_max > s_min else pred_scores
        )
        auroc = float(roc_auc_score(gt_binary, scores_norm))
    except Exception:
        auroc = np.nan

    return {"precision": prec, "recall": rec, "f1": f1, "auroc": auroc}


def compute_per_type_f1(
    pred_labels: np.ndarray,
    gt_by_type: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """F1 score for each anomaly type independently (treating only that type as positive)."""
    pred_binary = (pred_labels == -1).astype(int)
    results: Dict[str, float] = {}
    for atype, gt_mask in gt_by_type.items():
        if gt_mask.sum() == 0:
            results[atype] = np.nan
            continue
        results[atype] = float(f1_score(gt_mask.astype(int), pred_binary, zero_division=0))
    return results


def analyze_vote_calibration(
    vote_scores: np.ndarray,   # integer vote counts (0–7) per timestep
    gt_binary: np.ndarray,     # 0/1 ground truth
) -> pd.DataFrame:
    """
    Sweep vote thresholds 2–7.  For each threshold t, flag points with votes ≥ t
    and compute Precision / Recall / F1.
    Shows high-agreement points (≥5/7 votes) have higher precision than 3/7.
    """
    rows = []
    for t in range(2, 8):
        pred = (vote_scores >= t).astype(int)
        prec = float(precision_score(gt_binary, pred, zero_division=0))
        rec  = float(recall_score(gt_binary, pred, zero_division=0))
        f1   = float(f1_score(gt_binary, pred, zero_division=0))
        rows.append({
            "min_votes": t,
            "n_flagged": int(pred.sum()),
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
    return pd.DataFrame(rows)


# ── Per-dataset evaluation ─────────────────────────────────────────────────────

def evaluate_dataset(
    ds_name: str,
    y_test: np.ndarray,
    seasonal_period: int,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(y_test)
    y_contaminated, gt_by_type = inject_anomalies(y_test, seasonal_period, rng)

    gt_all = (
        gt_by_type["point_spike"]
        | gt_by_type["level_shift"]
        | gt_by_type["contextual"]
    ).astype(int)

    n_spike = int(gt_by_type["point_spike"].sum())
    n_shift = int(gt_by_type["level_shift"].sum())
    n_ctx   = int(gt_by_type["contextual"].sum())
    n_total = int(gt_all.sum())
    print(f"  n={n}  injected: {n_spike} spikes + {n_shift} shift-pts + {n_ctx} contextual = {n_total} total ({100*n_total/n:.1f}%)")

    y_json = json.dumps(y_contaminated.tolist())

    method_rows: List[dict] = []
    per_type_rows: List[dict] = []
    ens_vote_scores: np.ndarray | None = None

    # ── Individual methods ─────────────────────────────────────────────────────
    for mname, tool, kwargs in INDIVIDUAL_METHODS:
        print(f"    {mname:<20s}", end="", flush=True)
        t0  = time.perf_counter()
        res = _call_tool(tool, y_json, kwargs)
        elapsed = time.perf_counter() - t0

        labels = np.array(res["anomaly_labels"])
        scores = np.array(res["anomaly_scores"])

        m = compute_metrics(labels, scores, gt_all)
        pt = compute_per_type_f1(labels, gt_by_type)

        print(
            f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
            f"  AUROC={m['auroc']:.3f}  ({elapsed:.1f}s)"
        )

        method_rows.append({
            "dataset": ds_name, "method": mname,
            "n_detected": int((labels == -1).sum()),
            **m,
        })
        for atype, f1_val in pt.items():
            per_type_rows.append({
                "dataset": ds_name, "method": mname,
                "anomaly_type": atype, "f1": f1_val,
            })

    # ── Ensemble (min_votes = 3) ───────────────────────────────────────────────
    print(f"    {'Ensemble (3/7)':<20s}", end="", flush=True)
    t0      = time.perf_counter()
    ens_res = _call_ensemble(y_json, min_votes=3)
    elapsed = time.perf_counter() - t0

    ens_labels      = np.array(ens_res["anomaly_labels"])
    ens_vote_scores = np.array(ens_res["anomaly_scores"], dtype=float)

    m  = compute_metrics(ens_labels, ens_vote_scores, gt_all)
    pt = compute_per_type_f1(ens_labels, gt_by_type)

    print(
        f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
        f"  AUROC={m['auroc']:.3f}  ({elapsed:.1f}s)"
    )

    method_rows.append({
        "dataset": ds_name, "method": "Ensemble_3/7",
        "n_detected": int((ens_labels == -1).sum()),
        **m,
    })
    for atype, f1_val in pt.items():
        per_type_rows.append({
            "dataset": ds_name, "method": "Ensemble_3/7",
            "anomaly_type": atype, "f1": f1_val,
        })

    # ── Vote calibration analysis ──────────────────────────────────────────────
    cal_df = analyze_vote_calibration(ens_vote_scores, gt_all)
    cal_df.insert(0, "dataset", ds_name)

    return (
        pd.DataFrame(method_rows),
        pd.DataFrame(per_type_rows),
        cal_df,
    )


# ── Display ────────────────────────────────────────────────────────────────────

def print_method_table(df: pd.DataFrame, ds_name: str):
    sub = df[df["dataset"] == ds_name]
    print(f"\n  {'Method':<22s} {'Precision':>9s} {'Recall':>9s} {'F1':>9s} {'AUROC':>9s} {'N_det':>6s}")
    print(f"  {'─'*68}")
    for _, row in sub.iterrows():
        nm  = str(row["method"])
        mrk = " ◄" if nm == "Ensemble_3/7" else ""
        print(
            f"  {nm:<22s} {row['precision']:>9.3f} {row['recall']:>9.3f}"
            f" {row['f1']:>9.3f} {row['auroc']:>9.3f} {int(row['n_detected']):>6d}{mrk}"
        )


def print_calibration_table(cal_df: pd.DataFrame, ds_name: str):
    sub = cal_df[cal_df["dataset"] == ds_name]
    print(f"\n  Ensemble vote-threshold calibration:")
    print(f"  {'min_votes':>10s} {'n_flagged':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'─'*55}")
    for _, row in sub.iterrows():
        mrk = " ◄ default" if int(row["min_votes"]) == 3 else ""
        print(
            f"  {int(row['min_votes']):>10d} {int(row['n_flagged']):>10d}"
            f" {row['precision']:>10.3f} {row['recall']:>10.3f}"
            f" {row['f1']:>10.3f}{mrk}"
        )


def print_per_type_summary(per_type_df: pd.DataFrame):
    print("\n  Per-anomaly-type F1 (mean across datasets):")
    agg = (
        per_type_df
        .groupby(["method", "anomaly_type"])["f1"]
        .mean()
        .unstack()
        .reindex(columns=["point_spike", "level_shift", "contextual"])
        .sort_values("point_spike", ascending=False)
    )
    print(f"  {'Method':<22s} {'Spike':>8s} {'Shift':>8s} {'Ctx':>8s}")
    print(f"  {'─'*50}")
    for mname, row in agg.iterrows():
        print(
            f"  {str(mname):<22s}"
            f" {row.get('point_spike', np.nan):>8.3f}"
            f" {row.get('level_shift', np.nan):>8.3f}"
            f" {row.get('contextual',  np.nan):>8.3f}"
        )


def print_global_summary(method_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  GLOBAL SUMMARY — Mean across all datasets")
    print("=" * 70)
    agg = (
        method_df
        .groupby("method")[["precision", "recall", "f1", "auroc"]]
        .mean()
        .sort_values("f1", ascending=False)
    )
    print(f"  {'Method':<22s} {'Precision':>9s} {'Recall':>9s} {'F1':>9s} {'AUROC':>9s}")
    print(f"  {'─'*55}")
    for mname, row in agg.iterrows():
        mrk = " ◄" if mname == "Ensemble_3/7" else ""
        print(
            f"  {str(mname):<22s} {row['precision']:>9.3f} {row['recall']:>9.3f}"
            f" {row['f1']:>9.3f} {row['auroc']:>9.3f}{mrk}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly detection benchmark with synthetic injection")
    parser.add_argument("--ashrae-only", action="store_true", help="Skip ETT datasets")
    parser.add_argument("--seed",        type=int, default=SEED, help="RNG seed")
    args = parser.parse_args()

    t_start = time.perf_counter()

    datasets: Dict[str, Tuple[np.ndarray, int]] = {}

    for name, path in ASHRAE_DATASETS.items():
        if path.exists():
            y     = load_ashrae_series(path)
            split = int(len(y) * TRAIN_RATIO)
            datasets[name] = (y[split:], 24)
        else:
            print(f"  [skip] {name}: not found at {path}")

    if not args.ashrae_only:
        for name, path in ETT_DATASETS.items():
            if path.exists():
                y     = load_ett_series(path)
                split = int(len(y) * TRAIN_RATIO)
                datasets[name] = (y[split:], 24)
            else:
                print(f"  [skip] {name}: not found at {path}")

    all_method_dfs:   List[pd.DataFrame] = []
    all_per_type_dfs: List[pd.DataFrame] = []
    all_cal_dfs:      List[pd.DataFrame] = []

    for ds_name, (y_test, sp) in datasets.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  (sp={sp})")
        print(f"{'='*70}")

        # Deterministic per-dataset seed
        ds_rng = np.random.RandomState(args.seed + abs(hash(ds_name)) % 997)
        mdf, ptdf, cdf = evaluate_dataset(ds_name, y_test, sp, ds_rng)

        all_method_dfs.append(mdf)
        all_per_type_dfs.append(ptdf)
        all_cal_dfs.append(cdf)

        print_method_table(mdf, ds_name)
        print_calibration_table(cdf, ds_name)

    if not all_method_dfs:
        print("No datasets found. Exiting.")
        return

    all_method_df   = pd.concat(all_method_dfs,   ignore_index=True)
    all_per_type_df = pd.concat(all_per_type_dfs, ignore_index=True)
    all_cal_df      = pd.concat(all_cal_dfs,      ignore_index=True)

    print_global_summary(all_method_df)
    print_per_type_summary(all_per_type_df)

    # Save
    out_dir = ROOT / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_method_df.to_csv(out_dir / "anomaly_evaluation_results.csv", index=False)
    all_per_type_df.to_csv(out_dir / "anomaly_per_type_results.csv", index=False)
    all_cal_df.to_csv(out_dir / "anomaly_calibration_results.csv", index=False)

    elapsed = time.perf_counter() - t_start
    print(f"\nResults saved → {out_dir}/anomaly_*.csv")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
