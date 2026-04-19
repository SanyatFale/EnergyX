#!/usr/bin/env python3
"""
evaluate_lead.py — Real ground-truth anomaly detection benchmark using LEAD 1.0.

Evaluates all 7 individual detection methods + the production consensus ensemble
against LEAD 1.0 anomaly labels (human-verified) for the top 5 buildings by
anomaly count.  Each building is evaluated per meter type independently.

Unlike evaluate_anomaly.py (synthetic injection), this script uses real labels —
providing an externally credible evaluation not subject to injection-model bias.

Important caveat
----------------
LEAD 1.0 anomaly labels are assigned at the building level based on the electricity
meter.  For non-electricity meters (chilled water, steam, hot water) the same
building-level label is applied.  Interpret non-electricity results accordingly.

Usage
-----
  python evaluate_lead.py              # all 5 buildings, all meters
  python evaluate_lead.py --building 1319 1258    # subset of buildings

Results saved to:
  outputs/benchmark/lead_results_per_meter.csv
  outputs/benchmark/lead_results_per_building.csv
  outputs/benchmark/lead_results_global.csv
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from tinyts.tools.anomaly import (
    run_anomaly_ensemble,
    run_dbscan_anomaly,
    run_iqr_anomaly,
    run_isolation_forest,
    run_modified_zscore,
    run_rolling_anomaly,
    run_statistical_anomaly_detection,
    run_stl_anomaly,
)

# ── Config ─────────────────────────────────────────────────────────────────────

TOP5           = [1319, 1258, 439, 247, 1225]
CONSOLIDATED   = ROOT / "data" / "consolidated"
OUT_DIR        = ROOT / "outputs" / "benchmark"

# LEAD anomaly label ranks (from LEAD_ANOMALY_RANKING.md)
LEAD_RANK = {1319: 1, 1258: 2, 439: 3, 247: 4, 1225: 5}

METHODS = [
    ("ZScore",          run_statistical_anomaly_detection),
    ("MAD",             run_modified_zscore),
    ("RollingStats",    run_rolling_anomaly),
    ("IQR",             run_iqr_anomaly),
    ("STL",             run_stl_anomaly),
    ("IsolationForest", run_isolation_forest),
    ("DBSCAN",          run_dbscan_anomaly),
]

# ── Tool invocation ────────────────────────────────────────────────────────────

def _invoke(tool, y_json: str) -> Dict:
    n = len(json.loads(y_json))
    try:
        raw = tool.invoke({"data": y_json})
        return json.loads(raw)
    except Exception as e:
        return {
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
            "error": str(e),
        }


def _invoke_ensemble(y_json: str, min_votes: int = 3) -> Dict:
    n = len(json.loads(y_json))
    try:
        raw = run_anomaly_ensemble.invoke({"data": y_json, "min_votes": min_votes})
        return json.loads(raw)
    except Exception as e:
        return {
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
            "error": str(e),
        }


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    pred_labels: np.ndarray,   # -1 = anomaly, 1 = normal
    pred_scores: np.ndarray,
    gt_binary:   np.ndarray,   # 0/1 (1 = anomaly)
) -> Dict[str, float]:
    """Precision, Recall, F1, AUROC.  Returns NaN if no positives in GT."""
    if gt_binary.sum() == 0:
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, auroc=np.nan)

    pred_binary = (pred_labels == -1).astype(int)

    prec  = float(precision_score(gt_binary, pred_binary, zero_division=0))
    rec   = float(recall_score(gt_binary, pred_binary, zero_division=0))
    f1    = float(f1_score(gt_binary, pred_binary, zero_division=0))

    try:
        s_min, s_max = pred_scores.min(), pred_scores.max()
        scores_norm = (
            (pred_scores - s_min) / (s_max - s_min) if s_max > s_min
            else pred_scores
        )
        auroc = float(roc_auc_score(gt_binary, scores_norm))
    except Exception:
        auroc = np.nan

    n_detected = int(pred_binary.sum())
    n_tp = int((pred_binary & gt_binary).sum())
    n_fp = int((pred_binary & (1 - gt_binary)).sum())
    n_fn = int(((1 - pred_binary) & gt_binary).sum())

    return dict(
        precision=prec, recall=rec, f1=f1, auroc=auroc,
        n_detected=n_detected, n_tp=n_tp, n_fp=n_fp, n_fn=n_fn,
    )


# ── Per-meter evaluation ───────────────────────────────────────────────────────

def evaluate_meter(
    building_id: int,
    meter_type:  str,
    df:          pd.DataFrame,
) -> List[dict]:
    """
    Run all methods on a single building × meter series.
    Returns a list of result dicts (one per method).
    """
    # Series: drop NaN readings (keep index alignment via boolean mask)
    valid_mask = df["meter_reading"].notna()
    y_vals  = df.loc[valid_mask, "meter_reading"].values.astype(float)
    gt_raw  = df.loc[valid_mask, "anomaly"].values        # 1.0 or NaN

    # LEAD label: 1 → anomaly, NaN → normal
    gt_binary = np.where(np.isnan(gt_raw.astype(float)), 0, 1).astype(int)

    n_total = len(y_vals)
    n_gt    = int(gt_binary.sum())
    gt_rate = 100.0 * n_gt / n_total if n_total else 0.0

    if n_total < 10:
        print(f"    [skip] too few rows ({n_total})")
        return []

    y_json = json.dumps(y_vals.tolist())

    rows: List[dict] = []

    # ── Individual methods ─────────────────────────────────────────────────────
    for mname, tool in METHODS:
        t0  = time.perf_counter()
        res = _invoke(tool, y_json)
        elapsed = time.perf_counter() - t0

        labels = np.array(res["anomaly_labels"])
        scores = np.array(res["anomaly_scores"])
        m = compute_metrics(labels, scores, gt_binary)

        status = "ERR" if "error" in res else "ok"
        print(
            f"      {mname:<20s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
            f"  R={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}"
            f"  det={m.get('n_detected',0):4d}  TP={m.get('n_tp',0):4d}"
            f"  FP={m.get('n_fp',0):4d}  FN={m.get('n_fn',0):4d}"
            f"  ({elapsed:.1f}s){' ['+status+']' if status!='ok' else ''}"
        )
        rows.append(dict(
            building_id=building_id,
            lead_rank=LEAD_RANK[building_id],
            meter_type=meter_type,
            method=mname,
            n_series=n_total,
            n_gt_anomalies=n_gt,
            gt_rate_pct=round(gt_rate, 2),
            elapsed_s=round(elapsed, 2),
            **m,
        ))

    # ── Production ensemble (min_votes = 3/7) ─────────────────────────────────
    t0      = time.perf_counter()
    ens_res = _invoke_ensemble(y_json, min_votes=3)
    elapsed = time.perf_counter() - t0

    ens_labels = np.array(ens_res["anomaly_labels"])
    ens_scores = np.array(ens_res["anomaly_scores"], dtype=float)
    m = compute_metrics(ens_labels, ens_scores, gt_binary)

    status = "ERR" if "error" in ens_res else "ok"
    print(
        f"      {'Ensemble (3/7)':<20s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
        f"  R={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}"
        f"  det={m.get('n_detected',0):4d}  TP={m.get('n_tp',0):4d}"
        f"  FP={m.get('n_fp',0):4d}  FN={m.get('n_fn',0):4d}"
        f"  ({elapsed:.1f}s) ◄ PROD{' ['+status+']' if status!='ok' else ''}"
    )
    rows.append(dict(
        building_id=building_id,
        lead_rank=LEAD_RANK[building_id],
        meter_type=meter_type,
        method="Ensemble_3/7",
        n_series=n_total,
        n_gt_anomalies=n_gt,
        gt_rate_pct=round(gt_rate, 2),
        elapsed_s=round(elapsed, 2),
        **m,
    ))

    return rows


# ── Display helpers ────────────────────────────────────────────────────────────

def _bar(val: float, width: int = 20) -> str:
    """Tiny ASCII bar for F1 values."""
    if np.isnan(val):
        return " " * width
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled)


def print_building_summary(df: pd.DataFrame, building_id: int):
    sub = df[df["building_id"] == building_id]
    print(f"\n  {'Method':<20s} {'Meter':<16s} {'F1':>6s} {'P':>6s} {'R':>6s} "
          f"{'AUROC':>6s} {'GT':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s}  F1 bar")
    print(f"  {'─'*110}")
    for _, row in sub.sort_values(["meter_type", "f1"], ascending=[True, False]).iterrows():
        bar = _bar(row["f1"])
        mrk = " ◄" if row["method"] == "Ensemble_3/7" else ""
        print(
            f"  {row['method']:<20s} {row['meter_type']:<16s}"
            f" {row['f1']:>6.3f} {row['precision']:>6.3f} {row['recall']:>6.3f}"
            f" {row['auroc']:>6.3f} {int(row['n_gt_anomalies']):>5d}"
            f" {int(row['n_tp']):>5d} {int(row['n_fp']):>5d} {int(row['n_fn']):>5d}"
            f"  {bar}{mrk}"
        )


def print_global_summary(df: pd.DataFrame):
    print("\n" + "=" * 80)
    print("  GLOBAL SUMMARY — Mean across all buildings × meters")
    print("=" * 80)
    agg = (
        df.groupby("method")[["precision", "recall", "f1", "auroc"]]
        .mean()
        .sort_values("f1", ascending=False)
        .round(3)
    )
    print(f"  {'Method':<20s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'AUROC':>8s}  F1 bar")
    print(f"  {'─'*80}")
    for mname, row in agg.iterrows():
        bar = _bar(row["f1"])
        mrk = " ◄ PROD" if mname == "Ensemble_3/7" else ""
        print(
            f"  {str(mname):<20s} {row['precision']:>10.3f} {row['recall']:>8.3f}"
            f" {row['f1']:>8.3f} {row['auroc']:>8.3f}  {bar}{mrk}"
        )


def print_per_building_pivot(df: pd.DataFrame):
    """Show ensemble F1 per building × meter as a pivot."""
    print("\n  Ensemble (3/7) F1 per building × meter:")
    ens = df[df["method"] == "Ensemble_3/7"].copy()
    pivot = ens.pivot_table(
        index="building_id", columns="meter_type", values="f1", aggfunc="first"
    ).round(3)
    print(pivot.to_string())

    print("\n  Best method per building × meter (by F1):")
    best = (
        df.loc[df.groupby(["building_id", "meter_type"])["f1"].idxmax()]
        [["building_id", "meter_type", "method", "f1", "n_gt_anomalies"]]
        .sort_values(["building_id", "meter_type"])
        .reset_index(drop=True)
    )
    print(best.to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LEAD ground-truth anomaly evaluation")
    parser.add_argument(
        "--building", type=int, nargs="+", default=TOP5,
        help="Building IDs to evaluate (default: all 5)",
    )
    args = parser.parse_args()

    buildings = [b for b in args.building if b in TOP5]
    if not buildings:
        print("No valid building IDs. Choose from:", TOP5)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()

    all_rows: List[dict] = []

    for bid in buildings:
        path = CONSOLIDATED / f"building_{bid}_consolidated.csv"
        if not path.exists():
            print(f"\n[skip] {path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Building {bid}  (LEAD rank #{LEAD_RANK[bid]})")
        print(f"{'='*70}")

        df = pd.read_csv(path, parse_dates=["timestamp"])

        meters = sorted(df["meter_type"].unique())
        print(f"  Meters: {meters}")

        for mtype in meters:
            mdf = df[df["meter_type"] == mtype].sort_values("timestamp").reset_index(drop=True)
            n_gt = mdf["anomaly"].notna().sum()
            gt_rate = 100.0 * n_gt / len(mdf)
            print(f"\n  ── {mtype}  ({len(mdf):,} rows, GT anomalies: {n_gt} / {gt_rate:.1f}%)")

            rows = evaluate_meter(bid, mtype, mdf)
            all_rows.extend(rows)

        # Building summary
        bld_df = pd.DataFrame([r for r in all_rows if r["building_id"] == bid])
        if not bld_df.empty:
            print_building_summary(bld_df, bid)

    if not all_rows:
        print("No results produced.")
        return

    results_df = pd.DataFrame(all_rows)

    # ── Aggregate tables ───────────────────────────────────────────────────────

    # Per-meter granular results
    per_meter = results_df[[
        "building_id", "lead_rank", "meter_type", "method",
        "n_series", "n_gt_anomalies", "gt_rate_pct",
        "precision", "recall", "f1", "auroc",
        "n_detected", "n_tp", "n_fp", "n_fn", "elapsed_s",
    ]].sort_values(["building_id", "meter_type", "f1"], ascending=[True, True, False])

    # Per-building (mean across meters)
    per_building = (
        results_df
        .groupby(["building_id", "lead_rank", "method"])
        .agg(
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_auroc=("auroc", "mean"),
            total_gt_anomalies=("n_gt_anomalies", "sum"),
            total_tp=("n_tp", "sum"),
            total_fp=("n_fp", "sum"),
            total_fn=("n_fn", "sum"),
            n_meters=("meter_type", "nunique"),
        )
        .reset_index()
        .sort_values(["building_id", "mean_f1"], ascending=[True, False])
    )
    per_building = per_building.round({"mean_f1": 3, "mean_precision": 3,
                                        "mean_recall": 3, "mean_auroc": 3})

    # Global (mean across all buildings × meters)
    global_summary = (
        results_df
        .groupby("method")
        .agg(
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_auroc=("auroc", "mean"),
            total_tp=("n_tp", "sum"),
            total_fp=("n_fp", "sum"),
            total_fn=("n_fn", "sum"),
            n_building_meter_combos=("meter_type", "count"),
        )
        .reset_index()
        .sort_values("mean_f1", ascending=False)
        .round({"mean_f1": 3, "mean_precision": 3, "mean_recall": 3, "mean_auroc": 3})
    )

    # ── Print summaries ────────────────────────────────────────────────────────
    print_global_summary(results_df)
    print_per_building_pivot(results_df)

    print("\n  Per-building ensemble (3/7) performance:")
    ens_pb = per_building[per_building["method"] == "Ensemble_3/7"][[
        "building_id", "lead_rank", "n_meters",
        "total_gt_anomalies", "total_tp", "total_fp", "total_fn",
        "mean_f1", "mean_precision", "mean_recall",
    ]].sort_values("lead_rank")
    print(ens_pb.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    per_meter_path    = OUT_DIR / "lead_results_per_meter.csv"
    per_building_path = OUT_DIR / "lead_results_per_building.csv"
    global_path       = OUT_DIR / "lead_results_global.csv"

    per_meter.to_csv(per_meter_path, index=False)
    per_building.to_csv(per_building_path, index=False)
    global_summary.to_csv(global_path, index=False)

    elapsed = time.perf_counter() - t_wall
    print(f"\nSaved:")
    print(f"  {per_meter_path}")
    print(f"  {per_building_path}")
    print(f"  {global_path}")
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
