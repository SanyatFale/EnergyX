#!/usr/bin/env python3
"""
evaluate_lead_v2.py — LEAD ground-truth anomaly benchmark (STL-first pipeline).

5-step pipeline per building
─────────────────────────────
  Step 1 │ STL decomposition → residual  (all subsequent detectors run on residual)
  Step 2 │ Fast detectors:  Modified Z-score · Rolling Stats · IQR
  Step 3 │ Matrix Profile   (discord = unusual subsequences / rare shapes)
  Step 4 │ Isolation Forest on engineered features (rolling mean/std + lags)
  Step 5 │ Ensemble scoring (four strategies)

Usage
-----
  python evaluate_lead_v2.py                     # all 5 buildings
  python evaluate_lead_v2.py --building 439 1319 # subset

Results saved to:
  outputs/benchmark/lead_v2_per_building.csv
  outputs/benchmark/lead_v2_all_methods.csv
  outputs/benchmark/lead_v2_ensembles.csv
  outputs/benchmark/lead_v2_global.csv
  outputs/benchmark/lead_v2_global_ensembles.csv
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import stumpy
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parent
CONSOLIDATED = ROOT / "data" / "consolidated"
OUT_DIR      = ROOT / "outputs" / "benchmark"

TOP5      = [1319, 1258, 439, 247, 1225]
LEAD_RANK = {1319: 1, 1258: 2, 439: 3, 247: 4, 1225: 5}


# ── Utility ────────────────────────────────────────────────────────────────────

def _gt(df: pd.DataFrame) -> np.ndarray:
    return np.where(df["anomaly"].isna(), 0, df["anomaly"].fillna(0).astype(int)).astype(int)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month
    return df


def metrics(pred_binary: np.ndarray, pred_scores: np.ndarray, gt_binary: np.ndarray) -> dict:
    if gt_binary.sum() == 0:
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, auroc=np.nan,
                    n_detected=0, n_tp=0, n_fp=0, n_fn=0)
    prec = float(precision_score(gt_binary, pred_binary, zero_division=0))
    rec  = float(recall_score(gt_binary, pred_binary, zero_division=0))
    f1   = float(f1_score(gt_binary, pred_binary, zero_division=0))
    try:
        s_mn, s_mx = pred_scores.min(), pred_scores.max()
        sn    = (pred_scores - s_mn) / (s_mx - s_mn) if s_mx > s_mn else pred_scores
        auroc = float(roc_auc_score(gt_binary, sn))
    except Exception:
        auroc = np.nan
    return dict(
        precision=prec, recall=rec, f1=f1, auroc=auroc,
        n_detected=int(pred_binary.sum()),
        n_tp=int((pred_binary & gt_binary).sum()),
        n_fp=int((pred_binary & (1 - gt_binary)).sum()),
        n_fn=int(((1 - pred_binary) & gt_binary).sum()),
    )


# ── Step 1: STL decomposition ──────────────────────────────────────────────────

def stl_residual(y: np.ndarray, period: int = 24) -> np.ndarray:
    """
    Decompose y with STL and return the residual component.
    Falls back to raw y if STL fails (e.g. constant series).
    """
    n = len(y)
    if n < 2 * period + 1:
        return y.copy()
    try:
        result = STL(y, period=period, robust=True).fit()
        return result.resid
    except Exception:
        return y.copy()


# ── Step 2: Fast detectors (run on residual) ───────────────────────────────────

def detect_modified_zscore(r: np.ndarray, k: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """Modified Z-score (Iglewicz & Hoaglin) on the STL residual."""
    med    = np.median(r)
    mad    = np.median(np.abs(r - med)) or 1e-6
    scores = 0.6745 * np.abs(r - med) / mad
    return (scores > k).astype(int), scores


def detect_rolling_stats(
    r: np.ndarray,
    window: int = 24,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flag points outside k-sigma of a rolling mean±std on the residual."""
    s      = pd.Series(r)
    rm     = s.rolling(window, min_periods=1).mean()
    rs     = s.rolling(window, min_periods=1).std().fillna(1e-6)
    scores = ((s - rm) / rs).abs().values
    return (scores > k).astype(int), scores


def detect_iqr(r: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """IQR fence anomaly detection on the STL residual."""
    q1, q3 = np.percentile(r, 25), np.percentile(r, 75)
    iqr    = q3 - q1 or 1e-6
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    dist   = np.maximum(lo - r, r - hi)
    scores = np.clip(dist / iqr, 0, None)
    return (scores > 0).astype(int), scores


# ── Step 3: Matrix Profile ─────────────────────────────────────────────────────

def detect_matrix_profile(
    r: np.ndarray,
    m: int = 24,
    discord_pct: float = 97.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stumpy matrix profile on the STL residual.

    Discords (highest matrix profile distance) correspond to subsequences that
    are most unlike any other subsequence — unusual shapes, rare patterns, and
    structural anomalies.  Each discord of length m contributes an anomaly
    score to every point it covers; we take the maximum over all overlapping
    windows.

    Args:
        m:            subsequence length (default = 24 → one day for hourly data)
        discord_pct:  percentile threshold for flagging a discord
    """
    n = len(r)
    if n < 2 * m:
        return np.zeros(n, dtype=int), np.zeros(n)

    try:
        mp = stumpy.stump(r.astype(np.float64), m=m)
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)

    # mp[:, 0] = matrix profile distances, length n - m + 1
    mp_vals = mp[:, 0].astype(float)

    # Expand each subsequence score back to per-point coverage (max pooling)
    point_scores = np.zeros(n)
    for i, val in enumerate(mp_vals):
        point_scores[i : i + m] = np.maximum(point_scores[i : i + m], val)

    threshold = float(np.percentile(point_scores, discord_pct))
    pred      = (point_scores > threshold).astype(int)
    return pred, point_scores


# ── Step 4: Isolation Forest on engineered features ────────────────────────────

def detect_isolation_forest_features(
    r: np.ndarray,
    window: int = 24,
    contamination: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest on lag + rolling features derived from the STL residual.

    Features: residual · rolling_mean · rolling_std · lag_1 · lag_24.
    No weather columns — purely autoregressive structure.
    """
    n = len(r)
    s = pd.Series(r)
    feats = pd.DataFrame({
        "residual":     r,
        "rolling_mean": s.rolling(window, min_periods=1).mean(),
        "rolling_std":  s.rolling(window, min_periods=1).std().fillna(0),
        "lag_1":        s.shift(1).fillna(0),
        "lag_24":       s.shift(24).fillna(0),
    }).values

    try:
        scaler = StandardScaler()
        X      = scaler.fit_transform(feats)
        clf    = IsolationForest(contamination=contamination, n_estimators=150, random_state=42)
        preds  = clf.fit_predict(X)
        scores = -clf.score_samples(X)
        return (preds == -1).astype(int), scores
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)


# ── Change-point: rolling mean shift ──────────────────────────────────────────

def detect_rolling_mean_shift(
    y: np.ndarray,
    window: int = 24,
    k: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight change-point detector: flags sustained regime shifts.

    Computes the absolute difference between adjacent rolling-window means.
    A large jump signals a level shift (behavioral transition).  Runs on the
    raw series (not residual) so DC-level changes are visible.
    """
    s   = pd.Series(y)
    rm  = s.rolling(window, min_periods=1).mean()
    # Compare mean of current window vs mean of window shifted back by `window`
    shift  = rm.diff(window).abs().fillna(0)
    global_mad = float(np.median(shift[shift > 0])) if (shift > 0).any() else 1e-6
    scores = (shift / global_mad).values
    pred   = (scores > k).astype(int)
    return pred, scores


# ── Step 5: Ensemble strategies ────────────────────────────────────────────────

def build_ensembles(
    method_preds:  Dict[str, np.ndarray],
    method_scores: Dict[str, np.ndarray],
    aurocs:        Dict[str, float],
    gt:            np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Four ensemble strategies over all detector outputs."""
    names          = list(method_preds.keys())
    stacked_preds  = np.stack([method_preds[m]  for m in names], axis=1)
    stacked_scores = np.stack([method_scores[m] for m in names], axis=1)
    vote_counts    = stacked_preds.sum(axis=1)
    M              = len(names)
    score_frac     = vote_counts.astype(float) / M

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # 1. Majority ≥3 votes  (production default)
    results["Ensemble_Majority_3N"] = ((vote_counts >= 3).astype(int), score_frac)

    # 2. Majority ≥2 votes  (higher recall)
    results["Ensemble_Majority_2N"] = ((vote_counts >= 2).astype(int), score_frac)

    # 3. AUROC-weighted vote  (threshold 0.3)
    weights = np.array([max(0.5, aurocs.get(m, 0.5)) - 0.5 for m in names])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(M) / M
    weighted_score = stacked_preds @ weights
    results["Ensemble_Weighted"] = ((weighted_score >= 0.3).astype(int), weighted_score)

    # 4. Matrix-Profile Priority: MP fires + at least 1 other agrees
    mp_pred  = method_preds.get("MatrixProfile", np.zeros(len(gt), dtype=int))
    other_v  = vote_counts - mp_pred
    results["Ensemble_MP_Priority"] = (
        ((mp_pred == 1) & (other_v >= 1)).astype(int),
        mp_pred.astype(float) * 0.6 + score_frac * 0.4,
    )

    return results


# ── Per-building evaluation ────────────────────────────────────────────────────

def evaluate_building(bid: int, df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """Run the 5-step pipeline on the electricity meter of a single building."""
    df  = _add_calendar(df)
    y   = df["meter_reading"].values.astype(float)
    gt  = _gt(df)
    n   = len(y)
    n_gt   = int(gt.sum())
    gt_pct = 100.0 * n_gt / n
    print(f"  n={n}  GT anomalies={n_gt} ({gt_pct:.1f}%)")

    method_rows:     List[dict]            = []
    ensemble_preds:  Dict[str, np.ndarray] = {}
    ensemble_scores: Dict[str, np.ndarray] = {}
    aurocs:          Dict[str, float]      = {}

    def _record(name: str, pred: np.ndarray, score: np.ndarray, t: float) -> dict:
        m = metrics(pred, score, gt)
        aurocs[name]          = m["auroc"] if not np.isnan(m.get("auroc", np.nan)) else 0.5
        ensemble_preds[name]  = pred
        ensemble_scores[name] = score
        print(
            f"    {name:<30s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
            f"  R={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}"
            f"  det={m.get('n_detected',0):4d}  TP={m.get('n_tp',0):4d}"
            f"  FP={m.get('n_fp',0):4d}  FN={m.get('n_fn',0):4d}"
            f"  ({t:.1f}s)"
        )
        return dict(
            building_id=bid, lead_rank=LEAD_RANK[bid], method=name,
            n_series=n, n_gt_anomalies=n_gt, gt_rate_pct=round(gt_pct, 2),
            elapsed_s=round(t, 2), **m,
        )

    # ── Step 1: STL residual ───────────────────────────────────────────────────
    print("\n  [Step 1] STL decomposition → residual")
    t0 = time.perf_counter()
    r  = stl_residual(y, period=24)
    print(f"    residual computed  ({time.perf_counter() - t0:.1f}s)")

    # ── Step 2: Fast detectors on residual ────────────────────────────────────
    print("\n  [Step 2] Fast detectors (residual)")

    t0 = time.perf_counter()
    pred, score = detect_modified_zscore(r)
    method_rows.append(_record("ModifiedZScore", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_rolling_stats(r)
    method_rows.append(_record("RollingStats", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_iqr(r)
    method_rows.append(_record("IQR", pred, score, time.perf_counter() - t0))

    # ── Step 3: Matrix Profile ─────────────────────────────────────────────────
    print("\n  [Step 3] Matrix Profile (discord detection)")

    t0 = time.perf_counter()
    pred, score = detect_matrix_profile(r, m=24)
    method_rows.append(_record("MatrixProfile", pred, score, time.perf_counter() - t0))

    # ── Step 4: Isolation Forest on features ───────────────────────────────────
    print("\n  [Step 4] Isolation Forest (engineered features)")

    t0 = time.perf_counter()
    pred, score = detect_isolation_forest_features(r)
    method_rows.append(_record("IsolationForest_Features", pred, score, time.perf_counter() - t0))

    # Change-point on raw series (separate detector, not part of Steps 2-4)
    print("\n  [Bonus] Rolling mean-shift change-point (raw series)")

    t0 = time.perf_counter()
    pred, score = detect_rolling_mean_shift(y)
    method_rows.append(_record("RollingMeanShift_CP", pred, score, time.perf_counter() - t0))

    # ── Step 5: Ensemble scoring ───────────────────────────────────────────────
    print("\n  [Step 5] Ensemble strategies")
    ens_results = build_ensembles(ensemble_preds, ensemble_scores, aurocs, gt)

    ens_rows: List[dict] = []
    for ens_name, (pred, score) in ens_results.items():
        m   = metrics(pred, score, gt)
        mrk = " ◄ PROD" if ens_name == "Ensemble_Majority_3N" else ""
        print(
            f"    {ens_name:<30s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
            f"  R={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}"
            f"  det={m.get('n_detected',0):4d}  TP={m.get('n_tp',0):4d}"
            f"  FP={m.get('n_fp',0):4d}  FN={m.get('n_fn',0):4d}{mrk}"
        )
        ens_rows.append(dict(
            building_id=bid, lead_rank=LEAD_RANK[bid], ensemble=ens_name,
            n_series=n, n_gt_anomalies=n_gt, gt_rate_pct=round(gt_pct, 2),
            **m,
        ))

    return method_rows, ens_rows


# ── Display ────────────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 20) -> str:
    if np.isnan(v):
        return " " * w
    filled = int(round(v * w))
    return "█" * filled + "░" * (w - filled)


def print_global_method_summary(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("  GLOBAL SUMMARY — Methods (mean F1 across all buildings, electricity meter)")
    print("=" * 90)
    agg = (
        df.groupby("method")[["precision", "recall", "f1", "auroc"]]
        .mean()
        .sort_values("f1", ascending=False)
        .round(3)
    )
    print(f"  {'Method':<30s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s}  F1 bar")
    print("  " + "─" * 83)
    for mname, row in agg.iterrows():
        print(
            f"  {str(mname):<30s}"
            f" {row['precision']:>7.3f} {row['recall']:>7.3f}"
            f" {row['f1']:>7.3f} {row['auroc']:>7.3f}  {_bar(row['f1'])}"
        )


def print_global_ensemble_summary(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("  GLOBAL SUMMARY — Ensemble strategies")
    print("=" * 90)
    agg = (
        df.groupby("ensemble")[["precision", "recall", "f1", "auroc"]]
        .mean()
        .sort_values("f1", ascending=False)
        .round(3)
    )
    print(f"  {'Strategy':<30s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s}  F1 bar")
    print("  " + "─" * 80)
    for ename, row in agg.iterrows():
        mrk = " ◄ PROD" if ename == "Ensemble_Majority_3N" else ""
        print(
            f"  {str(ename):<30s}"
            f" {row['precision']:>7.3f} {row['recall']:>7.3f}"
            f" {row['f1']:>7.3f} {row['auroc']:>7.3f}  {_bar(row['f1'])}{mrk}"
        )


def print_building_method_table(mdf: pd.DataFrame, bid: int):
    sub = mdf[mdf["building_id"] == bid].sort_values("f1", ascending=False)
    print(f"\n  {'Method':<30s} {'F1':>7s} {'P':>7s} {'R':>7s}  {'GT':>5s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  F1 bar")
    print("  " + "─" * 100)
    for _, row in sub.iterrows():
        print(
            f"  {row['method']:<30s}"
            f" {row['f1']:>7.3f} {row['precision']:>7.3f} {row['recall']:>7.3f}"
            f"  {int(row['n_gt_anomalies']):>5d}  {int(row['n_tp']):>5d}"
            f"  {int(row['n_fp']):>5d}  {int(row['n_fn']):>5d}  {_bar(row['f1'])}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LEAD v2 anomaly benchmark (STL-first pipeline)")
    parser.add_argument("--building", type=int, nargs="+", default=TOP5)
    args      = parser.parse_args()
    buildings = [b for b in args.building if b in TOP5]
    if not buildings:
        print("No valid building IDs:", TOP5); sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_method_rows:   List[dict] = []
    all_ensemble_rows: List[dict] = []
    t_wall = time.perf_counter()

    for bid in buildings:
        path = CONSOLIDATED / f"building_{bid}_consolidated.csv"
        if not path.exists():
            print(f"\n[skip] {path} not found"); continue

        print(f"\n{'='*70}")
        print(f"  Building {bid}  (LEAD rank #{LEAD_RANK[bid]})")
        print(f"{'='*70}")

        df      = pd.read_csv(path, parse_dates=["timestamp"])
        elec_df = (
            df[df["meter_type"] == "electricity"]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if elec_df.empty:
            print("  No electricity meter data — skip"); continue

        mrows, erows = evaluate_building(bid, elec_df)
        all_method_rows.extend(mrows)
        all_ensemble_rows.extend(erows)

        bld_mdf = pd.DataFrame(mrows)
        print_building_method_table(bld_mdf, bid)

    if not all_method_rows:
        print("No results."); return

    method_df   = pd.DataFrame(all_method_rows)
    ensemble_df = pd.DataFrame(all_ensemble_rows)

    print_global_method_summary(method_df)
    print_global_ensemble_summary(ensemble_df)

    print("\n  Best ensemble F1 per building:")
    best_ens = (
        ensemble_df
        .loc[ensemble_df.groupby("building_id")["f1"].idxmax()]
        [["building_id", "lead_rank", "ensemble", "f1", "precision", "recall",
          "n_gt_anomalies", "n_tp", "n_fp", "n_fn"]]
        .sort_values("lead_rank")
        .round({"f1": 3, "precision": 3, "recall": 3})
    )
    print(best_ens.to_string(index=False))

    print("\n  Best individual method F1 per building:")
    best_mth = (
        method_df
        .loc[method_df.groupby("building_id")["f1"].idxmax()]
        [["building_id", "lead_rank", "method", "f1", "precision", "recall",
          "n_gt_anomalies", "n_tp", "n_fp", "n_fn"]]
        .sort_values("lead_rank")
        .round({"f1": 3, "precision": 3, "recall": 3})
    )
    print(best_mth.to_string(index=False))

    per_building = (
        method_df
        .groupby(["building_id", "lead_rank", "method"])
        .agg(f1=("f1", "first"), precision=("precision", "first"),
             recall=("recall", "first"), auroc=("auroc", "first"),
             n_gt=("n_gt_anomalies", "first"), n_tp=("n_tp", "first"),
             n_fp=("n_fp", "first"), n_fn=("n_fn", "first"))
        .reset_index()
        .sort_values(["building_id", "f1"], ascending=[True, False])
    )

    global_methods = (
        method_df
        .groupby("method")
        .agg(mean_f1=("f1", "mean"), mean_precision=("precision", "mean"),
             mean_recall=("recall", "mean"), mean_auroc=("auroc", "mean"),
             total_tp=("n_tp", "sum"), total_fp=("n_fp", "sum"),
             total_fn=("n_fn", "sum"))
        .reset_index()
        .sort_values("mean_f1", ascending=False)
        .round(3)
    )

    global_ensembles = (
        ensemble_df
        .groupby("ensemble")
        .agg(mean_f1=("f1", "mean"), mean_precision=("precision", "mean"),
             mean_recall=("recall", "mean"), mean_auroc=("auroc", "mean"),
             total_tp=("n_tp", "sum"), total_fp=("n_fp", "sum"),
             total_fn=("n_fn", "sum"))
        .reset_index()
        .sort_values("mean_f1", ascending=False)
        .round(3)
    )

    paths = {
        "lead_v2_all_methods.csv":      method_df,
        "lead_v2_per_building.csv":     per_building,
        "lead_v2_ensembles.csv":        ensemble_df,
        "lead_v2_global.csv":           global_methods,
        "lead_v2_global_ensembles.csv": global_ensembles,
    }
    for fname, tdf in paths.items():
        p = OUT_DIR / fname
        tdf.to_csv(p, index=False)
        print(f"  Saved → {p}")

    elapsed = time.perf_counter() - t_wall
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
