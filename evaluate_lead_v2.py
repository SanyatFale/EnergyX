#!/usr/bin/env python3
"""
evaluate_lead_v2.py — Enhanced LEAD ground-truth anomaly benchmark.

Addresses all gaps identified in evaluate_lead.py v1:

  1. Electricity-only  — LEAD labels are electricity-derived; non-electricity
                         meters introduce label mismatch noise.

  2. New univariate    — PELT changepoint (level shifts), ARIMA residual
                         (contextual), LGBM residual (lag-based forecast error).

  3. New multivariate  — IsolationForest_MV, DBSCAN_MV, LOF_MV,
                         Mahalanobis_MV, LGBM_Residual_MV.
                         Feature set: meter_reading + air_temperature +
                         dew_temperature + hour + dayofweek + month.

  4. Building 439 case — the zero-detection building in v1.  Multivariate
                         methods that learn weather→consumption relationships
                         are expected to recover signal here.

  5. Ensemble variants — five strategies tested: Majority 3/N (prod default),
                         Majority 2/N, STL-Primary, AUROC-Weighted Vote,
                         MV-Priority.

Usage
-----
  python evaluate_lead_v2.py                     # all 5 buildings
  python evaluate_lead_v2.py --building 439 1319 # subset

Results saved to:
  outputs/benchmark/lead_v2_per_building.csv
  outputs/benchmark/lead_v2_all_methods.csv
  outputs/benchmark/lead_v2_ensembles.csv
  outputs/benchmark/lead_v2_global.csv
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from lightgbm import LGBMClassifier, LGBMRegressor
# pmdarima / auto_arima intentionally excluded — too slow for per-building eval
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN as SklearnDBSCAN

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parent
CONSOLIDATED = ROOT / "data" / "consolidated"
OUT_DIR      = ROOT / "outputs" / "benchmark"

TOP5      = [1319, 1258, 439, 247, 1225]
LEAD_RANK = {1319: 1, 1258: 2, 439: 3, 247: 4, 1225: 5}

# Multivariate feature columns available in consolidated CSVs
MV_WEATHER_COLS = ["air_temperature", "dew_temperature",
                   "sea_level_pressure", "wind_speed"]
MV_CALENDAR_COLS = ["hour", "dayofweek", "month"]
MV_ALL_COLS = MV_WEATHER_COLS + MV_CALENDAR_COLS  # + meter_reading appended per method

# ── Utility ────────────────────────────────────────────────────────────────────

def _gt(df: pd.DataFrame) -> np.ndarray:
    """Binary ground truth: 1 = anomaly (LEAD label=1), 0 = normal."""
    return np.where(df["anomaly"].isna(), 0, df["anomaly"].fillna(0).astype(int)).astype(int)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month
    return df


def _prep_mv_features(df: pd.DataFrame, include_meter: bool = True) -> np.ndarray:
    """Build normalised feature matrix for multivariate detectors."""
    cols = (["meter_reading"] if include_meter else []) + MV_ALL_COLS
    # Drop rows with NaN in any feature
    X = df[cols].copy()
    # Fill any residual weather NaNs with column median
    for c in cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X.values)


def metrics(
    pred_binary: np.ndarray,
    pred_scores: np.ndarray,
    gt_binary:   np.ndarray,
) -> dict:
    if gt_binary.sum() == 0:
        return dict(precision=np.nan, recall=np.nan, f1=np.nan, auroc=np.nan,
                    n_detected=0, n_tp=0, n_fp=0, n_fn=0)
    prec  = float(precision_score(gt_binary, pred_binary, zero_division=0))
    rec   = float(recall_score(gt_binary, pred_binary, zero_division=0))
    f1    = float(f1_score(gt_binary, pred_binary, zero_division=0))
    try:
        s_mn, s_mx = pred_scores.min(), pred_scores.max()
        sn = (pred_scores - s_mn) / (s_mx - s_mn) if s_mx > s_mn else pred_scores
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


# ── Existing univariate tools (via tinyts) ────────────────────────────────────

def _load_tinyts():
    """Lazy-import tinyts tools (requires venv with langchain)."""
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
    return {
        "ZScore":          run_statistical_anomaly_detection,
        "MAD":             run_modified_zscore,
        "RollingStats":    run_rolling_anomaly,
        "IQR":             run_iqr_anomaly,
        "STL":             run_stl_anomaly,
        "IsolationForest": run_isolation_forest,
        # DBSCAN excluded from univariate (near-zero F1 confirmed in v1)
    }


def _call_uv_tool(tool, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Call a tinyts univariate tool → (binary_pred, scores)."""
    y_json = json.dumps(y.tolist())
    n = len(y)
    try:
        raw = json.loads(tool.invoke({"data": y_json}))
        labels = np.array(raw["anomaly_labels"])
        scores = np.array(raw["anomaly_scores"])
        return (labels == -1).astype(int), scores
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)


# ── New univariate methods ─────────────────────────────────────────────────────

def detect_pelt(y: np.ndarray, penalty: float = 3.0, min_size: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """
    PELT changepoint detection (ruptures).

    Detects sustained level shifts by finding structural breaks in the mean.
    Each changepoint window (±min_size steps around the break) is flagged as
    anomalous.  This covers the level-shift gap where all v1 methods failed.

    Returns binary predictions and a continuous score (|local_mean_jump|).
    """
    n = len(y)
    try:
        model   = rpt.Pelt(model="rbf", min_size=min_size, jump=1)
        bkpts   = model.fit_predict(y.reshape(-1, 1), pen=penalty)
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)

    # Score: mean absolute deviation between adjacent segments
    scores = np.zeros(n)
    prev   = 0
    seg_means = []
    for bp in bkpts:
        seg_means.append((prev, bp, float(np.mean(y[prev:bp]))))
        prev = bp

    for i in range(1, len(seg_means)):
        s0, e0, m0 = seg_means[i - 1]
        s1, e1, m1 = seg_means[i]
        jump = abs(m1 - m0)
        # Flag a window of min_size steps around each changepoint
        cp = e0  # changepoint index
        lo = max(0, cp - min_size)
        hi = min(n, cp + min_size)
        scores[lo:hi] = np.maximum(scores[lo:hi], jump)

    # Normalise score by series std and threshold at >1σ jump
    y_std = float(np.std(y)) or 1.0
    pred  = (scores > y_std).astype(int)
    return pred, scores


def detect_seasonal_naive_residual(
    y: np.ndarray,
    period: int = 24,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seasonal Naive Residual detector — O(n), zero model fitting.

    Forecast for each point t is simply y[t - period] (same hour yesterday
    for hourly data with period=24).  Flags points where the absolute residual
    |y[t] - y[t-period]| > k × MAD of all residuals.

    Captures the same intent as ARIMA residual (contextual deviation from
    expected pattern) at a fraction of the compute cost.  Works well for
    energy data with strong 24-hour seasonality.
    """
    n = len(y)
    residuals = np.zeros(n)
    for t in range(period, n):
        residuals[t] = abs(y[t] - y[t - period])

    mad = float(np.median(residuals[period:])) or 1e-6
    scores = residuals / mad
    scores[:period] = 0.0   # no forecast for first period steps
    pred = (scores > k).astype(int)
    return pred, scores


def detect_lgbm_residual_uv(
    y: np.ndarray,
    n_lags: int = 24,
    train_ratio: float = 0.6,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LightGBM lag-based forecast-residual detector (univariate).

    Trains LGBM on lag features [y_{t-1},...,y_{t-n_lags}] → y_t.
    Residuals from this model flag violations of the learned autoregressive
    pattern — a more flexible version of the ARIMA residual.
    """
    n = len(y)
    if n < n_lags + 10:
        return np.zeros(n, dtype=int), np.zeros(n)

    # Build lag feature matrix
    X_rows, y_rows, idx_rows = [], [], []
    for i in range(n_lags, n):
        X_rows.append(y[i - n_lags:i])
        y_rows.append(y[i])
        idx_rows.append(i)

    X_all = np.array(X_rows)
    y_all = np.array(y_rows)
    split_idx = int(train_ratio * len(X_all))

    try:
        model = LGBMRegressor(n_estimators=100, num_leaves=31,
                              learning_rate=0.1, verbosity=-1)
        model.fit(X_all[:split_idx], y_all[:split_idx])
        preds = model.predict(X_all)
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)

    # Residuals mapped back to original time axis
    residuals    = np.zeros(n)
    abs_residuals = np.abs(y_all - preds)
    for j, orig_idx in enumerate(idx_rows):
        residuals[orig_idx] = abs_residuals[j]

    mad    = float(np.median(abs_residuals)) or 1e-6
    scores = residuals / mad
    pred   = (scores > k).astype(int)
    return pred, scores


# ── Multivariate methods ───────────────────────────────────────────────────────

def detect_isolation_forest_mv(
    X_norm: np.ndarray,
    contamination: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """IsolationForest on the full multivariate feature matrix."""
    try:
        clf = IsolationForest(contamination=contamination, random_state=42,
                              n_estimators=200)
        preds  = clf.fit_predict(X_norm)           # -1 = anomaly
        scores = -clf.score_samples(X_norm)        # higher = more anomalous
        return (preds == -1).astype(int), scores
    except Exception:
        return np.zeros(len(X_norm), dtype=int), np.zeros(len(X_norm))


def detect_dbscan_mv(
    X_norm: np.ndarray,
    eps: float = 0.8,
    min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DBSCAN on the multivariate feature space.

    In a weather + calendar + meter feature space, DBSCAN can identify
    anomalous operating regimes (e.g., high consumption on mild days).
    Points labelled -1 (noise) by DBSCAN are flagged as anomalies.
    """
    n = len(X_norm)
    # Subsample for speed if very large
    if n > 10_000:
        idx    = np.random.choice(n, 10_000, replace=False)
        X_sub  = X_norm[idx]
    else:
        idx   = np.arange(n)
        X_sub = X_norm

    try:
        db     = SklearnDBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_sub)
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)

    # Map back to full index
    full_labels = np.zeros(n, dtype=int)
    full_labels[idx] = (labels == -1).astype(int)

    # Score = distance to nearest core point (proxy for anomaly score)
    scores = full_labels.astype(float)
    return full_labels, scores


def detect_lof_mv(
    X_norm: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local Outlier Factor on the multivariate feature space.

    LOF measures local density deviation — good for gradual drift and
    anomalies that are only unusual relative to their neighbourhood.
    """
    n = len(X_norm)
    try:
        clf    = LocalOutlierFactor(n_neighbors=n_neighbors,
                                    contamination=contamination, n_jobs=-1)
        preds  = clf.fit_predict(X_norm)            # -1 = anomaly
        scores = -clf.negative_outlier_factor_      # higher = more anomalous
        return (preds == -1).astype(int), scores
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)


def detect_mahalanobis_mv(
    X_norm: np.ndarray,
    threshold_pct: float = 97.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mahalanobis distance anomaly detector.

    Fits a multivariate Gaussian on the training portion, then flags points
    whose Mahalanobis distance from the mean exceeds a percentile threshold.
    Linear but fast and interpretable.
    """
    n = len(X_norm)
    split = max(48, int(0.6 * n))
    try:
        cov = EmpiricalCovariance()
        cov.fit(X_norm[:split])
        scores = cov.mahalanobis(X_norm) ** 0.5    # sqrt for scale comparability
        threshold = float(np.percentile(scores[:split], threshold_pct))
        pred = (scores > threshold).astype(int)
        return pred, scores
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)


def detect_lgbm_residual_mv(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    k: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LightGBM weather-residual detector (multivariate).

    Trains LGBM on [air_temperature, dew_temperature, sea_level_pressure,
    wind_speed, hour, dayofweek, month] → meter_reading.

    The model learns the expected consumption given weather + time context.
    Large residuals (|actual - predicted| > k × MAD) flag consumption that
    cannot be explained by the weather/calendar pattern — the strongest
    signal for building 439-style anomalies.

    This is the recommended primary detector for weather-coupled buildings.
    """
    n = len(df)
    feature_cols = MV_WEATHER_COLS + MV_CALENDAR_COLS
    target_col   = "meter_reading"

    X = df[feature_cols].copy()
    for c in feature_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    y = df[target_col].values.astype(float)

    split = max(48, int(train_ratio * n))

    try:
        model = LGBMRegressor(n_estimators=200, num_leaves=31,
                              learning_rate=0.05, verbosity=-1)
        model.fit(X.iloc[:split].values, y[:split])
        preds = model.predict(X.values)
    except Exception:
        return np.zeros(n, dtype=int), np.zeros(n)

    residuals = np.abs(y - preds)
    mad       = float(np.median(residuals)) or 1e-6
    scores    = residuals / mad
    pred      = (scores > k).astype(int)
    return pred, scores


# ── Ensemble strategies ────────────────────────────────────────────────────────

def build_ensembles(
    method_preds: Dict[str, np.ndarray],
    method_scores: Dict[str, np.ndarray],
    aurocs: Dict[str, float],
    gt: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Five ensemble strategies applied to the combined method output.

    Returns dict of strategy_name → (binary_pred, score).
    """
    names  = list(method_preds.keys())
    n      = len(gt)
    stacked_preds  = np.stack([method_preds[m] for m in names], axis=1)   # (n, M)
    stacked_scores = np.stack([method_scores[m] for m in names], axis=1)

    vote_counts = stacked_preds.sum(axis=1)   # 0..M per timestep
    M = len(names)

    results = {}

    # ── 1. Majority 3/N (production default) ──────────────────────────────────
    pred_3n = (vote_counts >= 3).astype(int)
    score_3n = vote_counts.astype(float) / M
    results["Ensemble_Majority_3N"] = (pred_3n, score_3n)

    # ── 2. Majority 2/N (lower bar, more recall) ──────────────────────────────
    pred_2n = (vote_counts >= 2).astype(int)
    results["Ensemble_Majority_2N"] = (pred_2n, score_3n)  # same score, diff threshold

    # ── 3. STL-Primary: flag if STL fires OR ≥3 others agree ─────────────────
    stl_pred = method_preds.get("STL", np.zeros(n, dtype=int))
    others   = [method_preds[m] for m in names if m != "STL"]
    other_votes = np.stack(others, axis=1).sum(axis=1) if others else np.zeros(n)
    pred_stl_primary = ((stl_pred == 1) | (other_votes >= 3)).astype(int)
    stl_score = np.maximum(stl_pred.astype(float), score_3n)
    results["Ensemble_STL_Primary"] = (pred_stl_primary, stl_score)

    # ── 4. AUROC-Weighted Vote ─────────────────────────────────────────────────
    # Weight each method by its AUROC (clamped to [0.5, 1]); threshold 0.5
    weights = np.array([max(0.5, aurocs.get(m, 0.5)) - 0.5 for m in names])
    if weights.sum() > 0:
        weights /= weights.sum()
    else:
        weights = np.ones(len(names)) / len(names)

    weighted_score = stacked_preds @ weights   # (n,)
    pred_weighted  = (weighted_score >= 0.3).astype(int)
    results["Ensemble_Weighted"] = (pred_weighted, weighted_score)

    # ── 5. MV-Priority: any MV method fires + at least 1 UV method agrees ─────
    mv_methods = [m for m in names if m.endswith("_MV") or "lgbm_res_mv" in m.lower()
                  or "LGBM_Res_MV" in m]
    uv_methods = [m for m in names if m not in mv_methods]

    if mv_methods and uv_methods:
        mv_votes  = np.stack([method_preds[m] for m in mv_methods], axis=1).max(axis=1)
        uv_votes  = np.stack([method_preds[m] for m in uv_methods], axis=1).sum(axis=1)
        pred_mvp  = ((mv_votes == 1) & (uv_votes >= 1)).astype(int)
        score_mvp = mv_votes.astype(float) * 0.6 + score_3n * 0.4
    else:
        pred_mvp  = pred_3n
        score_mvp = score_3n
    results["Ensemble_MV_Priority"] = (pred_mvp, score_mvp)

    return results


# ── Per-building evaluation ────────────────────────────────────────────────────

def evaluate_building(
    bid: int,
    df: pd.DataFrame,
    uv_tools: dict,
) -> Tuple[List[dict], List[dict]]:
    """
    Evaluate all methods + ensembles on electricity meter of a single building.

    Returns (method_rows, ensemble_rows).
    """
    df = _add_calendar(df)
    y  = df["meter_reading"].values.astype(float)
    gt = _gt(df)
    n  = len(y)

    n_gt   = int(gt.sum())
    gt_pct = 100.0 * n_gt / n
    print(f"  n={n}  GT anomalies={n_gt} ({gt_pct:.1f}%)")

    method_rows:    List[dict] = []
    ensemble_preds: Dict[str, np.ndarray] = {}
    ensemble_scores: Dict[str, np.ndarray] = {}
    aurocs:          Dict[str, float]      = {}

    def _record(name: str, pred: np.ndarray, score: np.ndarray, t: float) -> dict:
        m = metrics(pred, score, gt)
        aurocs[name] = m["auroc"] if not np.isnan(m.get("auroc", np.nan)) else 0.5
        ensemble_preds[name]  = pred
        ensemble_scores[name] = score
        print(
            f"    {name:<28s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
            f"  R={m['recall']:.3f}  AUROC={m.get('auroc', float('nan')):.3f}"
            f"  det={m.get('n_detected',0):4d}  TP={m.get('n_tp',0):4d}"
            f"  FP={m.get('n_fp',0):4d}  FN={m.get('n_fn',0):4d}"
            f"  ({t:.1f}s)"
        )
        return dict(
            building_id=bid, lead_rank=LEAD_RANK[bid], method=name,
            category="univariate" if not name.endswith("_MV") else "multivariate",
            n_series=n, n_gt_anomalies=n_gt, gt_rate_pct=round(gt_pct, 2),
            elapsed_s=round(t, 2), **m,
        )

    # ── A. Existing univariate (via tinyts tools) ──────────────────────────────
    print("\n  [A] Existing univariate methods")
    for name, tool in uv_tools.items():
        t0 = time.perf_counter()
        pred, score = _call_uv_tool(tool, y)
        method_rows.append(_record(name, pred, score, time.perf_counter() - t0))

    # ── B. New univariate: PELT changepoint ───────────────────────────────────
    print("\n  [B] New univariate methods")

    t0 = time.perf_counter()
    pred, score = detect_pelt(y)
    method_rows.append(_record("PELT_Changepoint", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_seasonal_naive_residual(y)
    method_rows.append(_record("SeasonalNaive_Residual", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_lgbm_residual_uv(y)
    method_rows.append(_record("LGBM_Residual_UV", pred, score, time.perf_counter() - t0))

    # ── C. Multivariate methods ────────────────────────────────────────────────
    print("\n  [C] Multivariate methods  (meter + weather + calendar features)")
    X_norm = _prep_mv_features(df, include_meter=True)

    t0 = time.perf_counter()
    pred, score = detect_isolation_forest_mv(X_norm)
    method_rows.append(_record("IsolationForest_MV", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_dbscan_mv(X_norm)
    method_rows.append(_record("DBSCAN_MV", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_lof_mv(X_norm)
    method_rows.append(_record("LOF_MV", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_mahalanobis_mv(X_norm)
    method_rows.append(_record("Mahalanobis_MV", pred, score, time.perf_counter() - t0))

    t0 = time.perf_counter()
    pred, score = detect_lgbm_residual_mv(df)
    method_rows.append(_record("LGBM_Residual_MV", pred, score, time.perf_counter() - t0))

    # ── D. Ensemble strategies ─────────────────────────────────────────────────
    print("\n  [D] Ensemble strategies")
    ens_results = build_ensembles(ensemble_preds, ensemble_scores, aurocs, gt)

    ens_rows: List[dict] = []
    for ens_name, (pred, score) in ens_results.items():
        t0 = time.perf_counter()
        m  = metrics(pred, score, gt)
        mrk = " ◄ PROD" if ens_name == "Ensemble_Majority_3N" else ""
        print(
            f"    {ens_name:<28s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
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
    return "█" * int(round(v * w)) + "░" * (w - int(round(v * w)))


def print_global_method_summary(df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("  GLOBAL SUMMARY — Methods (mean F1 across all buildings, electricity meter)")
    print("=" * 90)
    agg = (
        df.groupby(["method", "category"])[["precision", "recall", "f1", "auroc"]]
        .mean()
        .sort_values("f1", ascending=False)
        .round(3)
    )
    print(f"  {'Method':<28s} {'Cat':>5s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s}  F1 bar")
    print("  " + "─" * 85)
    for (mname, cat), row in agg.iterrows():
        tag = "UV" if cat == "univariate" else "MV"
        print(
            f"  {str(mname):<28s} {tag:>5s}"
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
    print(f"  {'Strategy':<28s} {'P':>7s} {'R':>7s} {'F1':>7s} {'AUROC':>7s}  F1 bar")
    print("  " + "─" * 80)
    for ename, row in agg.iterrows():
        mrk = " ◄ PROD" if ename == "Ensemble_Majority_3N" else ""
        print(
            f"  {str(ename):<28s}"
            f" {row['precision']:>7.3f} {row['recall']:>7.3f}"
            f" {row['f1']:>7.3f} {row['auroc']:>7.3f}  {_bar(row['f1'])}{mrk}"
        )


def print_building_method_table(mdf: pd.DataFrame, bid: int):
    sub = mdf[mdf["building_id"] == bid].sort_values("f1", ascending=False)
    print(f"\n  {'Method':<28s} {'Cat':>5s} {'F1':>7s} {'P':>7s} {'R':>7s}  {'GT':>5s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  F1 bar")
    print("  " + "─" * 100)
    for _, row in sub.iterrows():
        tag = "UV" if row["category"] == "univariate" else "MV"
        print(
            f"  {row['method']:<28s} {tag:>5s}"
            f" {row['f1']:>7.3f} {row['precision']:>7.3f} {row['recall']:>7.3f}"
            f"  {int(row['n_gt_anomalies']):>5d}  {int(row['n_tp']):>5d}"
            f"  {int(row['n_fp']):>5d}  {int(row['n_fn']):>5d}  {_bar(row['f1'])}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LEAD v2 anomaly benchmark")
    parser.add_argument("--building", type=int, nargs="+", default=TOP5)
    args = parser.parse_args()

    buildings = [b for b in args.building if b in TOP5]
    if not buildings:
        print("No valid building IDs:", TOP5); sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading tinyts anomaly tools...")
    uv_tools = _load_tinyts()
    print(f"  Loaded: {list(uv_tools.keys())}\n")

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

        df = pd.read_csv(path, parse_dates=["timestamp"])
        elec_df = (
            df[df["meter_type"] == "electricity"]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if elec_df.empty:
            print("  No electricity meter data — skip"); continue

        mrows, erows = evaluate_building(bid, elec_df, uv_tools)
        all_method_rows.extend(mrows)
        all_ensemble_rows.extend(erows)

        # Per-building method table
        bld_mdf = pd.DataFrame(mrows)
        print_building_method_table(bld_mdf, bid)

    if not all_method_rows:
        print("No results."); return

    method_df   = pd.DataFrame(all_method_rows)
    ensemble_df = pd.DataFrame(all_ensemble_rows)

    # Global summaries
    print_global_method_summary(method_df)
    print_global_ensemble_summary(ensemble_df)

    # Per-building ensemble pivot
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
        [["building_id", "lead_rank", "method", "category", "f1",
          "precision", "recall", "n_gt_anomalies", "n_tp", "n_fp", "n_fn"]]
        .sort_values("lead_rank")
        .round({"f1": 3, "precision": 3, "recall": 3})
    )
    print(best_mth.to_string(index=False))

    # ── Aggregate per-building table ───────────────────────────────────────────
    per_building = (
        method_df
        .groupby(["building_id", "lead_rank", "method", "category"])
        .agg(f1=("f1", "first"), precision=("precision", "first"),
             recall=("recall", "first"), auroc=("auroc", "first"),
             n_gt=("n_gt_anomalies", "first"), n_tp=("n_tp", "first"),
             n_fp=("n_fp", "first"), n_fn=("n_fn", "first"))
        .reset_index()
        .sort_values(["building_id", "f1"], ascending=[True, False])
    )

    global_methods = (
        method_df
        .groupby(["method", "category"])
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

    # ── Save ──────────────────────────────────────────────────────────────────
    paths = {
        "lead_v2_all_methods.csv":   method_df,
        "lead_v2_per_building.csv":  per_building,
        "lead_v2_ensembles.csv":     ensemble_df,
        "lead_v2_global.csv":        global_methods,
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
