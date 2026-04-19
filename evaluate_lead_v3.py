#!/usr/bin/env python3
"""
evaluate_lead_v3.py — Supervised anomaly detection using LEAD 1.0 labels.

Three stages
------------
1. STITCH  — Merge new engineered columns from train_features.csv into the
             top-5 consolidated building CSVs and save updated versions.

2. LAG     — Add meter-reading lag features (t-1, t-24, t-168) and rolling
             statistics (24-hour and 168-hour windows) to every dataset used
             for training and evaluation.

3. SUPERVISED MODEL
             — Train LGBM + RF classifiers on the 195 non-top-5 buildings
               (leave-top-5-out) using ALL available features.
             — Evaluate on top-5 buildings: Precision / Recall / F1 / AUROC.
             — Compare against v2 best (LGBM_Residual_MV, F1=0.185).
             — Output per-building anomaly predictions in a row-oriented CSV
               (building_id, timestamp, anomaly_pred, anomaly_prob).

Feature set for supervised model
---------------------------------
  Weather raw  : air_temp, dew_temp, cloud_coverage, sea_level_pressure,
                 wind_direction, wind_speed
  Weather lags : air_temp_mean/max/min/std lag-7d and lag-73h
  Calendar     : hour, weekday, month, cyclic sin/cos encodings, weekday_hour,
                 is_holiday
  GTE features : gte_hour, gte_weekday, gte_month, gte_primary_use,
                 gte_site_id, gte_meter, gte_meter_hour, gte_meter_weekday,
                 gte_meter_month, gte_meter_primary_use, gte_meter_site_id
                 (building-specific GTE excluded to prevent leakage)
  Meter lags   : meter_reading_lag1, lag24, lag168
  Rolling stats: rolling_mean_24h, rolling_std_24h,
                 rolling_mean_168h, rolling_std_168h
  Building     : square_feet, year_built (NaN-filled with median)

Outputs
-------
  data/consolidated/building_{id}_v3.csv         — stitched + lag-enriched CSVs
  outputs/supervised/anomaly_preds_building_{id}.csv  — per-building predictions
  outputs/supervised/anomaly_preds_all.csv       — combined (submission-like)
  outputs/supervised/evaluation_summary.csv      — P/R/F1/AUROC vs v2 baseline
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parent
CONSOLIDATED = ROOT / "data" / "consolidated"
ASHRAE_DIR   = ROOT / "data" / "ashrae-energy-prediction"
OUT_DIR      = ROOT / "outputs" / "supervised"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP5      = [1319, 1258, 439, 247, 1225]
LEAD_RANK = {1319: 1, 1258: 2, 439: 3, 247: 4, 1225: 5}

# ── Feature groups ─────────────────────────────────────────────────────────────

WEATHER_RAW = [
    "air_temperature", "dew_temperature", "cloud_coverage",
    "sea_level_pressure", "wind_direction", "wind_speed",
]
WEATHER_LAGS = [
    "air_temperature_mean_lag7", "air_temperature_max_lag7",
    "air_temperature_min_lag7", "air_temperature_std_lag7",
    "air_temperature_mean_lag73", "air_temperature_max_lag73",
    "air_temperature_min_lag73", "air_temperature_std_lag73",
]
CALENDAR = [
    # purely numeric — compound string cols (weekday_hour, building_*) excluded
    "hour", "weekday", "month",
    "hour_x", "hour_y", "month_x", "month_y", "weekday_x", "weekday_y",
    "is_holiday",
]
# GTE features — exclude building-specific ones (leakage in leave-building-out)
# Also exclude gte_building_id / gte_meter_building_id* (building-level leakage)
GTE = [
    "gte_hour", "gte_weekday", "gte_month",
    "gte_primary_use", "gte_site_id", "gte_meter",
    "gte_meter_hour", "gte_meter_weekday", "gte_meter_month",
    "gte_meter_primary_use", "gte_meter_site_id",
]
BUILDING_META = ["square_feet", "year_built"]
METER_LAGS = [
    "meter_reading_lag1", "meter_reading_lag24", "meter_reading_lag168",
]
ROLLING = [
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
]

ALL_FEATURES = (
    WEATHER_RAW + WEATHER_LAGS + CALENDAR + GTE + BUILDING_META
    + METER_LAGS + ROLLING
)


# ── Stage 1: Load and stitch train_features ────────────────────────────────────

NEW_COLS_FROM_TF = (
    WEATHER_LAGS + CALENDAR + GTE
    + ["precip_depth_1_hr"]    # may already exist but ensure present
)

def load_train_features() -> pd.DataFrame:
    print("Loading train_features.csv (all 200 buildings)...")
    tf = pd.read_csv(
        ASHRAE_DIR / "train_features.csv",
        parse_dates=["timestamp"],
        low_memory=False,
    )
    # Normalise anomaly column: NaN → 0
    tf["anomaly"] = tf["anomaly"].fillna(0).astype(int)
    print(f"  {len(tf):,} rows | {tf['building_id'].nunique()} buildings | "
          f"anomaly rate={tf['anomaly'].mean():.4f}")
    return tf


# ── Stage 2: Add lag + rolling features ───────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute meter-reading lags and rolling statistics IN-PLACE (sorted by ts).
    Works on any single-building, single-meter dataframe.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    y  = df["meter_reading"].values.astype(float)
    n  = len(y)

    # Lags
    lag1   = np.full(n, np.nan); lag1[1:]   = y[:-1]
    lag24  = np.full(n, np.nan); lag24[24:] = y[:-24]  if n > 24  else lag24
    lag168 = np.full(n, np.nan)
    if n > 168:
        lag168[168:] = y[:-168]

    df["meter_reading_lag1"]   = lag1
    df["meter_reading_lag24"]  = lag24
    df["meter_reading_lag168"] = lag168

    # Rolling (exclusive of current point)
    s = pd.Series(y)
    df["rolling_mean_24h"]  = s.shift(1).rolling(24,  min_periods=1).mean().values
    df["rolling_std_24h"]   = s.shift(1).rolling(24,  min_periods=1).std().values
    df["rolling_mean_168h"] = s.shift(1).rolling(168, min_periods=1).mean().values
    df["rolling_std_168h"]  = s.shift(1).rolling(168, min_periods=1).std().values

    return df


def stitch_and_update_consolidated(tf: pd.DataFrame):
    """
    For each top-5 building, merge new engineered columns from train_features
    into the consolidated CSV (electricity meter only for stitching), add lag
    features, and save as building_{id}_v3.csv.
    """
    tf_cols_to_add = [c for c in NEW_COLS_FROM_TF if c in tf.columns]

    for bid in TOP5:
        src = CONSOLIDATED / f"building_{bid}_consolidated.csv"
        dst = CONSOLIDATED / f"building_{bid}_v3.csv"
        if not src.exists():
            print(f"  [skip] {src} not found")
            continue

        df = pd.read_csv(src, parse_dates=["timestamp"])

        # Merge new engineered columns (train_features is building+timestamp level)
        tf_bid = tf[tf["building_id"] == bid][["timestamp"] + tf_cols_to_add].copy()
        tf_bid = tf_bid.drop_duplicates("timestamp")
        df = df.merge(tf_bid, on="timestamp", how="left")

        # Add lag features per meter type
        enriched = []
        for mtype, grp in df.groupby("meter_type"):
            enriched.append(add_lag_features(grp))
        df = pd.concat(enriched).sort_values(["meter", "timestamp"]).reset_index(drop=True)

        df.to_csv(dst, index=False)
        print(f"  Saved {dst.name}  ({len(df):,} rows, {len(df.columns)} cols)")


# ── Stage 3: Build training dataset from 195 buildings ────────────────────────

def build_training_set(tf: pd.DataFrame) -> pd.DataFrame:
    """
    From train_features (200 buildings), keep the 195 non-top-5 buildings,
    add lag + rolling features, and return the feature matrix + labels.
    """
    train_tf = tf[~tf["building_id"].isin(TOP5)].copy()
    print(f"  Training buildings: {train_tf['building_id'].nunique()} "
          f"| rows: {len(train_tf):,} "
          f"| anomaly rate: {train_tf['anomaly'].mean():.4f}")

    # Add lag + rolling per building
    print("  Adding lag features to training set (195 buildings)...")
    parts = []
    for bid, grp in train_tf.groupby("building_id"):
        parts.append(add_lag_features(grp))
    train_df = pd.concat(parts).reset_index(drop=True)
    return train_df


def build_test_set(tf: pd.DataFrame) -> pd.DataFrame:
    """Top-5 buildings from train_features, with lag features added."""
    test_tf = tf[tf["building_id"].isin(TOP5)].copy()
    parts = []
    for bid, grp in test_tf.groupby("building_id"):
        parts.append(add_lag_features(grp))
    return pd.concat(parts).reset_index(drop=True)


def prepare_X_y(
    df: pd.DataFrame,
    features: List[str],
    fill_meta_median: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extract feature matrix X and label vector y.
    Returns (X, y, median_dict) — median_dict used to fill NaN in test set.
    """
    X = df[features].copy()

    # Fill building meta NaN with median (computed from training if provided)
    meds = fill_meta_median or {}
    for col in BUILDING_META:
        if col in X.columns:
            med = meds.get(col, float(X[col].median()))
            X[col] = X[col].fillna(med)
            meds[col] = med

    # Fill remaining NaN with 0 (lags at series start, missing weather)
    X = X.fillna(0)
    y = df["anomaly"].values.astype(int)
    return X.values, y, meds


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    prec  = float(precision_score(y_true, y_pred, zero_division=0))
    rec   = float(recall_score(y_true, y_pred, zero_division=0))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = np.nan
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return dict(precision=prec, recall=rec, f1=f1, auroc=auroc,
                n_tp=tp, n_fp=fp, n_fn=fn,
                n_detected=int(y_pred.sum()), n_gt=int(y_true.sum()))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_wall = time.perf_counter()

    # ── Stage 1 & 2: Stitch + lag features into consolidated CSVs ─────────────
    print("\n" + "="*70)
    print("STAGE 1: Stitch train_features + lag features → building_*_v3.csv")
    print("="*70)
    tf = load_train_features()
    stitch_and_update_consolidated(tf)

    # ── Stage 3: Supervised training ──────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 2: Build training set (195 buildings, leave-top-5-out)")
    print("="*70)
    train_df = build_training_set(tf)
    test_df  = build_test_set(tf)

    # Determine which feature columns actually exist in the data
    avail_features = [f for f in ALL_FEATURES if f in train_df.columns]
    missing = [f for f in ALL_FEATURES if f not in train_df.columns]
    if missing:
        print(f"  [info] Features not in train_features (skipped): {missing}")
    print(f"  Using {len(avail_features)} features")

    print("\n  Preparing feature matrices...")
    X_train, y_train, meta_meds = prepare_X_y(train_df, avail_features)
    X_test_all, y_test_all, _   = prepare_X_y(test_df, avail_features,
                                               fill_meta_median=meta_meds)

    pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  X_train: {X_train.shape}  positives: {y_train.sum():,} "
          f"({100*y_train.mean():.2f}%)  scale_pos_weight≈{pos_weight}")
    print(f"  X_test : {X_test_all.shape}  positives: {y_test_all.sum():,}")

    # ── Train LGBM ────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 3a: Train LGBMClassifier")
    print("="*70)
    lgbm = LGBMClassifier(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    )
    t0 = time.perf_counter()
    lgbm.fit(X_train, y_train)
    print(f"  LGBM trained in {time.perf_counter()-t0:.1f}s")

    # ── Train RF ─────────────────────────────────────────────────────────────
    print("\nSTAGE 3b: Train RandomForestClassifier")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    print(f"  RF trained in {time.perf_counter()-t0:.1f}s")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature": avail_features,
        "lgbm_importance": lgbm.feature_importances_,
        "rf_importance":   rf.feature_importances_,
    }).sort_values("lgbm_importance", ascending=False)
    print("\n  Top 20 features (LGBM importance):")
    print(fi.head(20).to_string(index=False))

    # ── Evaluate per building ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 4: Per-building evaluation (top 5, electricity meter)")
    print("="*70)

    # v2 baseline for comparison (from evaluate_lead_v2 results)
    V2_BASELINE = {1319: 0.262, 1258: 0.208, 439: 0.147, 247: 0.167, 1225: 0.144}

    eval_rows = []
    all_pred_rows = []

    for bid in TOP5:
        bld_df = test_df[test_df["building_id"] == bid].copy()
        X_bld, y_bld, _ = prepare_X_y(bld_df, avail_features,
                                       fill_meta_median=meta_meds)

        print(f"\n  Building {bid} (LEAD rank #{LEAD_RANK[bid]}) "
              f"| n={len(y_bld):,} | GT={y_bld.sum()}")

        for model_name, model in [("LGBM", lgbm), ("RF", rf)]:
            prob  = model.predict_proba(X_bld)[:, 1]
            # Fixed threshold at 0.5 — contamination-rate percentile causes
            # n_detected==n_gt always, collapsing P==R==F1 artificially.
            thresh  = 0.5
            pred    = (prob >= thresh).astype(int)
            m = compute_metrics(y_bld, pred, prob)

            v2_f1 = V2_BASELINE.get(bid, 0.0)
            delta = m["f1"] - v2_f1
            sign  = "+" if delta >= 0 else ""
            print(
                f"    {model_name:<6s}  F1={m['f1']:.3f} ({sign}{delta:.3f} vs v2)"
                f"  P={m['precision']:.3f}  R={m['recall']:.3f}"
                f"  AUROC={m['auroc']:.3f}"
                f"  TP={m['n_tp']:4d}  FP={m['n_fp']:4d}  FN={m['n_fn']:4d}"
                f"  thresh={thresh:.3f}"
            )

            eval_rows.append(dict(
                building_id=bid, lead_rank=LEAD_RANK[bid],
                model=model_name,
                threshold=round(thresh, 4),
                v2_lgbm_residual_mv_f1=v2_f1,
                delta_f1=round(delta, 4),
                **m,
            ))

            # Save predictions
            pred_df = bld_df[["building_id", "timestamp", "anomaly"]].copy()
            pred_df["anomaly_pred"] = pred
            pred_df[f"anomaly_prob_{model_name.lower()}"] = prob
            all_pred_rows.append(pred_df.assign(model=model_name))

    # ── Global summary ────────────────────────────────────────────────────────
    eval_df = pd.DataFrame(eval_rows)
    print("\n" + "="*70)
    print("GLOBAL SUMMARY — Mean across top-5 buildings")
    print("="*70)
    g = eval_df.groupby("model")[
        ["f1", "precision", "recall", "auroc", "delta_f1"]
    ].mean().round(3)
    print(g.to_string())

    print("\n  Per-building F1 comparison (v2 LGBM_Residual_MV vs v3):")
    pivot = eval_df.pivot_table(
        index="building_id", columns="model", values="f1"
    ).round(3)
    pivot["v2_baseline"] = eval_df.groupby("building_id")["v2_lgbm_residual_mv_f1"].first()
    print(pivot.to_string())

    # ── Save outputs ─────────────────────────────────────────────────────────
    eval_df.to_csv(OUT_DIR / "evaluation_summary.csv", index=False)

    # Combined predictions (submission-like)
    all_preds = pd.concat(all_pred_rows, ignore_index=True)
    # One row per building×timestamp with both model probs
    lgbm_preds = all_preds[all_preds["model"] == "LGBM"][
        ["building_id", "timestamp", "anomaly",
         "anomaly_pred", "anomaly_prob_lgbm"]
    ]
    rf_preds = all_preds[all_preds["model"] == "RF"][
        ["building_id", "timestamp", "anomaly_prob_rf", "anomaly_pred"]
    ].rename(columns={"anomaly_pred": "anomaly_pred_rf"})

    combined = lgbm_preds.merge(
        rf_preds.drop(columns=[]), on=None, how="left",
        left_index=True, right_index=True
    )
    # Simpler: just save per-model
    lgbm_out = all_preds[all_preds["model"] == "LGBM"].drop(columns=["model"])
    rf_out   = all_preds[all_preds["model"] == "RF"].drop(columns=["model"])

    lgbm_out.to_csv(OUT_DIR / "anomaly_preds_lgbm.csv", index=False)
    rf_out.to_csv(OUT_DIR / "anomaly_preds_rf.csv", index=False)

    # Per-building
    for bid in TOP5:
        sub = all_preds[all_preds["building_id"] == bid]
        sub.to_csv(OUT_DIR / f"anomaly_preds_building_{bid}.csv", index=False)

    # Feature importance
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # ── Stage 5: Within-building temporal 80/20 training ──────────────────────
    run_within_building_stage(avail_features, eval_df)

    elapsed = time.perf_counter() - t_wall
    print(f"\nSaved to {OUT_DIR}/")
    print(f"Total runtime: {elapsed:.1f}s")


# ── Stage 5: Within-building temporal 80/20 training ──────────────────────────

def run_within_building_stage(
    avail_features: List[str],
    lbo_eval_df: pd.DataFrame,
) -> None:
    """
    Train LGBM + RF independently on each top-5 building using the first 80%
    of its own timestamps, then evaluate on the last 20%.

    Split design
    ------------
    Temporal (chronological), NOT stratified.
    • "Building A trains on its historical data to predict future anomalies"
      requires preserving time order so lag features remain causal.
    • Split point = row at index floor(n * 0.8), sorted by timestamp.
      For 8784 hours in 2016 this lands around 2016-10-16, giving
      ~7027 train / ~1757 test rows per building.
    • Lag features in _v3.csv were computed over the full 2016 series, but
      they only look backward, so no future information leaks into the
      training window or across the split boundary.
    • Building-specific GTE columns (gte_building_id, gte_meter_building_id)
      are included here — there is no cross-building leakage concern in a
      within-building model.

    Comparison output
    -----------------
    Saves within_building_evaluation.csv with columns matching
    evaluation_summary.csv, plus split_timestamp / n_train / n_test /
    gt_train / gt_test.  A printed table compares AUROC across both stages.
    """
    print("\n" + "="*70)
    print("STAGE 5: Within-building temporal 80/20 train/test split")
    print("="*70)
    print("  Each building trains on its OWN first 80% of 2016 timestamps")
    print("  and is evaluated on its own last 20%.")
    print("  Threshold fixed at 0.5 (same as Stage 3)")

    # Build lbo AUROC lookup for the final comparison table
    lbo_auroc = {}
    if lbo_eval_df is not None and len(lbo_eval_df):
        for _, row in lbo_eval_df.iterrows():
            lbo_auroc[(int(row["building_id"]), row["model"])] = row["auroc"]

    # Building-specific GTE columns excluded from Stage 3 (cross-building leakage)
    # but valid for within-building training.  Add them if present in the file.
    BUILDING_GTE_EXTRAS = [
        "gte_building_id",
        "gte_meter_building_id",
    ]

    V2_BASELINE = {1319: 0.262, 1258: 0.208, 439: 0.147, 247: 0.167, 1225: 0.144}

    wb_rows = []

    for bid in TOP5:
        bld_path = CONSOLIDATED / f"building_{bid}_v3.csv"
        if not bld_path.exists():
            print(f"  [skip] {bld_path} not found — run Stages 1+2 first")
            continue

        df = pd.read_csv(bld_path, parse_dates=["timestamp"], low_memory=False)

        # Electricity meter only (consistent with other stages)
        df = df[df["meter"] == 0].copy()
        df["anomaly"] = df["anomaly"].fillna(0).astype(int)
        df = df.sort_values("timestamp").reset_index(drop=True)

        n = len(df)
        split_idx = int(n * 0.8)
        split_ts  = df.iloc[split_idx]["timestamp"]

        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()

        gt_train = int(train_df["anomaly"].sum())
        gt_test  = int(test_df["anomaly"].sum())

        print(f"\n  Building {bid} (LEAD rank #{LEAD_RANK[bid]})")
        print(f"    Total rows : {n:,}  |  split @ {split_ts.date()}")
        print(f"    Train rows : {len(train_df):,}  |  anomalies: {gt_train} "
              f"({100*train_df['anomaly'].mean():.2f}%)")
        print(f"    Test  rows : {len(test_df):,}  |  anomalies: {gt_test} "
              f"({100*test_df['anomaly'].mean():.2f}%)")

        if gt_train == 0:
            print(f"    [skip] No anomalies in training split — cannot train")
            continue
        if gt_test == 0:
            print(f"    [skip] No anomalies in test split — metrics undefined")
            continue

        # Feature set = avail_features (cross-building safe set) +
        # building-specific GTE columns if present in this file
        extras = [c for c in BUILDING_GTE_EXTRAS if c in df.columns]
        bld_features = [f for f in avail_features if f in df.columns] + extras
        if extras:
            print(f"    Using {len(bld_features)} features "
                  f"(+{len(extras)} building-specific GTE: {extras})")
        else:
            print(f"    Using {len(bld_features)} features")

        # Compute training-split medians for meta columns (no test leakage)
        meta_meds = {}
        for col in BUILDING_META:
            if col in train_df.columns:
                meta_meds[col] = float(train_df[col].median())

        def _prep(frame: pd.DataFrame):
            X = frame[bld_features].copy()
            for col in BUILDING_META:
                if col in X.columns:
                    X[col] = X[col].fillna(meta_meds.get(col, 0))
            X = X.fillna(0)
            y = frame["anomaly"].values.astype(int)
            return X.values, y

        X_tr, y_tr = _prep(train_df)
        X_te, y_te = _prep(test_df)

        pos_weight = int((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        print(f"    scale_pos_weight={pos_weight}")

        # Train LGBM (lighter than Stage 3 — only ~7k rows per building)
        lgbm_wb = LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            n_jobs=-1,
            verbosity=-1,
            random_state=42,
        )
        t0 = time.perf_counter()
        lgbm_wb.fit(X_tr, y_tr)
        print(f"    LGBM trained in {time.perf_counter()-t0:.1f}s")

        # Train RF
        rf_wb = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        t0 = time.perf_counter()
        rf_wb.fit(X_tr, y_tr)
        print(f"    RF    trained in {time.perf_counter()-t0:.1f}s")

        # Evaluate both models
        for model_name, model in [("LGBM", lgbm_wb), ("RF", rf_wb)]:
            prob  = model.predict_proba(X_te)[:, 1]
            thresh = 0.5
            pred  = (prob >= thresh).astype(int)
            m = compute_metrics(y_te, pred, prob)

            lbo_a = lbo_auroc.get((bid, model_name), float("nan"))
            delta_auroc = m["auroc"] - lbo_a if not np.isnan(lbo_a) else float("nan")
            sign = "+" if delta_auroc >= 0 else ""

            v2_f1   = V2_BASELINE.get(bid, 0.0)
            delta_f1 = m["f1"] - v2_f1

            print(
                f"    {model_name:<6s}  F1={m['f1']:.3f} ({'+' if delta_f1>=0 else ''}{delta_f1:.3f} vs v2)"
                f"  P={m['precision']:.3f}  R={m['recall']:.3f}"
                f"  AUROC={m['auroc']:.3f} ({sign}{delta_auroc:.3f} vs LBO)"
                f"  TP={m['n_tp']:3d}  FP={m['n_fp']:3d}  FN={m['n_fn']:3d}"
            )

            wb_rows.append(dict(
                building_id=bid,
                lead_rank=LEAD_RANK[bid],
                model=model_name,
                split_timestamp=str(split_ts.date()),
                n_train=len(train_df),
                n_test=len(test_df),
                gt_train=gt_train,
                gt_test=gt_test,
                threshold=thresh,
                v2_lgbm_residual_mv_f1=v2_f1,
                delta_f1_vs_v2=round(delta_f1, 4),
                lbo_auroc=round(lbo_a, 4) if not np.isnan(lbo_a) else None,
                delta_auroc_vs_lbo=round(delta_auroc, 4) if not np.isnan(delta_auroc) else None,
                **m,
            ))

            # Save per-building test predictions
            out = test_df[["building_id", "timestamp", "anomaly"]].copy()
            out["anomaly_pred"] = pred
            out[f"anomaly_prob_{model_name.lower()}"] = prob
            out["split"] = "within_building_test_20pct"
            out.to_csv(
                OUT_DIR / f"wb_preds_building_{bid}_{model_name.lower()}.csv",
                index=False,
            )

    if not wb_rows:
        print("  No within-building results — check that _v3.csv files exist")
        return

    wb_df = pd.DataFrame(wb_rows)

    # ── Print comparison summary ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 5 SUMMARY — Within-building 80/20 vs Leave-building-out (LBO)")
    print("="*70)

    print("\n  Mean across all buildings:")
    g = wb_df.groupby("model")[
        ["f1", "precision", "recall", "auroc", "delta_auroc_vs_lbo"]
    ].mean().round(3)
    print(g.to_string())

    print("\n  Per-building AUROC  [WB = within-building | LBO = leave-building-out]:")
    header = f"  {'Bldg':>6}  {'Model':<6}  {'WB_AUROC':>8}  {'LBO_AUROC':>9}  {'Delta':>6}  {'WB_F1':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, row in wb_df.sort_values(["building_id", "model"]).iterrows():
        lbo_str = f"{row['lbo_auroc']:.3f}" if row["lbo_auroc"] else "   N/A"
        d = row["delta_auroc_vs_lbo"]
        d_str = f"{'+' if d >= 0 else ''}{d:.3f}" if d is not None else "   N/A"
        print(
            f"  {int(row['building_id']):>6}  {row['model']:<6}  "
            f"{row['auroc']:>8.3f}  {lbo_str:>9}  {d_str:>6}  {row['f1']:>6.3f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    wb_df.to_csv(OUT_DIR / "within_building_evaluation.csv", index=False)
    print(f"\n  Saved → outputs/supervised/within_building_evaluation.csv")
    print(f"  Saved → outputs/supervised/wb_preds_building_{{id}}_{{model}}.csv (×{len(TOP5)*2})")


if __name__ == "__main__":
    main()
