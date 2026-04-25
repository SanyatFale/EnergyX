"""
EnergyX — Robustness Evaluation
================================
Runs the same forecasting benchmark as evaluate_energy_benchmark.py across:

  ASHRAE: 10 buildings (diverse use types, sites, sizes)
    building_106, 492, 1195, 1274, 965, 614, 743, 451, 455, 1243

  IDEAL: all 4 variants × 2 homes = 8 datasets
    (variant_5, variant_15, variant_30, variant_60) × (home96, home128)

Reports per-dataset results plus a cross-dataset summary to show
whether model rankings hold across buildings and data variants.

Usage:
    venv/bin/python evaluate_robustness.py
    venv/bin/python evaluate_robustness.py --fast          # fewer eval windows
    venv/bin/python evaluate_robustness.py --ashrae-only
    venv/bin/python evaluate_robustness.py --ideal-only
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse all model/CV/metric logic from evaluate_energy_benchmark
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_energy_benchmark import (
    run_dataset,
    _load_ideal,
    _load_ashrae,
    HORIZON,
    TRAIN_RATIO,
    CV_FOLDS,
    UNI_MODELS,
    MV_MODELS,
    _print_summary,
    _print_quantile_summary,
)

ROOT = Path(__file__).resolve().parent

# ── Dataset definitions ───────────────────────────────────────────────────────

ASHRAE_BUILDINGS = [106, 492, 1195, 1274, 965, 614, 743, 451, 455, 1243]

IDEAL_VARIANTS  = ["variant_5", "variant_15", "variant_30", "variant_60"]
IDEAL_HOMES     = {
    "home96":  ("electric_combined.parquet", ["lag_1h", "lag_168h", "lag_24h"]),
    "home128": ("mains.parquet",             ["lag_1h", "lag_168h", "roll_mean_24h"]),
}

# Resolution-adjusted horizons: all target 24 real hours ahead.
# variant_60 = 60-min steps → 24 steps = 24h
# variant_30 = 30-min steps → 48 steps = 24h
# variant_15 = 15-min steps → 96 steps = 24h
# variant_5  =  5-min steps → 288 steps = 24h
VARIANT_HORIZONS = {
    "variant_5":  288,
    "variant_15":  96,
    "variant_30":  48,
    "variant_60":  24,
}


def build_datasets(ashrae_only: bool, ideal_only: bool) -> dict:
    datasets = {}

    if not ideal_only:
        mb_dir = ROOT / "data/enhanced/ashrae/multi_building"
        for bid in ASHRAE_BUILDINGS:
            csv = mb_dir / f"building_{bid}_elec.csv"
            if not csv.exists():
                print(f"  [WARN] {csv.name} not found — skipping building {bid}")
                continue
            name = f"ASHRAE_b{bid}"
            datasets[name] = dict(
                kind="ashrae_mb",
                path=csv,
                mv_feature_cols=["lag_1", "lag_168", "lag_24"],
            )

    if not ashrae_only:
        for variant in IDEAL_VARIANTS:
            for home, (target_file, mv_cols) in IDEAL_HOMES.items():
                base = ROOT / "data/enhanced/ideal" / variant / home
                if not base.exists():
                    print(f"  [WARN] {base} not found — skipping")
                    continue
                name = f"IDEAL_{variant}_{home}"
                datasets[name] = dict(
                    kind="ideal",
                    home_id=home,
                    variant=variant,
                    target_file=target_file,
                    mv_feature_cols=mv_cols,
                    horizon=VARIANT_HORIZONS[variant],
                )

    return datasets


def load_dataset(cfg: dict):
    if cfg["kind"] == "ashrae_mb":
        df = pd.read_csv(cfg["path"])
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        ts_col = next((c for c in ["timestamp", "date", "ts"] if c in df.columns), None)
        if ts_col:
            df = df.sort_values(ts_col).reset_index(drop=True)
        mv_cols = [c for c in cfg["mv_feature_cols"] if c in df.columns]
        lag_nans = [c for c in mv_cols if df[c].isna().any()]
        if lag_nans:
            df = df.dropna(subset=lag_nans).reset_index(drop=True)
        y = df["meter_reading"].values.astype(float)
        X = df[mv_cols].values.astype(float) if mv_cols else np.zeros((len(y), 1))
        return y, X, mv_cols

    if cfg["kind"] == "ideal":
        # _load_ideal expects variant-scoped path — patch ROOT inside it
        base = ROOT / "data/enhanced/ideal" / cfg["variant"] / cfg["home_id"]
        return _load_ideal_variant(base, cfg["target_file"], cfg["mv_feature_cols"])

    raise ValueError(f"Unknown kind: {cfg['kind']}")


def _load_ideal_variant(base: Path, target_file: str, mv_feature_cols: list):
    """Like _load_ideal but takes an explicit base path (supports any variant)."""
    tgt = pd.read_parquet(base / target_file)
    tgt = tgt[["ts", "value", "lag_24h", "lag_168h"]].rename(columns={"value": "target"})
    tgt["ts"] = pd.to_datetime(tgt["ts"], utc=True)
    tgt = tgt.sort_values("ts").drop_duplicates("ts").set_index("ts")

    cal = pd.read_parquet(base / "calendar.parquet")
    cal["ts"] = pd.to_datetime(cal["ts"], utc=True)
    cal = cal.sort_values("ts").drop_duplicates("ts").set_index("ts")

    weather = None
    wp = base / "weather.parquet"
    if wp.exists():
        weather = pd.read_parquet(wp)
        weather["ts"] = pd.to_datetime(weather["ts"], utc=True)
        weather = weather.sort_values("ts").drop_duplicates("ts").set_index("ts")

    room_temps = {}
    for room in ["kitchen", "livingroom"]:
        tp = base / room / "temperature.parquet"
        if tp.exists():
            df = pd.read_parquet(tp)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")[["value"]]
            room_temps[f"{room}_temp"] = df.rename(columns={"value": f"{room}_temp"})

    merged = tgt.copy()
    for col in ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]:
        if col in cal.columns:
            merged = merged.join(cal[[col]], how="left")
    if weather is not None:
        for col in ["temp", "rhum", "wspd", "pres"]:
            if col in weather.columns:
                merged = merged.join(weather[[col]], how="left")
    for col_name, df in room_temps.items():
        merged = merged.join(df, how="left")

    merged = merged.dropna(subset=["target"]).ffill().bfill()
    merged["lag_1h"]        = merged["target"].shift(1)
    merged["roll_mean_24h"] = merged["target"].shift(1).rolling(24).mean()

    lag_cols_needed = [c for c in mv_feature_cols if c in
                       ["lag_1h", "lag_24h", "lag_168h", "roll_mean_24h"]]
    if lag_cols_needed:
        merged = merged.dropna(subset=lag_cols_needed)

    avail = [c for c in mv_feature_cols if c in merged.columns]
    y  = merged["target"].values.astype(float)
    X  = merged[avail].values.astype(float) if avail else np.zeros((len(y), 1))
    ts = merged.index
    return y, X, avail, ts


# ── Main ─────────────────────────────────────────────────────────────────────

def main(fast: bool, ashrae_only: bool, ideal_only: bool):
    datasets = build_datasets(ashrae_only, ideal_only)
    if not datasets:
        print("No datasets found. Run build_ashrae_multi_building.py first.")
        return

    print(f"  Running {len(datasets)} datasets...\n")

    all_rows, all_timing, all_quant = [], [], []

    for ds_name, cfg in datasets.items():
        print(f"\nLoading {ds_name} ...", end=" ", flush=True)
        try:
            result = load_dataset(cfg)
            if len(result) == 4:
                y, X, feat_names, _ = result
            else:
                y, X, feat_names = result
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        horizon = cfg.get("horizon", None)
        h_label = f"  horizon={horizon} steps ({horizon * int(cfg.get('variant', 'variant_60').split('_')[1]) // 60}h)" if cfg.get("horizon") else ""
        print(f"n={len(y)}  feats={feat_names}{h_label}")

        r, t, q = run_dataset(ds_name, y, X, feat_names, fast=fast, horizon=horizon)
        all_rows.append(r)
        all_timing.append(t)
        all_quant.append(q)

    df         = pd.concat(all_rows,   ignore_index=True) if all_rows   else pd.DataFrame()
    df_timing  = pd.concat(all_timing, ignore_index=True) if all_timing else pd.DataFrame()
    df_quant   = pd.concat(all_quant,  ignore_index=True) if all_quant  else pd.DataFrame()

    out = ROOT / "outputs" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(       out / "robustness_results.csv",  index=False)
    df_timing.to_csv(out / "robustness_timing.csv",   index=False)
    df_quant.to_csv( out / "robustness_quantile.csv", index=False)
    print(f"\nResults saved to {out}/")

    if df.empty:
        return df

    _print_summary(df)
    _print_quantile_summary(df_quant)

    # Cross-dataset consistency: rank each model per dataset, report rank variance
    print(f"\n{'=' * 72}")
    print("  RANK CONSISTENCY — does the winner hold across datasets?")
    print(f"{'=' * 72}")
    rank_rows = []
    for ds, grp in df.groupby("dataset"):
        ranked = grp.sort_values("MAE").reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            rank_rows.append({"dataset": ds, "model": row["model"], "rank": rank})
    rank_df = pd.DataFrame(rank_rows)
    rank_agg = rank_df.groupby("model")["rank"].agg(["mean", "std", "min", "max"])
    rank_agg = rank_agg.sort_values("mean")
    print(f"  {'Model':<26s} {'MeanRank':>9s} {'StdRank':>8s} {'BestRank':>9s} {'WorstRank':>10s}")
    print(f"  {'─' * 68}")
    for name, row in rank_agg.iterrows():
        tag = " ★" if name.startswith("Ens_") else ""
        print(f"  {name:<26s} {row['mean']:9.2f} {row['std']:8.2f} {row['min']:9.0f} {row['max']:10.0f}{tag}")

    # IDEAL: variant effect — does temporal resolution affect model rankings?
    ideal_df = df[df["dataset"].str.startswith("IDEAL_variant")]
    if not ideal_df.empty:
        print(f"\n{'=' * 72}")
        print("  IDEAL VARIANT EFFECT — MAE by resolution (all horizons = 24h real time)")
        print(f"  variant_5=5min(h=288)  variant_15=15min(h=96)  variant_30=30min(h=48)  variant_60=60min(h=24)")
        print(f"{'=' * 72}")
        ideal_df = ideal_df.copy()
        ideal_df["variant"] = ideal_df["dataset"].str.extract(r"(variant_\d+)")
        ideal_df["home"]    = ideal_df["dataset"].str.extract(r"(home\d+)")
        for home, hgrp in ideal_df.groupby("home"):
            print(f"\n  {home}:")
            pivot = hgrp.pivot_table(index="model", columns="variant", values="MAE", aggfunc="mean")
            pivot = pivot.sort_values(pivot.columns[-1])
            print(pivot.to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnergyX Robustness Evaluation")
    parser.add_argument("--fast",        action="store_true", help="Fewer eval windows (quick sanity check)")
    parser.add_argument("--ashrae-only", action="store_true", help="Run only ASHRAE multi-building datasets")
    parser.add_argument("--ideal-only",  action="store_true", help="Run only IDEAL variant datasets")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           EnergyX — Robustness Evaluation                           ║")
    print("║   ASHRAE: 10 buildings  |  IDEAL: 4 variants × 2 homes             ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Horizon:  {HORIZON} steps  |  Split: {int(TRAIN_RATIO*100)}/{100-int(TRAIN_RATIO*100)}  |  CV folds: {CV_FOLDS}")

    t0 = time.time()
    main(fast=args.fast, ashrae_only=args.ashrae_only, ideal_only=args.ideal_only)
    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")
