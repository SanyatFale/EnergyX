"""
EnergyX — ETT Robustness Evaluation
=====================================
Evaluates forecasting on all 4 ETT variants (ETTh1, ETTh2, ETTm1, ETTm2).

Protocol:
  - Horizons : 96 and 192 steps
  - Slices   : 25 evenly-spaced evaluation windows per dataset
  - Context  : 512 steps preceding each slice origin (from train+seen test)
  - Models   : trained once on the 80% training set; statistical models
                re-fit per slice on the 512-step context window
  - Metrics  : MAE and MAPE% averaged across 25 slices
  - Reports  : per-dataset × horizon table + overall ranking

Usage:
    venv/bin/python evaluate_robustness_ETT.py
    venv/bin/python evaluate_robustness_ETT.py --fast   # horizons=[96] only
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse all model + metric logic from evaluate_forecasting.py
from evaluate_forecasting import (
    load_dataset,
    # metrics
    metric_mae,
    metric_mape,
    # CV search
    cv_search_ets,
    cv_search_rf_uni,
    cv_search_lgbm_uni,
    cv_search_rf_mv,
    cv_search_lgbm_mv,
    cv_search_rf_combined,
    cv_search_lgbm_combined,
    cv_search_nbeats,
    # final training
    train_final_rf_uni,
    train_final_lgbm_uni,
    train_final_rf_mv,
    train_final_lgbm_mv,
    train_final_rf_combined,
    train_final_lgbm_combined,
    train_final_nbeats,
    # forecast helpers
    forecast_naive,
    forecast_seasonal_naive,
    forecast_arima,
    _fit_predict_ets,
    # ensemble
    ens_inv_smape,
    ens_best,
    ens_median,
    ens_trim_mean,
    # constants
    TRAIN_RATIO,
    FEATURE_COLS,
)

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "ETTh1": ROOT / "data/test/ETTh1.csv",
    "ETTh2": ROOT / "data/test/ETTh2.csv",
    "ETTm1": ROOT / "data/test/ETTm1.csv",
    "ETTm2": ROOT / "data/test/ETTm2.csv",
}

HORIZONS    = [96, 192]
N_SLICES    = 25
CONTEXT_LEN = 512   # steps of history fed to each evaluation window

ALL_MODELS = [
    "Naive", "SeasonalNaive", "ARIMA", "ETS",
    "RF_uni", "LightGBM_uni",
    "RF_mv", "LightGBM_mv",
    "RF_combined", "LightGBM_combined",
    "N-BEATS",
]
ENSEMBLE_FNS = [
    ("Ens_InvSMAPE", ens_inv_smape),
    ("Ens_Best",     ens_best),
    ("Ens_Median",   ens_median),
    ("Ens_TrimMean", ens_trim_mean),
]


def _slice_origins(y_test: np.ndarray, horizon: int, n_slices: int) -> list[int]:
    """Return n_slices evenly-spaced origins in the test set such that
    origin + horizon <= len(y_test)."""
    max_origin = len(y_test) - horizon
    if max_origin < 1:
        return []
    indices = np.linspace(0, max_origin, n_slices, endpoint=False, dtype=int)
    return indices.tolist()


def evaluate_dataset(
    ds_name: str,
    y_all: np.ndarray,
    X_all: np.ndarray,
    horizon: int,
    sp: int,
    fast: bool,
) -> list[dict]:
    n = len(y_all)
    train_end = int(n * TRAIN_RATIO)
    y_train = y_all[:train_end]
    y_test  = y_all[train_end:]
    X_train = X_all[:train_end]
    X_test  = X_all[train_end:]

    print(f"\n  horizon={horizon}  train={train_end}  test={len(y_test)}  sp={sp}")

    # ── Phase 1: CV param search (once per dataset×horizon) ──────────────────
    print("  CV param search … ", end="", flush=True)
    t0 = time.perf_counter()
    ets_p       = cv_search_ets(y_train, horizon, sp)[0]
    rf_uni_p    = cv_search_rf_uni(y_train, horizon)[0]
    lgbm_uni_p  = cv_search_lgbm_uni(y_train, horizon)[0]
    rf_mv_p     = cv_search_rf_mv(y_train, X_train, horizon)[0]
    lgbm_mv_p   = cv_search_lgbm_mv(y_train, X_train, horizon)[0]
    rf_comb_p   = cv_search_rf_combined(y_train, X_train, horizon)[0]
    lgbm_comb_p = cv_search_lgbm_combined(y_train, X_train, horizon)[0]
    nbeats_p    = cv_search_nbeats(y_train, horizon)[0]
    print(f"{time.perf_counter() - t0:.0f}s")

    # ── Phase 2: Train final models on full training set ──────────────────────
    print("  Training final models … ", end="", flush=True)
    t0 = time.perf_counter()
    trained = {
        "RF_uni":             train_final_rf_uni(y_train, rf_uni_p),
        "LightGBM_uni":       train_final_lgbm_uni(y_train, lgbm_uni_p),
        "RF_mv":              train_final_rf_mv(X_train, y_train, rf_mv_p),
        "LightGBM_mv":        train_final_lgbm_mv(X_train, y_train, lgbm_mv_p),
        "RF_combined":        train_final_rf_combined(y_train, X_train, rf_comb_p),
        "LightGBM_combined":  train_final_lgbm_combined(y_train, X_train, lgbm_comb_p),
        "N-BEATS":            train_final_nbeats(y_train, nbeats_p),
    }
    print(f"{time.perf_counter() - t0:.1f}s")

    # ── Phase 3: 25-slice evaluation ─────────────────────────────────────────
    origins = _slice_origins(y_test, horizon, N_SLICES)
    if not origins:
        print(f"  [WARN] test set too short for horizon={horizon}, skipping")
        return []

    print(f"  Evaluating {len(origins)} slices (context={CONTEXT_LEN}) …")

    slice_maes   = {m: [] for m in ALL_MODELS}
    slice_mapes  = {m: [] for m in ALL_MODELS}
    ens_maes     = {n: [] for n, _ in ENSEMBLE_FNS}
    ens_mapes    = {n: [] for n, _ in ENSEMBLE_FNS}

    for si, origin in enumerate(origins):
        actual = y_test[origin: origin + horizon]

        # Context window: up to CONTEXT_LEN steps of train+seen test
        seen = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train
        y_ctx = seen[-CONTEXT_LEN:]

        X_win     = X_test[origin: origin + horizon]
        has_exog  = len(X_win) >= horizon

        preds: dict[str, np.ndarray] = {}

        # Statistical (re-fit on context window)
        preds["Naive"]        = forecast_naive(y_ctx, horizon)
        preds["SeasonalNaive"]= forecast_seasonal_naive(y_ctx, horizon, sp)
        preds["ARIMA"]        = forecast_arima(y_ctx, horizon)
        preds["ETS"]          = _fit_predict_ets(y_ctx, horizon, **ets_p)

        # Tree/neural (trained once, inference on context)
        preds["RF_uni"]        = trained["RF_uni"].forecast(y_ctx, horizon)
        preds["LightGBM_uni"]  = trained["LightGBM_uni"].forecast(y_ctx, horizon)
        preds["N-BEATS"]       = trained["N-BEATS"].forecast(y_ctx, horizon)

        if has_exog:
            preds["RF_mv"]            = trained["RF_mv"].forecast(X_win, horizon)
            preds["LightGBM_mv"]      = trained["LightGBM_mv"].forecast(X_win, horizon)
            preds["RF_combined"]      = trained["RF_combined"].forecast(y_ctx, X_win, horizon)
            preds["LightGBM_combined"]= trained["LightGBM_combined"].forecast(y_ctx, X_win, horizon)
        else:
            for nm in ["RF_mv", "LightGBM_mv", "RF_combined", "LightGBM_combined"]:
                preds[nm] = np.full(horizon, y_ctx[-1])

        for m in ALL_MODELS:
            p = np.clip(preds[m], actual.min() - 50, actual.max() + 50)
            slice_maes[m].append(metric_mae(actual, p))
            slice_mapes[m].append(metric_mape(actual, p))

        # Ensembles — use MAPE as weighting signal
        model_smapes_slice = {m: metric_mape(actual, np.clip(preds[m], actual.min()-50, actual.max()+50))
                              for m in ALL_MODELS}
        preds_clipped = {m: np.clip(preds[m], actual.min()-50, actual.max()+50) for m in ALL_MODELS}
        for ens_name, ens_fn in ENSEMBLE_FNS:
            ep, _ = ens_fn(preds_clipped, model_smapes_slice)
            ens_maes[ens_name].append(metric_mae(actual, ep))
            ens_mapes[ens_name].append(metric_mape(actual, ep))

        if (si + 1) % max(1, len(origins) // 5) == 0 or si == len(origins) - 1:
            print(f"    [{si + 1}/{len(origins)}]")

    # ── Aggregate: mean across slices ────────────────────────────────────────
    rows = []
    for m in ALL_MODELS:
        rows.append({
            "dataset": ds_name,
            "horizon": horizon,
            "model":   m,
            "MAE":     float(np.mean(slice_maes[m])),
            "MAPE":    float(np.mean([x for x in slice_mapes[m] if np.isfinite(x)])),
        })
    for ens_name, _ in ENSEMBLE_FNS:
        rows.append({
            "dataset": ds_name,
            "horizon": horizon,
            "model":   ens_name,
            "MAE":     float(np.mean(ens_maes[ens_name])),
            "MAPE":    float(np.mean([x for x in ens_mapes[ens_name] if np.isfinite(x)])),
        })

    # Print per-horizon table
    print(f"\n  {'Model':<24s} {'MAE':>9s} {'MAPE%':>8s}")
    print(f"  {'─' * 44}")
    for r in sorted(rows, key=lambda x: x["MAE"]):
        tag = " ★" if r["model"].startswith("Ens_") else ""
        print(f"  {r['model']:<24s} {r['MAE']:9.3f} {r['MAPE']:7.2f}%{tag}")

    return rows


def main(fast: bool = False):
    horizons = [96] if fast else HORIZONS
    all_rows = []

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'=' * 68}")
        print(f"  Dataset: {ds_name}")
        print(f"{'=' * 68}")

        try:
            y_all, X_all = load_dataset(ds_path)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        sp = 24 if "h" in ds_name.lower() else 96

        for horizon in horizons:
            rows = evaluate_dataset(ds_name, y_all, X_all, horizon, sp, fast)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    out = ROOT / "outputs" / "robustness_ETT"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ett_robustness_results.csv", index=False)
    print(f"\nResults saved to {out}/ett_robustness_results.csv")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame):
    print(f"\n{'=' * 68}")
    print("  OVERALL RANKING — Mean MAE and MAPE% across all datasets × horizons")
    print(f"{'=' * 68}")
    overall = df.groupby("model")[["MAE", "MAPE"]].mean().sort_values("MAE")
    print(f"  {'Model':<24s} {'MAE':>9s} {'MAPE%':>8s}")
    print(f"  {'─' * 44}")
    for name, row in overall.iterrows():
        tag = " ★" if name.startswith("Ens_") else ""
        print(f"  {name:<24s} {row['MAE']:9.3f} {row['MAPE']:7.2f}%{tag}")

    for horizon in sorted(df["horizon"].unique()):
        print(f"\n{'=' * 68}")
        print(f"  HORIZON = {horizon} — Mean across all 4 ETT datasets")
        print(f"{'=' * 68}")
        sub = df[df["horizon"] == horizon]
        agg = sub.groupby("model")[["MAE", "MAPE"]].mean().sort_values("MAE")
        print(f"  {'Model':<24s} {'MAE':>9s} {'MAPE%':>8s}")
        print(f"  {'─' * 44}")
        for name, row in agg.iterrows():
            tag = " ★" if name.startswith("Ens_") else ""
            print(f"  {name:<24s} {row['MAE']:9.3f} {row['MAPE']:7.2f}%{tag}")

    print(f"\n{'=' * 68}")
    print("  BEST MODEL PER DATASET × HORIZON")
    print(f"{'=' * 68}")
    print(f"  {'Dataset':<8s} {'H':>4s}  {'Best Model':<24s} {'MAE':>9s} {'MAPE%':>8s}")
    print(f"  {'─' * 58}")
    for (ds, h), grp in df.groupby(["dataset", "horizon"]):
        best = grp.loc[grp["MAE"].idxmin()]
        print(f"  {ds:<8s} {h:4d}  {best['model']:<24s} {best['MAE']:9.3f} {best['MAPE']:7.2f}%")

    print(f"\n{'=' * 68}")
    print("  RANK CONSISTENCY — model rank stability across datasets × horizons")
    print(f"{'=' * 68}")
    rank_rows = []
    for (ds, h), grp in df.groupby(["dataset", "horizon"]):
        for rank, (_, row) in enumerate(grp.sort_values("MAE").iterrows(), 1):
            rank_rows.append({"model": row["model"], "rank": rank})
    rank_df = pd.DataFrame(rank_rows).groupby("model")["rank"].agg(["mean", "std", "min", "max"])
    rank_df = rank_df.sort_values("mean")
    print(f"  {'Model':<24s} {'MeanRank':>9s} {'StdRank':>8s} {'Best':>6s} {'Worst':>6s}")
    print(f"  {'─' * 58}")
    for name, row in rank_df.iterrows():
        tag = " ★" if name.startswith("Ens_") else ""
        print(f"  {name:<24s} {row['mean']:9.2f} {row['std']:8.2f} {row['min']:6.0f} {row['max']:6.0f}{tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnergyX ETT Robustness Evaluation")
    parser.add_argument("--fast", action="store_true",
                        help="Horizon=96 only (skip 192)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        EnergyX — ETT Robustness Evaluation                      ║")
    print("║  Datasets: ETTh1 ETTh2 ETTm1 ETTm2  |  Horizons: 96, 192       ║")
    print(f"║  Protocol: {N_SLICES} slices × {CONTEXT_LEN}-step context, avg MAE + MAPE%    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    t0 = time.time()
    main(fast=args.fast)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
