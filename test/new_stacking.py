"""
TinyTS-Scientist — Stacking Ensemble Learning Experiment
=========================================================
True ensemble learning via stacking: a meta-learner trained on
out-of-fold base model predictions (not just aggregation).

Base models (top 3 from benchmark):
  1. N-BEATS       — best on ETTh1
  2. LightGBM_uni  — best overall univariate
  3. RF_combined   — best combined (lag + exogenous)

Meta-learners:
  - Ridge regression (linear stacking)
  - LightGBM (non-linear stacking)

Comparison baselines (aggregation):
  - Ens_InvSMAPE, Ens_Best, Ens_Median, Ens_TrimMean

Usage:
    python evaluate_stacking.py              # ETTh1 + ETTm1, all horizons
    python evaluate_stacking.py --fast       # ETTh1 only, h=24
"""

import argparse
import random
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Import helpers from the main benchmark script ─────────────────────────────

from evaluate_forecasting import (
    # Config
    ROOT, DATASETS, HORIZONS, TARGET, TIME_COL, FEATURE_COLS,
    TRAIN_RATIO, MAX_WINDOWS, CV_FOLDS, SEARCH_SPACES,
    # Metrics
    metric_mae, metric_rmse, metric_smape, metric_mape, metric_mase,
    compute_metrics,
    # CV
    cv_search_lgbm_uni, cv_search_rf_combined, cv_search_nbeats,
    # Feature construction
    _create_lag_features, _create_combined_features,
    _lag_col_names, _combined_col_names,
    # Recursive forecast
    _recursive_forecast, _recursive_forecast_combined,
    # Trained model holders
    TrainedUniModel, TrainedCombinedModel, TrainedNBEATS,
    # Train final models
    train_final_lgbm_uni, train_final_rf_combined, train_final_nbeats,
    # Statistical baselines (for context)
    forecast_naive, forecast_seasonal_naive,
    # Ensembles (aggregation baselines)
    ens_inv_smape, ens_best, ens_median, ens_trim_mean,
    # Data loading
    load_dataset,
)

# ── Stacking Config ──────────────────────────────────────────────────────────

BASE_MODELS = ["N-BEATS", "LightGBM_uni", "RF_combined"]
STACK_CV_FOLDS = 8  # more folds → more meta-training data
MIN_STACK_TRAIN = 200  # minimum points before first stacking fold


# ── Stacking: generate out-of-fold meta-features ────────────────────────────

def _generate_meta_features(
    y_train: np.ndarray,
    X_train: np.ndarray,
    horizon: int,
    sp: int,
    nbeats_params: dict,
    lgbm_uni_params: dict,
    rf_combined_params: dict,
) -> tuple:
    """Generate out-of-fold predictions from base models using expanding window.

    Returns:
        meta_X: (n_samples, 3) — each column is one base model's OOF predictions
        meta_y: (n_samples,) — actual values aligned with predictions
    """
    n = len(y_train)
    min_train = max(MIN_STACK_TRAIN, horizon * 3)

    if n < min_train + horizon:
        print(f"     WARNING: training set too small for stacking CV ({n} < {min_train + horizon})")
        return np.empty((0, 3)), np.empty(0)

    # Expanding window: generate non-overlapping OOF blocks
    # Each fold trains on data[:split_pt], predicts data[split_pt : split_pt+horizon]
    step = max(horizon, (n - min_train - horizon) // STACK_CV_FOLDS)
    fold_origins = []
    sp_cursor = min_train
    while sp_cursor + horizon <= n:
        fold_origins.append(sp_cursor)
        sp_cursor += step

    if not fold_origins:
        return np.empty((0, 3)), np.empty(0)

    print(f"     Stacking CV: {len(fold_origins)} folds, step={step}")

    all_preds = {m: [] for m in BASE_MODELS}
    all_actuals = []

    for fi, split_pt in enumerate(fold_origins):
        # 🔴 FIX: use 1-step ahead prediction only
        actual_block = np.array([y_train[split_pt]])
        h = 1

        if h == 0:
            continue

        y_fold = y_train[:split_pt]
        X_fold = X_train[:split_pt]
        # 🔴 FIX: 1-step ahead validation exogenous
        X_val = X_train[split_pt:split_pt+1]


        # N-BEATS
        try:
            from tinyts.tools.neural import train_simple_neural_model, forecast_neural_recursive
            nl = nbeats_params.get("n_lags", 24)
            model = train_simple_neural_model(
                y_fold, n_lags=nl,
                hidden_size=nbeats_params.get("hidden_size", 64),
                epochs=nbeats_params.get("epochs", 50),
                lr=0.001,
            )
            nbeats_preds = np.array(forecast_neural_recursive(model, y_fold[-nl:], 1))
        except Exception:
            nbeats_preds = np.full(h, y_fold[-1])
        all_preds["N-BEATS"].append(nbeats_preds)

        # LightGBM_uni
        try:
            import lightgbm as lgb
            nl = lgbm_uni_params.get("n_lags", 24)
            Xl, yt = _create_lag_features(y_fold, nl)
            cols = _lag_col_names(nl)
            model = lgb.LGBMRegressor(
                num_leaves=lgbm_uni_params.get("num_leaves", 31),
                learning_rate=lgbm_uni_params.get("learning_rate", 0.1),
                n_estimators=lgbm_uni_params.get("n_estimators", 100),
                random_state=42, verbose=-1,
            )
            model.fit(pd.DataFrame(Xl, columns=cols), yt)
            lgbm_preds = _recursive_forecast(model, y_fold[-nl:], h, cols)
        except Exception:
            lgbm_preds = np.full(h, y_fold[-1])
        all_preds["LightGBM_uni"].append(lgbm_preds)

        # RF_combined
        try:
            from sklearn.ensemble import RandomForestRegressor
            nl = rf_combined_params.get("n_lags", 24)
            Xc, yt = _create_combined_features(y_fold, X_fold, nl)
            cols = _combined_col_names(nl, FEATURE_COLS)
            model = RandomForestRegressor(
                n_estimators=rf_combined_params.get("n_estimators", 100),
                max_depth=rf_combined_params.get("max_depth", 10),
                random_state=42, n_jobs=-1,
            )
            model.fit(pd.DataFrame(Xc, columns=cols), yt)
            rf_preds = _recursive_forecast_combined(
                model, y_fold[-nl:], X_val[:1], 1, cols,
            )
        except Exception:
            rf_preds = np.full(h, y_fold[-1])
        all_preds["RF_combined"].append(rf_preds)

        all_actuals.append(actual_block)

        if (fi + 1) % max(1, len(fold_origins) // 3) == 0:
            print(f"     [fold {fi + 1}/{len(fold_origins)}]")

    if not all_actuals:
        return np.empty((0, 3)), np.empty(0)

    # Flatten into (n_samples, 3) meta-features
    meta_cols = []
    for m in BASE_MODELS:
        meta_cols.append(np.concatenate(all_preds[m]))

    meta_y = np.concatenate(all_actuals)

    # 🔴 FIX: last observed value feature
    last_vals = meta_y.copy()

    meta_X = np.column_stack(meta_cols + [last_vals])


    # Align lengths (safety)
    min_len = min(len(meta_X), len(meta_y))
    meta_X = meta_X[:min_len]
    meta_y = meta_y[:min_len]

    print(f"     Meta-training samples: {len(meta_y)}")
    return meta_X, meta_y


# ── Meta-learners ────────────────────────────────────────────────────────────

class StackedRidge:
    """Linear stacking via Ridge regression."""

    def __init__(self, alpha=1.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, meta_X, meta_y):
        if len(meta_X) < 5:
            self.fitted = False
            return
        self.model.fit(meta_X, meta_y)
        self.fitted = True
        coefs = self.model.coef_
        print(
            f"     Ridge coefs: "
            f"N-BEATS={coefs[0]:.3f}, "
            f"LightGBM_uni={coefs[1]:.3f}, "
            f"RF_combined={coefs[2]:.3f}, "
            f"LastVal={coefs[3]:.3f}"
        )

        print(f"     Ridge intercept: {self.model.intercept_:.3f}")

    def predict(self, meta_X):
        if not self.fitted:
            return np.mean(meta_X, axis=1)
        return self.model.predict(meta_X)


class StackedLightGBM:
    """Non-linear stacking via LightGBM meta-learner."""

    def __init__(self):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor(
            num_leaves=4, learning_rate=0.02, n_estimators=50,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        self.fitted = False

    def fit(self, meta_X, meta_y):
        if len(meta_X) < 10:
            self.fitted = False
            return
        cols = [f"base_{m}" for m in BASE_MODELS]
        self.model.fit(pd.DataFrame(meta_X, columns=cols), meta_y)
        self.fitted = True
        imp = dict(zip(cols, self.model.feature_importances_))
        print(f"     LGBM meta-learner importances: {imp}")

    def predict(self, meta_X):
        if not self.fitted:
            return np.mean(meta_X, axis=1)
        cols = [f"base_{m}" for m in BASE_MODELS]
        return self.model.predict(pd.DataFrame(meta_X, columns=cols))


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_stacking(datasets=None, horizons=None, fast=False):
    if datasets is None:
        datasets = {"ETTh2": DATASETS["ETTh2"] ,"ETTm1": DATASETS["ETTm1"], "ETTm2": DATASETS["ETTm2"]}
    if horizons is None:
        horizons = HORIZONS
    if fast:
        datasets = {"ETTh1": DATASETS["ETTh1"]}
        horizons = [24]

    rows = []

    for ds_name, ds_path in datasets.items():
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {ds_name}  ({ds_path.name})")
        print(f"{'=' * 70}")

        y_all, X_all = load_dataset(ds_path)
        n = len(y_all)
        train_end = int(n * TRAIN_RATIO)
        y_train = y_all[:train_end]
        y_test = y_all[train_end:]
        X_train = X_all[:train_end]
        X_test = X_all[train_end:]
        sp = 24 if "h" in ds_name.lower() else 96

        print(f"  n={n}  train={train_end}  test={len(y_test)}  sp={sp}")

        for horizon in horizons:
            print(f"\n  ── Horizon {horizon} {'─' * 50}")

            # ── Phase 1: CV param search for base models ──
            print("     CV param search for base models …", end="", flush=True)
            t0 = time.time()

            nbeats_p, _   = cv_search_nbeats(y_train, horizon)
            lgbm_uni_p, _ = cv_search_lgbm_uni(y_train, horizon)
            rf_comb_p, _  = cv_search_rf_combined(y_train, X_train, horizon)

            print(f" {time.time() - t0:.0f}s")
            print(f"     N-BEATS:      n_lags={nbeats_p.get('n_lags')}, hidden={nbeats_p.get('hidden_size')}, epochs={nbeats_p.get('epochs')}")
            print(f"     LightGBM_uni: n_lags={lgbm_uni_p.get('n_lags')}, leaves={lgbm_uni_p.get('num_leaves')}, lr={lgbm_uni_p.get('learning_rate')}")
            print(f"     RF_combined:  n_lags={rf_comb_p.get('n_lags')}, trees={rf_comb_p.get('n_estimators')}, depth={rf_comb_p.get('max_depth')}")

            # ── Phase 2: Generate meta-features via stacking CV ──
            print("     Generating OOF meta-features …")
            t0 = time.time()
            meta_X, meta_y = _generate_meta_features(
                y_train, X_train, horizon, sp,
                nbeats_p, lgbm_uni_p, rf_comb_p,
            )
            print(f"     Meta-feature generation: {time.time() - t0:.1f}s")

            # ── Phase 3: Train meta-learners ──
            print("     Training meta-learners …")
            # 🔴 FIX: horizon-wise stacking

            ridge_meta = []

            for h_step in range(horizon):

                idx = np.arange(h_step, len(meta_y), horizon)

                model = StackedRidge(alpha=1.0)

                if len(idx) > 5:
                    model.fit(meta_X[idx], meta_y[idx])

                ridge_meta.append(model)


            # 🔴 FIX: Horizon-wise LightGBM stacking

            lgbm_meta = []

            for h_step in range(horizon):

                # select samples belonging to this horizon step
                idx = np.arange(h_step, len(meta_y), horizon)

                model = StackedLightGBM()

                if len(idx) > 10:
                    model.fit(meta_X[idx], meta_y[idx])

                lgbm_meta.append(model)

            print(f"     Trained {len(lgbm_meta)} LightGBM meta-models (one per horizon step)")


            # ── Phase 4: Train final base models on full training set ──
            print("     Training final base models …", end="", flush=True)
            t0 = time.time()

            trained_nbeats = train_final_nbeats(y_train, nbeats_p)
            trained_lgbm   = train_final_lgbm_uni(y_train, lgbm_uni_p)
            trained_rf_comb = train_final_rf_combined(y_train, X_train, rf_comb_p)

            print(f" {time.time() - t0:.1f}s")

            # ── Phase 5: Rolling-origin evaluation ──
            total_windows = (len(y_test) - horizon) // horizon + 1
            if total_windows < 1:
                print(f"     Test set too short, skipping.")
                continue
            if total_windows > MAX_WINDOWS:
                indices = np.linspace(0, total_windows - 1, MAX_WINDOWS, dtype=int)
            else:
                indices = np.arange(total_windows)
            origins = [i * horizon for i in indices]
            n_win = len(origins)
            print(f"     Rolling eval: {n_win} windows (of {total_windows})")

            # Storage for individual model predictions
            ALL_MODELS = ["N-BEATS", "LightGBM_uni", "RF_combined"]
            preds_store = {m: [] for m in ALL_MODELS}
            # Meta-learner predictions
            stack_ridge_preds = []
            stack_lgbm_preds = []
            actuals_store = []

            for wi, origin in enumerate(origins):
                end = origin + horizon
                if end > len(y_test):
                    break

                actual = y_test[origin:end]
                actuals_store.append(actual)
                h = len(actual)

                y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train
                X_win = X_test[origin:end]
                has_exog = len(X_win) >= horizon

                # Base model predictions
                nbeats_pred = trained_nbeats.forecast(y_ctx, h)
                lgbm_pred = trained_lgbm.forecast(y_ctx, h)
                if has_exog:
                    rf_pred = trained_rf_comb.forecast(y_ctx, X_win, h)
                else:
                    rf_pred = np.full(h, y_ctx[-1])

                preds_store["N-BEATS"].append(nbeats_pred)
                preds_store["LightGBM_uni"].append(lgbm_pred)
                preds_store["RF_combined"].append(rf_pred)

                # Stack base predictions into meta-features for this window
                last_val = y_ctx[-1]

                window_meta_X =np.column_stack([
                    nbeats_pred,
                    lgbm_pred,
                    rf_pred,
                    np.full(h, last_val)
                ])


                # Meta-learner predictions
                ridge_preds = []

                for h_step in range(h):

                    ridge_preds.append(
                        ridge_meta[h_step].predict(
                            window_meta_X[h_step:h_step+1]
                        )[0]
                    )

                stack_ridge_preds.append(np.array(ridge_preds))

                # 🔴 FIX: Horizon-wise LightGBM prediction

                lgbm_preds = []

                for h_step in range(h):

                    pred = lgbm_meta[h_step].predict(
                        window_meta_X[h_step:h_step+1]
                    )[0]

                    lgbm_preds.append(pred)

                stack_lgbm_preds.append(np.array(lgbm_preds))


                if (wi + 1) % max(1, n_win // 5) == 0 or wi == n_win - 1:
                    print(f"     [{wi + 1}/{n_win}]")

            if not actuals_store:
                continue

            # ── Aggregate metrics ──
            y_true_all = np.concatenate(actuals_store)

            # Base model concat (used for aggregation baselines)
            model_concat = {}
            model_smapes = {}
            for name in ALL_MODELS:
                y_pred_all = np.concatenate(preds_store[name])
                y_pred_all = np.clip(y_pred_all, y_true_all.min() - 50, y_true_all.max() + 50)
                model_concat[name] = y_pred_all
                m = compute_metrics(y_true_all, y_pred_all, y_train, sp)
                model_smapes[name] = m["SMAPE"]

            # Stacking predictions
            stack_ridge_all = np.clip(
                np.concatenate(stack_ridge_preds),
                y_true_all.min() - 50, y_true_all.max() + 50,
            )
            stack_lgbm_all = np.clip(
                np.concatenate(stack_lgbm_preds),
                y_true_all.min() - 50, y_true_all.max() + 50,
            )

            # Aggregation baselines
            agg_inv, _      = ens_inv_smape(model_concat, model_smapes)
            agg_best, _     = ens_best(model_concat, model_smapes)
            agg_median, _   = ens_median(model_concat, model_smapes)
            agg_trim, _     = ens_trim_mean(model_concat, model_smapes)

            # Compute all metrics
            results_list = [
                ("N-BEATS",         np.concatenate(preds_store["N-BEATS"])),
                ("LightGBM_uni",    np.concatenate(preds_store["LightGBM_uni"])),
                ("RF_combined",     np.concatenate(preds_store["RF_combined"])),
                ("─── Aggregation", None),
                ("Agg_InvSMAPE",    agg_inv),
                ("Agg_Best",        agg_best),
                ("Agg_Median",      agg_median),
                ("Agg_TrimMean",    agg_trim),
                ("─── Stacking",    None),
                ("Stack_Ridge",     stack_ridge_all),
                ("Stack_LightGBM",  stack_lgbm_all),
            ]

            print(f"\n     {'Model':<22s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE%':>8s} {'MASE':>8s}")
            print(f"     {'─' * 58}")

            for name, preds in results_list:
                if preds is None:
                    print(f"     {name}")
                    continue
                preds_clipped = np.clip(preds, y_true_all.min() - 50, y_true_all.max() + 50)
                m = compute_metrics(y_true_all, preds_clipped, y_train, sp)
                rows.append({"dataset": ds_name, "horizon": horizon, "model": name, **m})
                print(
                    f"     {name:<22s} {m['MAE']:9.3f} {m['RMSE']:9.3f} "
                    f"{m['SMAPE']:7.2f}% {m['MASE']:8.4f}"
                )

    # ── Save ──
    df = pd.DataFrame(rows)
    out_dir = ROOT / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "new_stacking_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("  SUMMARY — Stacking vs Aggregation")
    print(f"{'=' * 70}")

    # Filter to ensemble/stacking models only
    ens_models = ["Agg_InvSMAPE", "Agg_Best", "Agg_Median", "Agg_TrimMean",
                  "Stack_Ridge", "Stack_LightGBM"]

    for (ds, h), grp in df.groupby(["dataset", "horizon"]):
        sub = grp[grp["model"].isin(ens_models + BASE_MODELS)]
        sub = sub.sort_values("MAE")
        print(f"\n  {ds} H={h}")
        print(f"  {'Model':<22s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE%':>8s} {'MASE':>8s}")
        print(f"  {'─' * 58}")
        for _, row in sub.iterrows():
            tag = " ★" if row["model"].startswith("Stack_") else ""
            print(
                f"  {row['model']:<22s} {row['MAE']:9.3f} {row['RMSE']:9.3f} "
                f"  {row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}"
            )

    # Overall
    print(f"\n{'=' * 70}")
    print("  OVERALL — Mean across datasets and horizons")
    print(f"{'=' * 70}")
    ens_df = df[df["model"].isin(ens_models + BASE_MODELS)]
    overall = ens_df.groupby("model")[["MAE", "RMSE", "SMAPE", "MASE"]].mean().sort_values("MAE")
    print(f"  {'Model':<22s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE%':>8s} {'MASE':>8s}")
    print(f"  {'─' * 58}")
    for name, row in overall.iterrows():
        tag = " ★" if name.startswith("Stack_") else ""
        print(
            f"  {name:<22s} {row['MAE']:9.3f} {row['RMSE']:9.3f} "
            f"  {row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}"
        )

    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyTS Stacking Ensemble Experiment")
    parser.add_argument("--fast", action="store_true",
                        help="Quick test (ETTh1 only, h=24)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   TinyTS-Scientist — Stacking Ensemble Learning Experiment     ║")
    print("║   ICLR 2026 TSALM Workshop                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Base models:  {BASE_MODELS}")
    print(f"  Meta-learners: Ridge, LightGBM")
    print(f"  Stacking CV:  {STACK_CV_FOLDS} folds (expanding window)")

    t_start = time.time()
    results = evaluate_stacking(fast=args.fast)
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
