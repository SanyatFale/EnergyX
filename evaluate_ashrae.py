"""
ASHRAE Building Energy Benchmark — EnergyX
==========================================
Evaluates forecasting accuracy on ASHRAE electricity and chilled-water datasets.
Target column: meter_reading

Three comparison groups, each timed:

  Group A — Standalone baselines
              Default params, no CV search, no ensemble.
              Direct stand-ins for Seasonal Naïve / AutoARIMA / ETS / ML defaults.

  Group B — Deterministic AutoML / EnergyX ensemble (univariate)
              Same CV hyperparameter search + inverse-SMAPE ensemble that the
              EnergyX agent runs mechanically, but with no LLM calls.
              This is the W4 ablation baseline: it proves that the accuracy
              comes from the model/ensemble design, while the LLM contributes
              the NL interface, adaptive selection, and explanations.

  Group C — EnergyX multivariate ensemble
              Features selected by LightGBM importance > IMPORTANCE_THRESHOLD
              (default 200). RF_mv + LGBM_mv CV-tuned + ensemble.
              Also runs a feature-count ablation study: shows how MAE changes
              as features are added in descending importance order, so we can
              justify the threshold choice empirically rather than arbitrarily.

Split      : 80 / 20 train / test (chronological)
Horizons   : 24, 96, 168 steps (hourly → 1-day, 4-day, 7-day ahead)
Metrics    : MAE, RMSE, SMAPE, MASE
Seasonal period: 24 (daily cycle for hourly data)

Usage:
    python evaluate_ashrae.py
    python evaluate_ashrae.py --horizons 24              # single horizon
    python evaluate_ashrae.py --skip-nbeats              # skip slow neural model
    python evaluate_ashrae.py --importance-threshold 150 # custom threshold
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Import core utilities from the ETT benchmark (avoid code duplication) ─────
from evaluate_forecasting import (
    _create_lag_features,
    _lag_col_names,
    _recursive_forecast,
    _cv_score_univariate,
    _cv_score_multivariate,
    _param_combos,
    compute_metrics,
    metric_mape,
    metric_smape,
    forecast_naive,
    forecast_seasonal_naive,
    forecast_arima,
    _fit_predict_ets,
    _fit_predict_rf_uni,
    _fit_predict_lgbm_uni,
    _fit_predict_nbeats,
    _fit_predict_rf_mv,
    _fit_predict_lgbm_mv,
    cv_search_ets,
    cv_search_rf_uni,
    cv_search_lgbm_uni,
    cv_search_rf_mv,
    cv_search_lgbm_mv,
    cv_search_nbeats,
    train_final_rf_uni,
    train_final_lgbm_uni,
    train_final_rf_mv,
    train_final_lgbm_mv,
    train_final_nbeats,
    TrainedUniModel,
    TrainedMVModel,
    TrainedNBEATS,
    ens_inv_smape,
    MAX_WINDOWS,
    SEARCH_SPACES,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET = "meter_reading"
SEASONAL_PERIOD = 24        # daily seasonality for hourly data
HORIZONS = [24, 96, 168]
TRAIN_RATIO = 0.80
IMPORTANCE_THRESHOLD = 200  # select features with LightGBM importance > this

# Exogenous feature candidates (weather + calendar — excludes pre-computed lags
# which are target-derived and would create leakage for a fair MV baseline)
EXOG_CANDIDATES = [
    "air_temperature", "cloud_coverage", "dew_temperature",
    "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
    "hour", "dayofweek", "month", "day", "is_weekend",
]

DATASETS = {
    "ASHRAE_elec":    ROOT / "data/raw/building_energy_data (copy 1)_elec.csv",
    "ASHRAE_chilled": ROOT / "data/raw/building_energy_data (copy 1)_chilled.csv",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ashrae(path: Path) -> pd.DataFrame:
    """Load an ASHRAE CSV, handling both named and unnamed timestamp columns."""
    df = pd.read_csv(path)
    # Chilled-water file exports index as an unnamed column
    if df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    return df


def extract_arrays(df: pd.DataFrame):
    """Return (y, X_exog, available_feature_names)."""
    y = df[TARGET].values.astype(float)
    available = [c for c in EXOG_CANDIDATES if c in df.columns]
    X = df[available].fillna(method="ffill").fillna(0.0).values.astype(float)
    return y, X, available


# ── Feature importance ─────────────────────────────────────────────────────────

def compute_feature_importance(y_train: np.ndarray, X_train: np.ndarray,
                                feat_names: list) -> pd.Series:
    """Train LightGBM on training set; return importance Series sorted descending."""
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(pd.DataFrame(X_train, columns=feat_names), y_train)
    return pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)


def select_features_by_threshold(importance: pd.Series,
                                  threshold: int = IMPORTANCE_THRESHOLD) -> list:
    """
    Return all features whose LightGBM split importance exceeds threshold.
    Falls back to the single most important feature if none pass.
    """
    selected = importance[importance > threshold].index.tolist()
    if not selected:
        selected = [importance.idxmax()]
    return selected


def feature_count_ablation(y_train: np.ndarray, y_test: np.ndarray,
                             X_train: np.ndarray, X_test: np.ndarray,
                             feat_names: list, importance: pd.Series,
                             horizon: int) -> pd.DataFrame:
    """
    Evaluate a direct LightGBM model on the first `horizon` test points for each
    cumulative prefix of features (sorted by descending importance).

    Returns a DataFrame with columns: n_feat, features, MAE, RMSE, SMAPE.
    This ablation justifies the threshold choice: we pick the threshold where
    adding more features stops improving (or starts hurting) MAE.
    """
    rows = []
    ordered = importance.index.tolist()
    for k in range(1, len(ordered) + 1):
        feats = ordered[:k]
        idx = [feat_names.index(f) for f in feats]
        Xtr = pd.DataFrame(X_train[:, idx], columns=feats)
        Xte = pd.DataFrame(X_test[:idx.__len__(), idx] if False else X_test[:, idx], columns=feats)
        mdl = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        mdl.fit(Xtr, y_train)
        pred = mdl.predict(Xte.iloc[:horizon])
        actual = y_test[:horizon]
        mae  = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        denom = np.abs(actual) + np.abs(pred)
        smape = float(np.mean(2 * np.abs(actual - pred) / np.where(denom > 0, denom, 1)) * 100)
        rows.append({
            "n_feat": k,
            "features": ", ".join(feats),
            "importance_sum": int(importance[feats].sum()),
            "MAE": mae, "RMSE": rmse, "SMAPE": smape,
        })
    return pd.DataFrame(rows)


# ── Rolling evaluation core ────────────────────────────────────────────────────

def _get_origins(y_test: np.ndarray, horizon: int):
    """Return rolling-origin start indices, capped at MAX_WINDOWS."""
    total = max(1, (len(y_test) - horizon) // horizon + 1)
    n_win = min(total, MAX_WINDOWS)
    if total > MAX_WINDOWS:
        idx = np.linspace(0, total - 1, n_win, dtype=int)
    else:
        idx = np.arange(total)
    return [int(i) * horizon for i in idx], n_win


def rolling_eval_uni(fn_map: dict, y_train, y_test, origins, horizon, sp):
    """
    Rolling-origin evaluation for univariate model functions.

    fn_map: {model_name: callable(y_ctx, horizon) -> array-like}

    Returns:
        preds_store : {name: list[np.ndarray]}  — per-window predictions
        y_true_all  : np.ndarray                — concatenated actuals
        infer_times : {name: float}             — total inference seconds
    """
    preds_store = {n: [] for n in fn_map}
    infer_times = {n: 0.0 for n in fn_map}
    actuals = []

    for origin in origins:
        end = origin + horizon
        if end > len(y_test):
            break
        actual = y_test[origin:end]
        actuals.append(actual)
        y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train

        for name, fn in fn_map.items():
            t0 = time.perf_counter()
            try:
                p = np.asarray(fn(y_ctx, horizon), dtype=float)
            except Exception:
                p = np.full(horizon, float(y_ctx[-1]))
            infer_times[name] += time.perf_counter() - t0
            preds_store[name].append(p)

    y_true_all = np.concatenate(actuals) if actuals else np.array([])
    return preds_store, y_true_all, infer_times


def rolling_eval_mv(fn_map: dict, y_train, y_test, X_test_mv, origins, horizon, sp):
    """
    Rolling-origin evaluation for multivariate model functions.

    fn_map: {model_name: callable(X_win, horizon) -> array-like}
    X_test_mv: feature matrix for the entire test split

    Returns: preds_store, infer_times (y_true_all already known from uni eval)
    """
    preds_store = {n: [] for n in fn_map}
    infer_times = {n: 0.0 for n in fn_map}

    for origin in origins:
        end = origin + horizon
        if end > len(y_test):
            break
        X_win = X_test_mv[origin:end]
        y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train

        for name, fn in fn_map.items():
            t0 = time.perf_counter()
            try:
                p = np.asarray(fn(X_win, horizon), dtype=float)
            except Exception:
                p = np.full(horizon, float(y_ctx[-1]))
            infer_times[name] += time.perf_counter() - t0
            preds_store[name].append(p)

    return preds_store, infer_times


def aggregate_metrics(preds_store, y_true_all, y_train, sp, infer_times):
    """Compute MAE/RMSE/SMAPE/MASE for each model in preds_store."""
    results = {}
    for name, windows in preds_store.items():
        y_pred = np.concatenate(windows)
        clip_lo = y_true_all.min() - abs(y_true_all.max()) * 0.5
        clip_hi = y_true_all.max() + abs(y_true_all.max()) * 0.5
        y_pred = np.clip(y_pred, clip_lo, clip_hi)
        m = compute_metrics(y_true_all, y_pred, y_train, sp)
        m["infer_s"] = infer_times.get(name, 0.0)
        results[name] = m
    return results


def compute_ensemble(preds_store, y_true_all, y_train, sp):
    """Inverse-SMAPE weighted ensemble over all models in preds_store."""
    concat = {n: np.concatenate(v) for n, v in preds_store.items()}
    smapes = {n: metric_smape(y_true_all, p) for n, p in concat.items()}
    ens_preds, _ = ens_inv_smape(concat, smapes)
    return compute_metrics(y_true_all, ens_preds, y_train, sp), smapes


# ── Printing helpers ───────────────────────────────────────────────────────────

def _print_row(name, m, prefix="  "):
    print(
        f"{prefix}{name:<24s}  MAE={m['MAE']:8.3f}  RMSE={m['RMSE']:8.3f}"
        f"  SMAPE={m['SMAPE']:6.2f}%  MASE={m['MASE']:.4f}"
        f"  infer={m.get('infer_s', 0.0):.2f}s"
    )


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate_ashrae(horizons=None, skip_nbeats=False, importance_threshold=IMPORTANCE_THRESHOLD):
    if horizons is None:
        horizons = HORIZONS

    all_rows = []   # metric rows
    time_rows = []  # timing summary rows

    out_dir_pre = ROOT / "outputs" / "benchmark"
    out_dir_pre.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'=' * 72}")
        print(f"  Dataset : {ds_name}")
        print(f"  Path    : {ds_path.name}")
        print(f"{'=' * 72}")

        df = load_ashrae(ds_path)
        y_all, X_all, feat_names = extract_arrays(df)
        n = len(y_all)
        split = int(n * TRAIN_RATIO)
        y_train, y_test = y_all[:split], y_all[split:]
        X_train_all, X_test_all = X_all[:split], X_all[split:]

        print(f"  Rows: {n}   Train: {split}   Test: {len(y_test)}")
        print(f"  Exog candidates ({len(feat_names)}): {feat_names}")

        # ── Feature importance + ablation (once per dataset) ─────────────────
        print("\n  [Feature Importance] Computing on training set …", end="", flush=True)
        t_fi = time.perf_counter()
        fi_series = compute_feature_importance(y_train, X_train_all, feat_names)
        selected_feats = select_features_by_threshold(fi_series, importance_threshold)
        fi_time = time.perf_counter() - t_fi
        print(f" {fi_time:.1f}s")
        print(f"  Ranked importances (threshold={importance_threshold}):")
        for feat, imp in fi_series.items():
            marker = " ◄ selected" if feat in selected_feats else ""
            print(f"    {feat:<30s} {imp:6.0f}{marker}")
        print(f"  → {len(selected_feats)} features selected: {selected_feats}")

        # ── Feature-count ablation (at H=96 as representative horizon) ───────
        print(f"\n  [Feature Ablation] MAE as features added in importance order (H=96):")
        abl_df = feature_count_ablation(
            y_train, y_test, X_train_all, X_test_all, feat_names, fi_series, horizon=96
        )
        print(f"  {'n':>3}  {'MAE':>8}  {'RMSE':>8}  {'SMAPE%':>7}  features")
        print(f"  {'─' * 65}")
        for _, row in abl_df.iterrows():
            mark = " ◄ threshold" if int(row["n_feat"]) == len(selected_feats) else ""
            print(
                f"  {int(row['n_feat']):>3}  {row['MAE']:8.2f}"
                f"  {row['RMSE']:8.2f}  {row['SMAPE']:6.2f}%  {row['features']}{mark}"
            )
        # Save ablation
        abl_path = out_dir_pre / f"feature_ablation_{ds_name}.csv"
        abl_df.to_csv(abl_path, index=False)

        # Slice selected feature columns
        sel_idx = [feat_names.index(f) for f in selected_feats]
        X_train_mv = X_train_all[:, sel_idx]
        X_test_mv = X_test_all[:, sel_idx]

        for horizon in horizons:
            print(f"\n  ── Horizon H={horizon} {'─' * 52}")
            origins, n_win = _get_origins(y_test, horizon)
            print(f"     Rolling windows: {n_win}")

            # ═════════════════════════════════════════════════════════════════
            # GROUP A  —  Standalone baselines (default params, no CV search)
            # ═════════════════════════════════════════════════════════════════
            print(f"\n  [Group A] Standalone baselines — default params, no CV search")
            t_A = time.perf_counter()

            # Train tree/neural once with fixed default params
            t0 = time.perf_counter()
            rf_def = train_final_rf_uni(
                y_train, {"n_lags": 24, "n_estimators": 100, "max_depth": 10}
            )
            lgbm_def = train_final_lgbm_uni(
                y_train,
                {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100},
            )
            train_def_time = time.perf_counter() - t0

            fn_A = {
                "Naive":          lambda yc, h: forecast_naive(yc, h),
                "SeasonalNaive":  lambda yc, h: forecast_seasonal_naive(yc, h, SEASONAL_PERIOD),
                "ARIMA":          lambda yc, h: forecast_arima(yc, h),
                "ETS_default":    lambda yc, h: _fit_predict_ets(
                    yc, h, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD
                ),
                "RF_default":     lambda yc, h: rf_def.forecast(yc, h),
                "LGBM_default":   lambda yc, h: lgbm_def.forecast(yc, h),
            }
            if not skip_nbeats:
                t0 = time.perf_counter()
                nbeats_def = train_final_nbeats(
                    y_train, {"n_lags": 24, "hidden_size": 64, "epochs": 50}
                )
                train_def_time += time.perf_counter() - t0
                fn_A["NBEATS_default"] = lambda yc, h: nbeats_def.forecast(yc, h)

            predsA, y_true_all, infer_A = rolling_eval_uni(
                fn_A, y_train, y_test, origins, horizon, SEASONAL_PERIOD
            )
            resA = aggregate_metrics(predsA, y_true_all, y_train, SEASONAL_PERIOD, infer_A)
            t_A_total = time.perf_counter() - t_A

            print(f"     Total time: {t_A_total:.1f}s  (train defaults: {train_def_time:.1f}s)")
            for name, m in resA.items():
                _print_row(name, m, prefix="     ")
                all_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "A_standalone", "model": name, **m,
                })
                time_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "A_standalone", "model": name,
                    "cv_s": 0.0, "train_s": train_def_time,
                    "infer_s": m["infer_s"], "total_s": t_A_total,
                })

            # ═════════════════════════════════════════════════════════════════
            # GROUP B  —  Deterministic AutoML / EnergyX CV ensemble (univariate)
            # This is what EnergyX does mechanically, minus the LLM calls.
            # ═════════════════════════════════════════════════════════════════
            print(f"\n  [Group B] Deterministic AutoML / EnergyX CV ensemble (univariate)")
            t_B = time.perf_counter()

            cv_B = {}
            train_B = {}

            print("     CV search: ETS …", end="", flush=True)
            t0 = time.perf_counter()
            ets_p, _ = cv_search_ets(y_train, horizon, SEASONAL_PERIOD)
            cv_B["ETS_cv"] = time.perf_counter() - t0
            print(f" {cv_B['ETS_cv']:.1f}s → {ets_p}")

            print("     CV search: RF_uni …", end="", flush=True)
            t0 = time.perf_counter()
            rf_p, _ = cv_search_rf_uni(y_train, horizon)
            cv_B["RF_cv"] = time.perf_counter() - t0
            print(f" {cv_B['RF_cv']:.1f}s → n_lags={rf_p.get('n_lags')}, trees={rf_p.get('n_estimators')}")

            print("     CV search: LGBM_uni …", end="", flush=True)
            t0 = time.perf_counter()
            lgbm_p, _ = cv_search_lgbm_uni(y_train, horizon)
            cv_B["LGBM_cv"] = time.perf_counter() - t0
            print(f" {cv_B['LGBM_cv']:.1f}s → n_lags={lgbm_p.get('n_lags')}, leaves={lgbm_p.get('num_leaves')}")

            if not skip_nbeats:
                print("     CV search: N-BEATS …", end="", flush=True)
                t0 = time.perf_counter()
                nbeats_p, _ = cv_search_nbeats(y_train, horizon)
                cv_B["NBEATS_cv"] = time.perf_counter() - t0
                print(f" {cv_B['NBEATS_cv']:.1f}s → n_lags={nbeats_p.get('n_lags')}, hidden={nbeats_p.get('hidden_size')}")

            # Train final models with best params
            t0 = time.perf_counter()
            rf_cv = train_final_rf_uni(y_train, rf_p)
            train_B["RF_cv"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            lgbm_cv = train_final_lgbm_uni(y_train, lgbm_p)
            train_B["LGBM_cv"] = time.perf_counter() - t0

            fn_B = {
                "Naive":         lambda yc, h: forecast_naive(yc, h),
                "SeasonalNaive": lambda yc, h: forecast_seasonal_naive(yc, h, SEASONAL_PERIOD),
                "ARIMA":         lambda yc, h: forecast_arima(yc, h),
                "ETS_cv":        lambda yc, h: _fit_predict_ets(yc, h, **ets_p),
                "RF_cv":         lambda yc, h: rf_cv.forecast(yc, h),
                "LGBM_cv":       lambda yc, h: lgbm_cv.forecast(yc, h),
            }
            if not skip_nbeats:
                t0 = time.perf_counter()
                nbeats_cv = train_final_nbeats(y_train, nbeats_p)
                train_B["NBEATS_cv"] = time.perf_counter() - t0
                fn_B["NBEATS_cv"] = lambda yc, h: nbeats_cv.forecast(yc, h)

            predsB, _, infer_B = rolling_eval_uni(
                fn_B, y_train, y_test, origins, horizon, SEASONAL_PERIOD
            )
            resB = aggregate_metrics(predsB, y_true_all, y_train, SEASONAL_PERIOD, infer_B)
            ens_m_B, smapes_B = compute_ensemble(predsB, y_true_all, y_train, SEASONAL_PERIOD)
            ens_m_B["infer_s"] = 0.0

            t_B_total = time.perf_counter() - t_B
            total_cv_B = sum(cv_B.values())
            total_train_B = sum(train_B.values())
            print(
                f"     Total time: {t_B_total:.1f}s"
                f"  (CV search: {total_cv_B:.1f}s,"
                f" train: {total_train_B:.1f}s)"
            )

            for name, m in resB.items():
                _print_row(name, m, prefix="     ")
                all_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "B_cv_automl", "model": name, **m,
                })
                time_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "B_cv_automl", "model": name,
                    "cv_s": cv_B.get(name, 0.0),
                    "train_s": train_B.get(name, 0.0),
                    "infer_s": m["infer_s"],
                    "total_s": t_B_total,
                })

            _print_row("EnergyX_Ensemble", ens_m_B, prefix="     ** ")
            all_rows.append({
                "dataset": ds_name, "horizon": horizon,
                "group": "B_cv_automl", "model": "EnergyX_Ensemble", **ens_m_B,
            })
            time_rows.append({
                "dataset": ds_name, "horizon": horizon,
                "group": "B_cv_automl", "model": "EnergyX_Ensemble",
                "cv_s": total_cv_B, "train_s": total_train_B,
                "infer_s": 0.0, "total_s": t_B_total,
            })

            # ═════════════════════════════════════════════════════════════════
            # GROUP C  —  EnergyX multivariate ensemble (threshold-selected)
            # ═════════════════════════════════════════════════════════════════
            print(
                f"\n  [Group C] EnergyX MV ensemble — {len(selected_feats)} features"
                f" (importance>{importance_threshold}): {selected_feats}"
            )
            t_C = time.perf_counter()
            cv_C = {}
            train_C = {}

            print("     CV search: RF_mv …", end="", flush=True)
            t0 = time.perf_counter()
            rf_mv_p, _ = cv_search_rf_mv(y_train, X_train_mv, horizon)
            cv_C["RF_mv"] = time.perf_counter() - t0
            print(f" {cv_C['RF_mv']:.1f}s → trees={rf_mv_p.get('n_estimators')}")

            print("     CV search: LGBM_mv …", end="", flush=True)
            t0 = time.perf_counter()
            lgbm_mv_p, _ = cv_search_lgbm_mv(y_train, X_train_mv, horizon)
            cv_C["LGBM_mv"] = time.perf_counter() - t0
            print(f" {cv_C['LGBM_mv']:.1f}s → leaves={lgbm_mv_p.get('num_leaves')}")

            t0 = time.perf_counter()
            rf_mv_model = train_final_rf_mv(X_train_mv, y_train, rf_mv_p)
            train_C["RF_mv"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            lgbm_mv_model = train_final_lgbm_mv(X_train_mv, y_train, lgbm_mv_p)
            train_C["LGBM_mv"] = time.perf_counter() - t0

            fn_C = {
                "RF_mv":   lambda xw, h: rf_mv_model.forecast(xw, h),
                "LGBM_mv": lambda xw, h: lgbm_mv_model.forecast(xw, h),
            }

            predsC, infer_C = rolling_eval_mv(
                fn_C, y_train, y_test, X_test_mv, origins, horizon, SEASONAL_PERIOD
            )
            resC = aggregate_metrics(predsC, y_true_all, y_train, SEASONAL_PERIOD, infer_C)
            ens_m_C, _ = compute_ensemble(predsC, y_true_all, y_train, SEASONAL_PERIOD)
            ens_m_C["infer_s"] = 0.0

            t_C_total = time.perf_counter() - t_C
            total_cv_C = sum(cv_C.values())
            total_train_C = sum(train_C.values())
            print(
                f"     Total time: {t_C_total:.1f}s"
                f"  (CV search: {total_cv_C:.1f}s,"
                f" train: {total_train_C:.1f}s)"
            )

            for name, m in resC.items():
                _print_row(name, m, prefix="     ")
                all_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "C_mv", "model": name, **m,
                })
                time_rows.append({
                    "dataset": ds_name, "horizon": horizon,
                    "group": "C_mv", "model": name,
                    "cv_s": cv_C.get(name, 0.0),
                    "train_s": train_C.get(name, 0.0),
                    "infer_s": m["infer_s"],
                    "total_s": t_C_total,
                })

            _print_row("EnergyX_MV_Ensemble", ens_m_C, prefix="     ** ")
            all_rows.append({
                "dataset": ds_name, "horizon": horizon,
                "group": "C_mv", "model": "EnergyX_MV_Ensemble", **ens_m_C,
            })
            time_rows.append({
                "dataset": ds_name, "horizon": horizon,
                "group": "C_mv", "model": "EnergyX_MV_Ensemble",
                "cv_s": total_cv_C, "train_s": total_train_C,
                "infer_s": 0.0, "total_s": t_C_total,
            })

    # ── Save & summary ────────────────────────────────────────────────────────
    out_dir = ROOT / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(all_rows)
    df_tim = pd.DataFrame(time_rows)
    res_path = out_dir / "ashrae_benchmark_results.csv"
    tim_path = out_dir / "ashrae_timing.csv"
    df_res.to_csv(res_path, index=False)
    df_tim.to_csv(tim_path, index=False)

    _print_summary(df_res, df_tim)
    print(f"\nResults saved → {res_path}")
    print(f"Timing  saved → {tim_path}")
    return df_res, df_tim


# ── Summary printing ──────────────────────────────────────────────────────────

def _print_summary(df_res: pd.DataFrame, df_tim: pd.DataFrame):
    print(f"\n{'=' * 72}")
    print("  SUMMARY — Mean metrics across datasets × horizons")
    print(f"{'=' * 72}")

    GROUP_LABELS = {
        "A_standalone": "A — Standalone baselines",
        "B_cv_automl":  "B — Deterministic AutoML / EnergyX CV ensemble",
        "C_mv":         "C — EnergyX MV ensemble (top-3 features)",
    }

    for ds in df_res["dataset"].unique():
        sub_ds = df_res[df_res["dataset"] == ds]
        print(f"\n  {ds}")
        for h in sorted(sub_ds["horizon"].unique()):
            sub = sub_ds[sub_ds["horizon"] == h]
            print(f"\n    Horizon = {h}")
            for grp_key, grp_label in GROUP_LABELS.items():
                grp = sub[sub["group"] == grp_key].copy()
                if grp.empty:
                    continue
                print(f"      [{grp_label}]")
                print(f"      {'Model':<24s} {'MAE':>8s} {'RMSE':>8s} {'SMAPE%':>8s} {'MASE':>8s}")
                print(f"      {'─' * 62}")
                for _, row in grp.sort_values("MAE").iterrows():
                    tag = " ★" if "Ensemble" in row["model"] else ""
                    print(
                        f"      {row['model']:<24s}"
                        f" {row['MAE']:8.3f} {row['RMSE']:8.3f}"
                        f" {row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}"
                    )

    print(f"\n{'=' * 72}")
    print("  TIMING SUMMARY — Mean total wall-clock time per (dataset × horizon)")
    print(f"{'=' * 72}")
    tim_agg = (
        df_tim.groupby(["dataset", "horizon", "group"])["total_s"]
        .max()   # total_s is the same for all models in a group
        .reset_index()
    )
    print(f"  {'Dataset':<20s} {'H':>5s}  {'Group':<20s} {'Total(s)':>10s}")
    print(f"  {'─' * 62}")
    for _, row in tim_agg.iterrows():
        print(
            f"  {row['dataset']:<20s} {row['horizon']:>5d}"
            f"  {row['group']:<20s} {row['total_s']:10.1f}s"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASHRAE benchmark for EnergyX")
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=HORIZONS,
        help="Forecast horizons to evaluate (default: 24 96 168)",
    )
    parser.add_argument(
        "--skip-nbeats", action="store_true",
        help="Skip N-BEATS (slow neural model) for faster runs",
    )
    parser.add_argument(
        "--importance-threshold", type=int, default=IMPORTANCE_THRESHOLD,
        help=f"Min LightGBM feature importance for Group C MV selection (default: {IMPORTANCE_THRESHOLD})",
    )
    args = parser.parse_args()
    evaluate_ashrae(
        horizons=args.horizons,
        skip_nbeats=args.skip_nbeats,
        importance_threshold=args.importance_threshold,
    )
