"""
EnergyX — ETT Forecasting Benchmark
=============================================
ICLR 2026 TSALM Workshop Evaluation

Mirrors the agentic pipeline exactly:
  1. Expanding-window CV with parameter grid search (3 folds, same as
     ``_cv_univariate`` / ``_cv_multivariate`` in agent_tools.py).
  2. Best params selected by lowest mean MAPE across folds.
  3. Final model trained on full training set with best params.
  4. Rolling-origin evaluation on held-out 20% test set.

Models (11):
  Univariate  — Naive, SeasonalNaive, ARIMA (auto_arima), ETS,
                N-BEATS, RF_uni, LightGBM_uni
  Multivariate — RF_mv, LightGBM_mv  (concurrent features, matches
                 agent MV mode)
  Combined    — RF_combined, LightGBM_combined  (lag features +
                concurrent exogenous features)

Ensemble strategies (4):
  Ens_InvSMAPE   — inverse-SMAPE weighted (our default)
  Ens_Best       — best single model by SMAPE
  Ens_Median     — median across all models
  Ens_TrimMean   — trimmed mean (drop 3 worst, average rest)

Metrics: MAE, RMSE, SMAPE, MASE
Horizons: 24, 96, 168 steps
Datasets: ETTh1, ETTh2 (hourly), ETTm1, ETTm2 (15-min)

Usage:
    python evaluate_forecasting.py              # full run
    python evaluate_forecasting.py --fast       # 2 datasets, h=24
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

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "ETTh1": ROOT / "data/test/ETTh1.csv",
    "ETTh2": ROOT / "data/test/ETTh2.csv",
    "ETTm1": ROOT / "data/test/ETTm1.csv",
    "ETTm2": ROOT / "data/test/ETTm2.csv",
}

HORIZONS = [24, 96, 168]
TARGET = "OT"
TIME_COL = "date"
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TRAIN_RATIO = 0.8
ARIMA_CONTEXT_CAP = 720
MAX_WINDOWS = 50
CV_FOLDS = 3
MAX_COMBOS = 10  # random-sample cap, same as agent

# Search spaces — identical to MODEL_TEMPLATES in agent_tools.py
SEARCH_SPACES = {
    "ETS": {
        "trend": ["add", "mul", "none"],
        "seasonal": ["add", "mul", "none"],
        "seasonal_periods": [12, 24],  # agent defaults to 12; we add correct sp
    },
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15],
        "n_lags": [6, 12, 24],
    },
    "LightGBM": {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.1, 0.3],
        "n_estimators": [50, 100, 200],
        "n_lags": [6, 12, 24],
    },
    "N-BEATS": {
        "n_lags": [12, 24, 48],
        "hidden_size": [32, 64, 128],
        "epochs": [30, 50, 100],
    },
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def metric_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def metric_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def metric_smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def metric_mape(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def metric_mase(y_true, y_pred, y_train, sp=1):
    naive_err = np.abs(y_train[sp:] - y_train[:-sp])
    scale = np.mean(naive_err) if len(naive_err) > 0 else 1.0
    if scale < 1e-9:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def compute_metrics(y_true, y_pred, y_train, sp):
    return {
        "MAE": metric_mae(y_true, y_pred),
        "RMSE": metric_rmse(y_true, y_pred),
        "SMAPE": metric_smape(y_true, y_pred),
        "MASE": metric_mase(y_true, y_pred, y_train, sp),
    }


# ── CV param search — mirrors agent_tools._cv_univariate exactly ─────────────

def _param_combos(space):
    """Generate parameter combinations, capped at MAX_COMBOS (same as agent)."""
    if not space:
        return [{}]
    keys = list(space.keys())
    vals = list(space.values())
    combos = [dict(zip(keys, c)) for c in product(*vals)]
    if len(combos) > MAX_COMBOS:
        random.seed(42)
        combos = random.sample(combos, MAX_COMBOS)
    return combos


def _cv_score_univariate(fit_predict_fn, y, horizon, cv_folds=CV_FOLDS):
    """Expanding-window CV on univariate series. Returns mean MAPE.
    Matches the agent's _cv_univariate logic."""
    n = len(y)
    min_train = max(50, horizon * 2)
    if n < min_train + horizon:
        return float("inf")
    step = max(1, (n - min_train - horizon) // cv_folds)
    scores = []
    for i in range(cv_folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            preds = fit_predict_fn(y[:sp], horizon)
            actual = y[sp : sp + len(preds)]
            scores.append(metric_mape(actual, np.asarray(preds)))
        except Exception:
            scores.append(float("inf"))
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("inf")


def _cv_score_multivariate(fit_predict_fn, y, X, horizon, cv_folds=CV_FOLDS):
    """Expanding-window CV on multivariate data. Returns mean MAPE.
    Matches agent's _cv_multivariate."""
    n = len(y)
    min_train = max(50, horizon * 2)
    if n < min_train + horizon:
        return float("inf")
    step = max(1, (n - min_train - horizon) // max(cv_folds, 1))
    scores = []
    for i in range(max(cv_folds, 1)):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            preds = fit_predict_fn(y[:sp], X[:sp], X[sp : sp + horizon], horizon)
            actual = y[sp : sp + len(preds)]
            scores.append(metric_mape(actual, np.asarray(preds)))
        except Exception:
            scores.append(float("inf"))
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("inf")


# ── Feature construction ─────────────────────────────────────────────────────

def _create_lag_features(y, n_lags):
    X, yt = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags : i])
        yt.append(y[i])
    return np.array(X), np.array(yt)


def _create_combined_features(y, X_exog, n_lags):
    """Create [lag_1 … lag_k, feat_1 … feat_m] aligned feature matrix."""
    Xl, yt = _create_lag_features(y, n_lags)
    Xe = X_exog[n_lags:]  # align exogenous with lagged target
    assert len(Xl) == len(Xe), f"Length mismatch: lags={len(Xl)}, exog={len(Xe)}"
    return np.hstack([Xl, Xe]), yt


def _lag_col_names(n_lags):
    return [f"lag_{i+1}" for i in range(n_lags)]


def _combined_col_names(n_lags, feat_names):
    return _lag_col_names(n_lags) + list(feat_names)


# ── Recursive forecasting ────────────────────────────────────────────────────

def _recursive_forecast(model, last_lags, horizon, cols):
    """Univariate recursive: shift lag window, predict one step at a time."""
    preds = []
    current = last_lags.copy()
    for _ in range(horizon):
        row = pd.DataFrame([current], columns=cols)
        p = float(model.predict(row)[0])
        preds.append(p)
        current = np.append(current[1:], p)
    return np.array(preds)


def _recursive_forecast_combined(model, last_lags, X_future, horizon, cols):
    """Combined recursive: lag window shifts, exogenous features come from X_future."""
    preds = []
    current_lags = last_lags.copy()
    for t in range(min(horizon, len(X_future))):
        row_vals = np.concatenate([current_lags, X_future[t]])
        row = pd.DataFrame([row_vals], columns=cols)
        p = float(model.predict(row)[0])
        preds.append(p)
        current_lags = np.append(current_lags[1:], p)
    return np.array(preds)


# ── Model: train + forecast functions ─────────────────────────────────────────
# Each returns (model_state_dict) from train, np.ndarray from forecast.

# --- Statistical ---

def forecast_naive(y_ctx, horizon):
    return np.full(horizon, y_ctx[-1])

def forecast_seasonal_naive(y_ctx, horizon, sp):
    season = y_ctx[-sp:]
    return np.tile(season, int(np.ceil(horizon / sp)))[:horizon]

def forecast_arima(y_ctx, horizon):
    y = y_ctx[-ARIMA_CONTEXT_CAP:]
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            y, start_p=0, max_p=5, start_q=0, max_q=5,
            start_d=0, max_d=2, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore", trace=False,
        )
        return np.asarray(model.predict(n_periods=horizon), dtype=float)
    except Exception:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            return np.asarray(ARIMA(y, order=(1,1,1)).fit().forecast(steps=horizon), dtype=float)
        except Exception:
            return np.full(horizon, y[-1])

def _fit_predict_ets(y_train, horizon, trend="add", seasonal="add", seasonal_periods=24):
    """Fit ETS and return predictions. Used in both CV and final forecast."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y_train[-ARIMA_CONTEXT_CAP:]
    t = trend if trend != "none" else None
    s = seasonal if seasonal != "none" else None
    # Multiplicative components require strictly positive data
    if (t == "mul" or s == "mul") and np.any(y <= 0):
        t = "add" if t == "mul" else t
        s = "add" if s == "mul" else s
    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=seasonal_periods if s else None,
            trend=t,
            seasonal=s,
        )
        return np.asarray(model.fit(optimized=True).forecast(horizon), dtype=float)
    except Exception:
        # Ultimate fallback: simple exponential smoothing (no trend, no season)
        try:
            model = ExponentialSmoothing(y, trend=None, seasonal=None)
            return np.asarray(model.fit(optimized=True).forecast(horizon), dtype=float)
        except Exception:
            return np.full(horizon, y[-1])


# --- Tree univariate ---

def _fit_predict_rf_uni(y_train, horizon, n_lags=24, n_estimators=100, max_depth=10):
    from sklearn.ensemble import RandomForestRegressor
    Xl, yt = _create_lag_features(y_train, n_lags)
    cols = _lag_col_names(n_lags)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _recursive_forecast(model, y_train[-n_lags:], horizon, cols)


def _fit_predict_lgbm_uni(y_train, horizon, n_lags=24, num_leaves=31,
                           learning_rate=0.1, n_estimators=100):
    import lightgbm as lgb
    Xl, yt = _create_lag_features(y_train, n_lags)
    cols = _lag_col_names(n_lags)
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _recursive_forecast(model, y_train[-n_lags:], horizon, cols)


# --- Tree multivariate (concurrent features — agent MV mode) ---

def _fit_predict_rf_mv(y_train, X_train, X_test, horizon,
                        n_estimators=100, max_depth=10):
    from sklearn.ensemble import RandomForestRegressor
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return np.asarray(
        model.predict(pd.DataFrame(X_test[:horizon], columns=cols)), dtype=float,
    )


def _fit_predict_lgbm_mv(y_train, X_train, X_test, horizon,
                           num_leaves=31, learning_rate=0.1, n_estimators=100):
    import lightgbm as lgb
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return np.asarray(
        model.predict(pd.DataFrame(X_test[:horizon], columns=cols)), dtype=float,
    )


# --- Tree combined (lag + exogenous features) ---

def _fit_predict_rf_combined(y_train, X_train, X_test, horizon,
                              n_lags=24, n_estimators=100, max_depth=10):
    from sklearn.ensemble import RandomForestRegressor
    Xc, yt = _create_combined_features(y_train, X_train, n_lags)
    cols = _combined_col_names(n_lags, FEATURE_COLS)
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _recursive_forecast_combined(
        model, y_train[-n_lags:], X_test[:horizon], horizon, cols,
    )


def _fit_predict_lgbm_combined(y_train, X_train, X_test, horizon,
                                n_lags=24, num_leaves=31, learning_rate=0.1,
                                n_estimators=100):
    import lightgbm as lgb
    Xc, yt = _create_combined_features(y_train, X_train, n_lags)
    cols = _combined_col_names(n_lags, FEATURE_COLS)
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _recursive_forecast_combined(
        model, y_train[-n_lags:], X_test[:horizon], horizon, cols,
    )


# --- Neural (N-BEATS) ---

def _fit_predict_nbeats(y_train, horizon, n_lags=24, hidden_size=64, epochs=50):
    from tinyts.tools.neural import train_simple_neural_model, forecast_neural_recursive
    model = train_simple_neural_model(
        y_train, n_lags=n_lags, hidden_size=hidden_size,
        epochs=epochs, lr=0.001,
    )
    return np.array(forecast_neural_recursive(model, y_train[-n_lags:], horizon))


# ── CV param search per model ────────────────────────────────────────────────

def cv_search_ets(y_train, horizon, sp):
    """CV search over ETS configs. Returns best params."""
    space = SEARCH_SPACES["ETS"].copy()
    space["seasonal_periods"] = [sp]  # use correct sp for this dataset
    combos = _param_combos(space)
    best_score, best_params = float("inf"), {"trend": "add", "seasonal": "add", "seasonal_periods": sp}
    for p in combos:
        score = _cv_score_univariate(
            lambda y_tr, h: _fit_predict_ets(y_tr, h, **p),
            y_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_rf_uni(y_train, horizon):
    combos = _param_combos(SEARCH_SPACES["RF"])
    best_score, best_params = float("inf"), {"n_lags": 24, "n_estimators": 100, "max_depth": 10}
    for p in combos:
        score = _cv_score_univariate(
            lambda y_tr, h, _p=p: _fit_predict_rf_uni(y_tr, h, **_p),
            y_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_lgbm_uni(y_train, horizon):
    combos = _param_combos(SEARCH_SPACES["LightGBM"])
    best_score, best_params = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        score = _cv_score_univariate(
            lambda y_tr, h, _p=p: _fit_predict_lgbm_uni(y_tr, h, **_p),
            y_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_rf_mv(y_train, X_train, horizon):
    combos = _param_combos({"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]})
    best_score, best_params = float("inf"), {"n_estimators": 100, "max_depth": 10}
    for p in combos:
        score = _cv_score_multivariate(
            lambda y_tr, X_tr, X_te, h, _p=p: _fit_predict_rf_mv(y_tr, X_tr, X_te, h, **_p),
            y_train, X_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_lgbm_mv(y_train, X_train, horizon):
    combos = _param_combos({"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200]})
    best_score, best_params = float("inf"), {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        score = _cv_score_multivariate(
            lambda y_tr, X_tr, X_te, h, _p=p: _fit_predict_lgbm_mv(y_tr, X_tr, X_te, h, **_p),
            y_train, X_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_rf_combined(y_train, X_train, horizon):
    combos = _param_combos(SEARCH_SPACES["RF"])
    best_score, best_params = float("inf"), {"n_lags": 24, "n_estimators": 100, "max_depth": 10}
    for p in combos:
        score = _cv_score_multivariate(
            lambda y_tr, X_tr, X_te, h, _p=p: _fit_predict_rf_combined(y_tr, X_tr, X_te, h, **_p),
            y_train, X_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_lgbm_combined(y_train, X_train, horizon):
    combos = _param_combos(SEARCH_SPACES["LightGBM"])
    best_score, best_params = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        score = _cv_score_multivariate(
            lambda y_tr, X_tr, X_te, h, _p=p: _fit_predict_lgbm_combined(y_tr, X_tr, X_te, h, **_p),
            y_train, X_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


def cv_search_nbeats(y_train, horizon):
    combos = _param_combos(SEARCH_SPACES["N-BEATS"])
    best_score, best_params = float("inf"), {"n_lags": 24, "hidden_size": 64, "epochs": 50}
    for p in combos:
        score = _cv_score_univariate(
            lambda y_tr, h, _p=p: _fit_predict_nbeats(y_tr, h, **_p),
            y_train, horizon,
        )
        if score < best_score:
            best_score, best_params = score, p
    return best_params, best_score


# ── Trained model holders ────────────────────────────────────────────────────
# After CV, we train final models on the full training set and store them
# so rolling evaluation only does inference (not re-training tree/neural).

class TrainedUniModel:
    """Stores a trained univariate tree/neural model for fast re-forecast."""
    def __init__(self, model, n_lags, cols):
        self.model = model
        self.n_lags = n_lags
        self.cols = cols

    def forecast(self, y_ctx, horizon):
        return _recursive_forecast(self.model, y_ctx[-self.n_lags:], horizon, self.cols)


class TrainedCombinedModel:
    """Stores a trained combined (lag + exog) model."""
    def __init__(self, model, n_lags, cols):
        self.model = model
        self.n_lags = n_lags
        self.cols = cols

    def forecast(self, y_ctx, X_future, horizon):
        return _recursive_forecast_combined(
            self.model, y_ctx[-self.n_lags:], X_future[:horizon], horizon, self.cols,
        )


class TrainedMVModel:
    """Stores a trained concurrent-feature MV model."""
    def __init__(self, model, cols):
        self.model = model
        self.cols = cols

    def forecast(self, X_future, horizon):
        X_df = pd.DataFrame(X_future[:horizon], columns=self.cols)
        return np.asarray(self.model.predict(X_df), dtype=float)


class TrainedNBEATS:
    def __init__(self, model, n_lags):
        self.model = model
        self.n_lags = n_lags

    def forecast(self, y_ctx, horizon):
        from tinyts.tools.neural import forecast_neural_recursive
        return np.array(forecast_neural_recursive(
            self.model, y_ctx[-self.n_lags:], horizon,
        ))


def train_final_rf_uni(y_train, params):
    from sklearn.ensemble import RandomForestRegressor
    nl = params.get("n_lags", 24)
    Xl, yt = _create_lag_features(y_train, nl)
    cols = _lag_col_names(nl)
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return TrainedUniModel(model, nl, cols)


def train_final_lgbm_uni(y_train, params):
    import lightgbm as lgb
    nl = params.get("n_lags", 24)
    Xl, yt = _create_lag_features(y_train, nl)
    cols = _lag_col_names(nl)
    model = lgb.LGBMRegressor(
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 100),
        random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return TrainedUniModel(model, nl, cols)


def train_final_rf_mv(X_train, y_train, params):
    from sklearn.ensemble import RandomForestRegressor
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return TrainedMVModel(model, cols)


def train_final_lgbm_mv(X_train, y_train, params):
    import lightgbm as lgb
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = lgb.LGBMRegressor(
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 100),
        random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return TrainedMVModel(model, cols)


def train_final_rf_combined(y_train, X_train, params):
    from sklearn.ensemble import RandomForestRegressor
    nl = params.get("n_lags", 24)
    Xc, yt = _create_combined_features(y_train, X_train, nl)
    cols = _combined_col_names(nl, FEATURE_COLS)
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        random_state=42, n_jobs=-1,
    )
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return TrainedCombinedModel(model, nl, cols)


def train_final_lgbm_combined(y_train, X_train, params):
    import lightgbm as lgb
    nl = params.get("n_lags", 24)
    Xc, yt = _create_combined_features(y_train, X_train, nl)
    cols = _combined_col_names(nl, FEATURE_COLS)
    model = lgb.LGBMRegressor(
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 100),
        random_state=42, verbose=-1,
    )
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return TrainedCombinedModel(model, nl, cols)


def train_final_nbeats(y_train, params):
    from tinyts.tools.neural import train_simple_neural_model
    nl = params.get("n_lags", 24)
    model = train_simple_neural_model(
        y_train, n_lags=nl,
        hidden_size=params.get("hidden_size", 64),
        epochs=params.get("epochs", 50),
        lr=0.001,
    )
    return TrainedNBEATS(model, nl)


# ── Ensemble strategies ───────────────────────────────────────────────────────

def ens_inv_smape(model_preds, model_smapes):
    """Inverse-SMAPE weighted — mirrors combine_forecasts() in agent_tools."""
    weights = {n: 1.0 / max(s, 0.01) for n, s in model_smapes.items()}
    total = sum(weights.values())
    weights = {n: w / total for n, w in weights.items()}
    combined = sum(weights[n] * p for n, p in model_preds.items())
    return combined, "InvSMAPE"


def ens_best(model_preds, model_smapes):
    """Best single model."""
    best = min(model_smapes, key=model_smapes.get)
    return model_preds[best], f"Best={best}"


def ens_median(model_preds, model_smapes):
    """Median ensemble — robust to outlier models."""
    stacked = np.stack(list(model_preds.values()), axis=0)
    return np.median(stacked, axis=0), "Median"


def ens_trim_mean(model_preds, model_smapes):
    """Trimmed mean: drop 3 worst models by SMAPE, average the rest."""
    ranked = sorted(model_smapes, key=model_smapes.get)
    n_keep = max(3, len(ranked) - 3)
    keep = ranked[:n_keep]
    stacked = np.stack([model_preds[n] for n in keep], axis=0)
    return np.mean(stacked, axis=0), f"TrimMean(keep={n_keep})"


# ── Main evaluation loop ─────────────────────────────────────────────────────

def load_dataset(path):
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    y = df[TARGET].values.astype(float)
    X = df[FEATURE_COLS].values.astype(float)
    return y, X


def evaluate(datasets=None, horizons=None, fast=False):
    if datasets is None:
        datasets = DATASETS
    if horizons is None:
        horizons = HORIZONS
    if fast:
        datasets = {k: v for i, (k, v) in enumerate(datasets.items()) if i < 2}
        horizons = [24]

    rows = []
    timing_rows = []   # per-model timing: cv_s, train_s, infer_s

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

            # ── Phase 1: CV param search (per-model timed) ──────────────────
            print("     CV param search (per model):")
            cv_times = {}

            t0 = time.perf_counter()
            ets_params, _  = cv_search_ets(y_train, horizon, sp)
            cv_times["ETS"] = time.perf_counter() - t0
            print(f"       ETS          {cv_times['ETS']:5.1f}s  best={ets_params}")

            t0 = time.perf_counter()
            rf_uni_p, _    = cv_search_rf_uni(y_train, horizon)
            cv_times["RF_uni"] = time.perf_counter() - t0
            print(f"       RF_uni       {cv_times['RF_uni']:5.1f}s  n_lags={rf_uni_p.get('n_lags')}, trees={rf_uni_p.get('n_estimators')}, depth={rf_uni_p.get('max_depth')}")

            t0 = time.perf_counter()
            lgbm_uni_p, _  = cv_search_lgbm_uni(y_train, horizon)
            cv_times["LightGBM_uni"] = time.perf_counter() - t0
            print(f"       LightGBM_uni {cv_times['LightGBM_uni']:5.1f}s  n_lags={lgbm_uni_p.get('n_lags')}, leaves={lgbm_uni_p.get('num_leaves')}, lr={lgbm_uni_p.get('learning_rate')}")

            t0 = time.perf_counter()
            rf_mv_p, _     = cv_search_rf_mv(y_train, X_train, horizon)
            cv_times["RF_mv"] = time.perf_counter() - t0
            print(f"       RF_mv        {cv_times['RF_mv']:5.1f}s  trees={rf_mv_p.get('n_estimators')}, depth={rf_mv_p.get('max_depth')}")

            t0 = time.perf_counter()
            lgbm_mv_p, _   = cv_search_lgbm_mv(y_train, X_train, horizon)
            cv_times["LightGBM_mv"] = time.perf_counter() - t0
            print(f"       LightGBM_mv  {cv_times['LightGBM_mv']:5.1f}s  leaves={lgbm_mv_p.get('num_leaves')}, lr={lgbm_mv_p.get('learning_rate')}")

            t0 = time.perf_counter()
            rf_comb_p, _   = cv_search_rf_combined(y_train, X_train, horizon)
            cv_times["RF_combined"] = time.perf_counter() - t0
            print(f"       RF_combined  {cv_times['RF_combined']:5.1f}s  n_lags={rf_comb_p.get('n_lags')}, trees={rf_comb_p.get('n_estimators')}")

            t0 = time.perf_counter()
            lgbm_comb_p, _ = cv_search_lgbm_combined(y_train, X_train, horizon)
            cv_times["LightGBM_combined"] = time.perf_counter() - t0
            print(f"       LGBM_combined{cv_times['LightGBM_combined']:5.1f}s  n_lags={lgbm_comb_p.get('n_lags')}, leaves={lgbm_comb_p.get('num_leaves')}")

            t0 = time.perf_counter()
            nbeats_p, _    = cv_search_nbeats(y_train, horizon)
            cv_times["N-BEATS"] = time.perf_counter() - t0
            print(f"       N-BEATS      {cv_times['N-BEATS']:5.1f}s  n_lags={nbeats_p.get('n_lags')}, hidden={nbeats_p.get('hidden_size')}, epochs={nbeats_p.get('epochs')}")
            print(f"       Total CV     {sum(cv_times.values()):5.1f}s")

            # ── Phase 2: Train final models (per-model timed) ────────────────
            print("     Training final models (per model):")
            train_times = {}
            trained = {}

            t0 = time.perf_counter()
            trained["RF_uni"] = train_final_rf_uni(y_train, rf_uni_p)
            train_times["RF_uni"] = time.perf_counter() - t0
            print(f"       RF_uni           {train_times['RF_uni']:.2f}s")

            t0 = time.perf_counter()
            trained["LightGBM_uni"] = train_final_lgbm_uni(y_train, lgbm_uni_p)
            train_times["LightGBM_uni"] = time.perf_counter() - t0
            print(f"       LightGBM_uni     {train_times['LightGBM_uni']:.2f}s")

            t0 = time.perf_counter()
            trained["RF_mv"] = train_final_rf_mv(X_train, y_train, rf_mv_p)
            train_times["RF_mv"] = time.perf_counter() - t0
            print(f"       RF_mv            {train_times['RF_mv']:.2f}s")

            t0 = time.perf_counter()
            trained["LightGBM_mv"] = train_final_lgbm_mv(X_train, y_train, lgbm_mv_p)
            train_times["LightGBM_mv"] = time.perf_counter() - t0
            print(f"       LightGBM_mv      {train_times['LightGBM_mv']:.2f}s")

            t0 = time.perf_counter()
            trained["RF_combined"] = train_final_rf_combined(y_train, X_train, rf_comb_p)
            train_times["RF_combined"] = time.perf_counter() - t0
            print(f"       RF_combined      {train_times['RF_combined']:.2f}s")

            t0 = time.perf_counter()
            trained["LightGBM_combined"] = train_final_lgbm_combined(y_train, X_train, lgbm_comb_p)
            train_times["LightGBM_combined"] = time.perf_counter() - t0
            print(f"       LightGBM_combined{train_times['LightGBM_combined']:.2f}s")

            t0 = time.perf_counter()
            trained["N-BEATS"] = train_final_nbeats(y_train, nbeats_p)
            train_times["N-BEATS"] = time.perf_counter() - t0
            print(f"       N-BEATS          {train_times['N-BEATS']:.2f}s")
            print(f"       Total train      {sum(train_times.values()):.2f}s")

            # ── Phase 3: Rolling-origin evaluation ──
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

            ALL_MODELS = [
                "Naive", "SeasonalNaive", "ARIMA", "ETS", "N-BEATS",
                "RF_uni", "LightGBM_uni",
                "RF_mv", "LightGBM_mv",
                "RF_combined", "LightGBM_combined",
            ]
            preds_store = {m: [] for m in ALL_MODELS}
            infer_times = {m: 0.0 for m in ALL_MODELS}
            actuals_store = []

            for wi, origin in enumerate(origins):
                end = origin + horizon
                if end > len(y_test):
                    break

                actual = y_test[origin:end]
                actuals_store.append(actual)

                # Context = train + observed test so far
                y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train
                X_win = X_test[origin:end]
                has_exog = len(X_win) >= horizon

                # --- Statistical (re-fit per window, same as agent per-call) ---
                for _nm, _fn_args in [
                    ("Naive",        lambda: forecast_naive(y_ctx, horizon)),
                    ("SeasonalNaive",lambda: forecast_seasonal_naive(y_ctx, horizon, sp)),
                    ("ARIMA",        lambda: forecast_arima(y_ctx, horizon)),
                    ("ETS",          lambda: _fit_predict_ets(y_ctx, horizon, **ets_params)),
                ]:
                    _t = time.perf_counter()
                    preds_store[_nm].append(_fn_args())
                    infer_times[_nm] += time.perf_counter() - _t

                # --- Tree / Neural (trained once, inference per window) ---
                for _nm, _fn in [
                    ("N-BEATS",      lambda: trained["N-BEATS"].forecast(y_ctx, horizon)),
                    ("RF_uni",       lambda: trained["RF_uni"].forecast(y_ctx, horizon)),
                    ("LightGBM_uni", lambda: trained["LightGBM_uni"].forecast(y_ctx, horizon)),
                ]:
                    _t = time.perf_counter()
                    preds_store[_nm].append(_fn())
                    infer_times[_nm] += time.perf_counter() - _t

                if has_exog:
                    for _nm, _fn in [
                        ("RF_mv",              lambda: trained["RF_mv"].forecast(X_win, horizon)),
                        ("LightGBM_mv",        lambda: trained["LightGBM_mv"].forecast(X_win, horizon)),
                        ("RF_combined",        lambda: trained["RF_combined"].forecast(y_ctx, X_win, horizon)),
                        ("LightGBM_combined",  lambda: trained["LightGBM_combined"].forecast(y_ctx, X_win, horizon)),
                    ]:
                        _t = time.perf_counter()
                        preds_store[_nm].append(_fn())
                        infer_times[_nm] += time.perf_counter() - _t
                else:
                    for nm in ["RF_mv", "LightGBM_mv", "RF_combined", "LightGBM_combined"]:
                        preds_store[nm].append(np.full(horizon, y_ctx[-1]))

                if (wi + 1) % max(1, n_win // 5) == 0 or wi == n_win - 1:
                    print(f"     [{wi + 1}/{n_win}]")

            if not actuals_store:
                continue

            # ── Aggregate metrics ──────────────────────────────────────────────
            y_true_all = np.concatenate(actuals_store)
            model_concat = {}
            model_smapes = {}

            print(
                f"\n     {'Model':<22s} {'MAE':>9s} {'RMSE':>9s}"
                f" {'SMAPE%':>8s} {'MASE':>8s} {'Infer(s)':>10s}"
            )
            print(f"     {'─' * 70}")

            for name in ALL_MODELS:
                y_pred_all = np.concatenate(preds_store[name])
                # safety: clip wildly divergent forecasts
                y_pred_all = np.clip(y_pred_all, y_true_all.min() - 50, y_true_all.max() + 50)
                model_concat[name] = y_pred_all
                m = compute_metrics(y_true_all, y_pred_all, y_train, sp)
                model_smapes[name] = m["SMAPE"]
                infer_s = infer_times[name]
                rows.append({"dataset": ds_name, "horizon": horizon, "model": name, **m})
                timing_rows.append({
                    "dataset": ds_name, "horizon": horizon, "model": name,
                    "cv_s":    cv_times.get(name, 0.0),
                    "train_s": train_times.get(name, 0.0),
                    "infer_s": infer_s,
                    "total_s": cv_times.get(name, 0.0) + train_times.get(name, 0.0) + infer_s,
                })
                print(
                    f"     {name:<22s} {m['MAE']:9.3f} {m['RMSE']:9.3f}"
                    f" {m['SMAPE']:7.2f}% {m['MASE']:8.4f} {infer_s:10.2f}s"
                )

            # ── Ensembles ──────────────────────────────────────────────────────
            print(f"     {'─' * 70}")
            ENSEMBLE_FNS = [
                ("Ens_InvSMAPE", ens_inv_smape),
                ("Ens_Best",     ens_best),
                ("Ens_Median",   ens_median),
                ("Ens_TrimMean", ens_trim_mean),
            ]
            for ens_name, ens_fn in ENSEMBLE_FNS:
                ens_preds, label = ens_fn(model_concat, model_smapes)
                em = compute_metrics(y_true_all, ens_preds, y_train, sp)
                rows.append({"dataset": ds_name, "horizon": horizon, "model": ens_name, **em})
                timing_rows.append({
                    "dataset": ds_name, "horizon": horizon, "model": ens_name,
                    "cv_s": sum(cv_times.values()),
                    "train_s": sum(train_times.values()),
                    "infer_s": 0.0,
                    "total_s": sum(cv_times.values()) + sum(train_times.values()),
                })
                print(
                    f"     {ens_name:<22s} {em['MAE']:9.3f} {em['RMSE']:9.3f}"
                    f" {em['SMAPE']:7.2f}% {em['MASE']:8.4f}  ({label})"
                )

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df_timing = pd.DataFrame(timing_rows)
    out_dir = ROOT / "outputs" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ett_benchmark_results.csv"
    tim_path = out_dir / "ett_timing.csv"
    df.to_csv(csv_path, index=False)
    df_timing.to_csv(tim_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Timing  saved to {tim_path}")

    print_summary(df)
    print_timing_summary(df_timing)
    return df


# ── Pretty-print ─────────────────────────────────────────────────────────────

def print_summary(df):
    print(f"\n{'=' * 70}")
    print("  SUMMARY — Mean metrics across datasets")
    print(f"{'=' * 70}")

    for h in sorted(df["horizon"].unique()):
        sub = df[df["horizon"] == h]
        agg = sub.groupby("model")[["MAE", "RMSE", "SMAPE", "MASE"]].mean().sort_values("MAE")
        print(f"\n  Horizon = {h}")
        print(f"  {'Model':<22s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE%':>8s} {'MASE':>8s}")
        print(f"  {'─' * 58}")
        for name, row in agg.iterrows():
            tag = " ★" if name.startswith("Ens_") else ""
            print(
                f"  {name:<22s} {row['MAE']:9.3f} {row['RMSE']:9.3f} "
                f"{row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}"
            )

    print(f"\n{'=' * 70}")
    print("  OVERALL RANKING — Mean across all horizons and datasets")
    print(f"{'=' * 70}")
    overall = df.groupby("model")[["MAE", "RMSE", "SMAPE", "MASE"]].mean().sort_values("MAE")
    print(f"  {'Model':<22s} {'MAE':>9s} {'RMSE':>9s} {'SMAPE%':>8s} {'MASE':>8s}")
    print(f"  {'─' * 58}")
    for name, row in overall.iterrows():
        tag = " ★" if name.startswith("Ens_") else ""
        print(
            f"  {name:<22s} {row['MAE']:9.3f} {row['RMSE']:9.3f} "
            f"{row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}"
        )

    print(f"\n{'=' * 70}")
    print("  BEST MODEL PER DATASET × HORIZON (by MAE)")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':<8s} {'H':>4s}  {'Best Model':<22s} {'MAE':>9s} {'SMAPE%':>8s}")
    print(f"  {'─' * 58}")
    for (ds, h), grp in df.groupby(["dataset", "horizon"]):
        best = grp.loc[grp["MAE"].idxmin()]
        print(
            f"  {ds:<8s} {h:4d}  {best['model']:<22s} "
            f"{best['MAE']:9.3f} {best['SMAPE']:7.2f}%"
        )


def print_timing_summary(df_timing: pd.DataFrame):
    """Print per-model timing: CV search, final training, and inference seconds."""
    print(f"\n{'=' * 70}")
    print("  TIMING SUMMARY — Mean across datasets (seconds)")
    print(f"{'=' * 70}")

    for h in sorted(df_timing["horizon"].unique()):
        sub = df_timing[df_timing["horizon"] == h]
        agg = (
            sub.groupby("model")[["cv_s", "train_s", "infer_s", "total_s"]]
            .mean()
            .sort_values("total_s", ascending=False)
        )
        print(f"\n  Horizon = {h}")
        print(
            f"  {'Model':<24s} {'CV(s)':>8s} {'Train(s)':>9s}"
            f" {'Infer(s)':>9s} {'Total(s)':>9s}"
        )
        print(f"  {'─' * 64}")
        for name, row in agg.iterrows():
            print(
                f"  {name:<24s} {row['cv_s']:8.2f} {row['train_s']:9.2f}"
                f" {row['infer_s']:9.2f} {row['total_s']:9.2f}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnergyX ETT Benchmark")
    parser.add_argument("--fast", action="store_true",
                        help="Quick sanity check (2 datasets, horizon=24)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        EnergyX — ETT Forecasting Benchmark            ║")
    print("║        ICLR 2026 TSALM Workshop                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Datasets:    {list(DATASETS.keys())}")
    print(f"  Horizons:    {HORIZONS}")
    print(f"  Split:       {int(TRAIN_RATIO*100)}/{100 - int(TRAIN_RATIO*100)} train/test")
    print(f"  CV folds:    {CV_FOLDS} (expanding window, same as agent)")
    print(f"  Max windows: {MAX_WINDOWS}")
    print(f"  Models:      7 uni + 2 mv + 2 combined + 4 ensembles = 15")

    t_start = time.time()
    results = evaluate(fast=args.fast)
    elapsed = time.time() - t_start
    print(f"\nTotal benchmark time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
