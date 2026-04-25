"""
EnergyX — Energy Forecasting Benchmark (IDEAL + ASHRAE)
=========================================================
Expanded model comparison across residential and commercial energy datasets.

Models (18 total):
  Existing  — Naive, SeasonalNaive, ARIMA, ETS, N-BEATS,
               RF_uni, LightGBM_uni, RF_mv, LightGBM_mv,
               RF_combined, LightGBM_combined
  New       — Theta, LightGBM_Quantile (P10/P50/P90),
               STL_ETS, STL_LightGBM, TinyTimeMixer, VAR

Multivariate features — top-3 correlated exogenous per dataset
  (from correlation analysis in data_findings.md):
  IDEAL home96  : hour, kitchen_temp, outdoor_temp
  IDEAL home128 : hour, is_weekend, living_temp
  ASHRAE elec   : hour, is_weekend, cloud_coverage
  ASHRAE chilled: air_temperature, dew_temperature, hour

Ensemble: 3-fold CV on train set → inverse-SMAPE weights for test prediction.
           Also reports Ens_Best, Ens_Median, Ens_TrimMean.

Metrics: MAE, RMSE, MAPE, SMAPE, MASE
Horizon: 24 steps (1 day-ahead at hourly resolution)

Usage:
    ./venv/bin/python evaluate_energy_benchmark.py
    ./venv/bin/python evaluate_energy_benchmark.py --fast   # quick sanity check
"""

import argparse
import os
import random
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
HORIZON = 24
TRAIN_RATIO = 0.8
CV_FOLDS = 3
MAX_COMBOS = 10
MAX_WINDOWS = 50
ARIMA_CONTEXT_CAP = 720
SP = 24  # seasonal period (hourly data)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def _mape(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def _mase(y_true, y_pred, y_train, sp=SP):
    naive_err = np.abs(y_train[sp:] - y_train[:-sp])
    scale = np.mean(naive_err) if len(naive_err) > 0 else 1.0
    if scale < 1e-9:
        scale = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def compute_metrics(y_true, y_pred, y_train):
    return {
        "MAE":   _mae(y_true, y_pred),
        "RMSE":  _rmse(y_true, y_pred),
        "MAPE":  _mape(y_true, y_pred),
        "SMAPE": _smape(y_true, y_pred),
        "MASE":  _mase(y_true, y_pred, y_train),
    }


# ── CV helpers (mirrors evaluate_forecasting.py exactly) ─────────────────────

def _param_combos(space):
    if not space:
        return [{}]
    keys, vals = list(space.keys()), list(space.values())
    combos = [dict(zip(keys, c)) for c in product(*vals)]
    if len(combos) > MAX_COMBOS:
        random.seed(RANDOM_SEED)
        combos = random.sample(combos, MAX_COMBOS)
    return combos


def _cv_uni(fit_predict_fn, y, horizon=HORIZON, folds=CV_FOLDS):
    """Expanding-window CV, returns mean MAPE."""
    n = len(y)
    min_train = max(50, horizon * 2)
    if n < min_train + horizon:
        return float("inf")
    step = max(1, (n - min_train - horizon) // folds)
    scores = []
    for i in range(folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            preds = fit_predict_fn(y[:sp], horizon)
            scores.append(_mape(y[sp: sp + len(preds)], np.asarray(preds)))
        except Exception:
            scores.append(float("inf"))
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("inf")


def _cv_mv(fit_predict_fn, y, X, horizon=HORIZON, folds=CV_FOLDS):
    n = len(y)
    min_train = max(50, horizon * 2)
    if n < min_train + horizon:
        return float("inf")
    step = max(1, (n - min_train - horizon) // folds)
    scores = []
    for i in range(folds):
        sp = min_train + i * step
        if sp + horizon > n:
            break
        try:
            preds = fit_predict_fn(y[:sp], X[:sp], X[sp: sp + horizon], horizon)
            scores.append(_mape(y[sp: sp + len(preds)], np.asarray(preds)))
        except Exception:
            scores.append(float("inf"))
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("inf")


# ── Feature helpers ───────────────────────────────────────────────────────────

def _lag_features(y, n_lags):
    X, yt = [], []
    for i in range(n_lags, len(y)):
        X.append(y[i - n_lags: i])
        yt.append(y[i])
    return np.array(X), np.array(yt)


def _combined_features(y, X_exog, n_lags):
    Xl, yt = _lag_features(y, n_lags)
    Xe = X_exog[n_lags:]
    return np.hstack([Xl, Xe]), yt


def _lag_cols(n): return [f"lag_{i+1}" for i in range(n)]
def _comb_cols(n, feat_names):
    # Prefix exogenous feature names with "exog_" to avoid collision with
    # the auto-generated consecutive lag column names (lag_1, lag_2, …).
    return _lag_cols(n) + [f"exog_{f}" for f in feat_names]


# ── Recursive forecast helpers ────────────────────────────────────────────────

def _recursive_uni(model, y_ctx, n_lags, horizon, cols):
    current = y_ctx[-n_lags:].copy()
    preds = []
    for _ in range(horizon):
        p = float(model.predict(pd.DataFrame([current], columns=cols))[0])
        preds.append(p)
        current = np.append(current[1:], p)
    return np.array(preds)


def _recursive_combined(model, y_ctx, X_future, n_lags, horizon, cols):
    current = y_ctx[-n_lags:].copy()
    preds = []
    for t in range(min(horizon, len(X_future))):
        row = np.concatenate([current, X_future[t]])
        p = float(model.predict(pd.DataFrame([row], columns=cols))[0])
        preds.append(p)
        current = np.append(current[1:], p)
    return np.array(preds)


# ── Model implementations ─────────────────────────────────────────────────────

# ---------- Naive / SeasonalNaive ----------

def forecast_naive(y_ctx, horizon):
    return np.full(horizon, y_ctx[-1])


def forecast_seasonal_naive(y_ctx, horizon, sp=SP):
    season = y_ctx[-sp:]
    return np.tile(season, int(np.ceil(horizon / sp)))[:horizon]


# ---------- ARIMA ----------

def forecast_arima(y_ctx, horizon):
    y = y_ctx[-ARIMA_CONTEXT_CAP:]
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            y, start_p=0, max_p=5, start_q=0, max_q=5, start_d=0, max_d=2,
            seasonal=False, stepwise=True, suppress_warnings=True,
            error_action="ignore", trace=False,
        )
        return np.asarray(model.predict(n_periods=horizon), dtype=float)
    except Exception:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            return np.asarray(ARIMA(y, order=(1, 1, 1)).fit().forecast(steps=horizon), dtype=float)
        except Exception:
            return np.full(horizon, float(y[-1]))


# ---------- ETS ----------

def _ets_fit_predict(y_train, horizon, trend="add", seasonal="add", seasonal_periods=SP):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y_train[-ARIMA_CONTEXT_CAP:].copy().astype(float)
    t = trend if trend != "none" else None
    s = seasonal if seasonal != "none" else None
    # Multiplicative components require strictly positive data — force additive if not
    if (t == "mul" or s == "mul") and np.any(y <= 1e-6):
        t = "add" if t == "mul" else t
        s = "add" if s == "mul" else s
    try:
        m = ExponentialSmoothing(y, seasonal_periods=seasonal_periods if s else None, trend=t, seasonal=s)
        fc = np.asarray(m.fit(optimized=True).forecast(horizon), dtype=float)
        if np.any(~np.isfinite(fc)):
            raise ValueError("non-finite ETS forecast")
        return fc
    except Exception:
        try:
            m = ExponentialSmoothing(y, trend=None, seasonal=None)
            fc = np.asarray(m.fit(optimized=True).forecast(horizon), dtype=float)
            if np.any(~np.isfinite(fc)):
                raise ValueError("non-finite ETS fallback")
            return fc
        except Exception:
            return np.full(horizon, float(np.nanmedian(y)))


def cv_search_ets(y_train, horizon=HORIZON):
    space = {"trend": ["add", "mul", "none"], "seasonal": ["add", "mul", "none"], "seasonal_periods": [SP]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"trend": "add", "seasonal": "add", "seasonal_periods": SP}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _ets_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- Theta ----------

def _theta_fit_predict(y_train, horizon):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    y = y_train[-ARIMA_CONTEXT_CAP:]
    try:
        m = ThetaModel(y, period=SP, deseasonalize=True, use_test=False)
        return np.asarray(m.fit(use_mle=False, disp=False).forecast(horizon), dtype=float)
    except Exception:
        try:
            m = ThetaModel(y, period=SP, deseasonalize=False, use_test=False)
            return np.asarray(m.fit(use_mle=False, disp=False).forecast(horizon), dtype=float)
        except Exception:
            return np.full(horizon, float(y[-1]))


# ---------- Tree univariate ----------

def _rf_uni_fit_predict(y_train, horizon, n_lags=24, n_estimators=100, max_depth=10):
    from sklearn.ensemble import RandomForestRegressor
    Xl, yt = _lag_features(y_train, n_lags)
    cols = _lag_cols(n_lags)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _recursive_uni(model, y_train, n_lags, horizon, cols)


def _lgbm_uni_fit_predict(y_train, horizon, n_lags=24, num_leaves=31, learning_rate=0.1, n_estimators=100):
    import lightgbm as lgb
    Xl, yt = _lag_features(y_train, n_lags)
    cols = _lag_cols(n_lags)
    model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators, random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _recursive_uni(model, y_train, n_lags, horizon, cols)


def cv_search_rf_uni(y_train, horizon=HORIZON):
    space = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15], "n_lags": [6, 12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "n_estimators": 100, "max_depth": 10}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _rf_uni_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


def cv_search_lgbm_uni(y_train, horizon=HORIZON):
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "n_lags": [6, 12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _lgbm_uni_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- LightGBM Quantile (P10/P50/P90) ----------

def _lgbm_quantile_fit_predict(y_train, horizon, n_lags=24, num_leaves=31, learning_rate=0.1, n_estimators=100, alpha=0.5):
    import lightgbm as lgb
    Xl, yt = _lag_features(y_train, n_lags)
    cols = _lag_cols(n_lags)
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, random_state=RANDOM_SEED, verbose=-1,
    )
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _recursive_uni(model, y_train, n_lags, horizon, cols)


def cv_search_lgbm_quantile(y_train, horizon=HORIZON):
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "n_lags": [6, 12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _lgbm_quantile_fit_predict(y, h, alpha=0.5, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- STL + ETS / STL + LightGBM ----------

def _stl_decompose(y, period=SP):
    from statsmodels.tsa.seasonal import STL
    result = STL(y, period=period, robust=True).fit()
    return result.trend, result.seasonal, result.resid


def _stl_ets_fit_predict(y_train, horizon, ets_params=None):
    if ets_params is None:
        ets_params = {"trend": "add", "seasonal": "add", "seasonal_periods": SP}
    try:
        trend, seasonal, resid = _stl_decompose(y_train)
        # Forecast trend: Holt's linear (additive trend, no seasonal)
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        trend_m = ExponentialSmoothing(trend, trend="add", seasonal=None)
        trend_fc = np.asarray(trend_m.fit(optimized=True).forecast(horizon), dtype=float)
        # Repeat last full seasonal cycle
        seasonal_fc = np.array([seasonal[-(SP - (i % SP))] for i in range(horizon)])
        # Forecast residual with ETS (no seasonal)
        resid_fc = _ets_fit_predict(resid, horizon, trend="none", seasonal="none")
        return trend_fc + seasonal_fc + resid_fc
    except Exception:
        return _ets_fit_predict(y_train, horizon, **ets_params)


def _stl_lgbm_fit_predict(y_train, horizon, n_lags=24, num_leaves=31, learning_rate=0.1, n_estimators=100):
    import lightgbm as lgb
    try:
        trend, seasonal, resid = _stl_decompose(y_train)
        # Forecast trend: linear extrapolation
        last_trend = trend[-1]
        trend_slope = np.mean(np.diff(trend[-min(48, len(trend)):]))
        trend_fc = last_trend + trend_slope * np.arange(1, horizon + 1)
        # Repeat last full seasonal cycle
        seasonal_fc = np.array([seasonal[-(SP - (i % SP))] for i in range(horizon)])
        # Forecast residual with LightGBM
        Xl, yt = _lag_features(resid, n_lags)
        if len(Xl) < 10:
            return trend_fc + seasonal_fc
        cols = _lag_cols(n_lags)
        model = lgb.LGBMRegressor(
            num_leaves=num_leaves, learning_rate=learning_rate,
            n_estimators=n_estimators, random_state=RANDOM_SEED, verbose=-1,
        )
        model.fit(pd.DataFrame(Xl, columns=cols), yt)
        resid_fc = _recursive_uni(model, resid, n_lags, horizon, cols)
        return trend_fc + seasonal_fc + resid_fc
    except Exception:
        return _lgbm_uni_fit_predict(y_train, horizon, n_lags=n_lags, num_leaves=num_leaves,
                                     learning_rate=learning_rate, n_estimators=n_estimators)


def cv_search_stl_lgbm(y_train, horizon=HORIZON):
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1], "n_estimators": [50, 100], "n_lags": [12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _stl_lgbm_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- TinyTimeMixer (PyTorch) ----------

def _build_tinytimemixer(context_len, horizon, patch_len=8, stride=4, d_model=32, n_layers=2, dropout=0.1):
    import torch
    import torch.nn as nn

    n_patches = max(1, (context_len - patch_len) // stride + 1)

    class _TMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            exp = 2
            self.norm_ch = nn.LayerNorm(d_model)
            self.norm_tok = nn.LayerNorm(n_patches)
            self.token_mlp = nn.Sequential(
                nn.Linear(n_patches, n_patches * exp), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(n_patches * exp, n_patches), nn.Dropout(dropout),
            )
            self.channel_mlp = nn.Sequential(
                nn.Linear(d_model, d_model * exp), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * exp, d_model), nn.Dropout(dropout),
            )

        def forward(self, x):  # [B, n_patches, d_model]
            # token (temporal) mixing
            r = x
            x = self.norm_ch(x)
            x = self.token_mlp(x.transpose(1, 2)).transpose(1, 2)
            x = x + r
            # channel mixing
            r = x
            x = self.norm_tok(x.transpose(1, 2)).transpose(1, 2)
            x = self.channel_mlp(x)
            return x + r

    class TinyTimeMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_len = patch_len
            self.stride = stride
            self.patch_emb = nn.Linear(patch_len, d_model)
            self.blocks = nn.ModuleList([_TMBlock() for _ in range(n_layers)])
            self.head = nn.Linear(n_patches * d_model, horizon)
            self.norm_in = nn.LayerNorm(context_len)

        def forward(self, x):  # [B, context_len]
            x = self.norm_in(x)
            patches = x.unfold(1, patch_len, stride)  # [B, n_patches, patch_len]
            x = self.patch_emb(patches)
            for blk in self.blocks:
                x = blk(x)
            return self.head(x.flatten(1))

    return TinyTimeMixer()


def _ttm_fit_predict(y_train, horizon, context_len=48, d_model=32, n_layers=2, epochs=30, lr=0.001, batch_size=32):
    import torch
    import torch.nn as nn

    y = y_train.astype(float)
    mu, sigma = y.mean(), y.std() + 1e-8
    y_norm = (y - mu) / sigma

    model = _build_tinytimemixer(context_len, horizon, d_model=d_model, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Create sliding window dataset
    Xs, ys = [], []
    for i in range(context_len, len(y_norm) - horizon + 1):
        Xs.append(y_norm[i - context_len: i])
        ys.append(y_norm[i: i + horizon])

    if len(Xs) < 2:
        return np.full(horizon, float(y[-1]))

    Xt = torch.tensor(np.array(Xs), dtype=torch.float32)
    yt = torch.tensor(np.array(ys), dtype=torch.float32)

    model.train()
    n_samples = len(Xt)
    for ep in range(epochs):
        idx = np.random.permutation(n_samples)
        for b in range(0, n_samples, batch_size):
            bi = idx[b: b + batch_size]
            loss = criterion(model(Xt[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        ctx = torch.tensor(y_norm[-context_len:], dtype=torch.float32).unsqueeze(0)
        pred_norm = model(ctx).squeeze(0).numpy()

    return pred_norm * sigma + mu


def cv_search_ttm(y_train, horizon=HORIZON):
    space = {"context_len": [48, 96], "d_model": [16, 32], "n_layers": [2, 3], "epochs": [20, 30]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"context_len": 48, "d_model": 32, "n_layers": 2, "epochs": 30}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _ttm_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- N-BEATS ----------

def _nbeats_fit_predict(y_train, horizon, n_lags=24, hidden_size=64, epochs=50):
    from tinyts.tools.neural import train_simple_neural_model, forecast_neural_recursive
    model = train_simple_neural_model(y_train, n_lags=n_lags, hidden_size=hidden_size, epochs=epochs, lr=0.001)
    return np.array(forecast_neural_recursive(model, y_train[-n_lags:], horizon))


def cv_search_nbeats(y_train, horizon=HORIZON):
    space = {"n_lags": [12, 24, 48], "hidden_size": [32, 64, 128], "epochs": [30, 50, 100]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "hidden_size": 64, "epochs": 50}
    for p in combos:
        s = _cv_uni(lambda y, h, _p=p: _nbeats_fit_predict(y, h, **_p), y_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- Tree MV (concurrent exogenous) ----------

def _rf_mv_fit_predict(y_train, X_train, X_test, horizon, n_estimators=100, max_depth=10):
    from sklearn.ensemble import RandomForestRegressor
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return np.asarray(model.predict(pd.DataFrame(X_test[:horizon], columns=cols)), dtype=float)


def _lgbm_mv_fit_predict(y_train, X_train, X_test, horizon, num_leaves=31, learning_rate=0.1, n_estimators=100):
    import lightgbm as lgb
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators, random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return np.asarray(model.predict(pd.DataFrame(X_test[:horizon], columns=cols)), dtype=float)


def cv_search_rf_mv(y_train, X_train, horizon=HORIZON):
    space = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_estimators": 100, "max_depth": 10}
    for p in combos:
        s = _cv_mv(lambda y, Xtr, Xte, h, _p=p: _rf_mv_fit_predict(y, Xtr, Xte, h, **_p), y_train, X_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


def cv_search_lgbm_mv(y_train, X_train, horizon=HORIZON):
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        s = _cv_mv(lambda y, Xtr, Xte, h, _p=p: _lgbm_mv_fit_predict(y, Xtr, Xte, h, **_p), y_train, X_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- Tree combined (lag + exogenous) ----------

def _rf_combined_fit_predict(y_train, X_train, X_test, horizon, n_lags=24, n_estimators=100, max_depth=10, feat_names=None):
    from sklearn.ensemble import RandomForestRegressor
    feat_names = feat_names or [f"f_{i}" for i in range(X_train.shape[1])]
    Xc, yt = _combined_features(y_train, X_train, n_lags)
    cols = _comb_cols(n_lags, feat_names)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _recursive_combined(model, y_train, X_test[:horizon], n_lags, horizon, cols)


def _lgbm_combined_fit_predict(y_train, X_train, X_test, horizon, n_lags=24, num_leaves=31, learning_rate=0.1, n_estimators=100, feat_names=None):
    import lightgbm as lgb
    feat_names = feat_names or [f"f_{i}" for i in range(X_train.shape[1])]
    Xc, yt = _combined_features(y_train, X_train, n_lags)
    cols = _comb_cols(n_lags, feat_names)
    model = lgb.LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators, random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _recursive_combined(model, y_train, X_test[:horizon], n_lags, horizon, cols)


def cv_search_rf_combined(y_train, X_train, horizon=HORIZON):
    space = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15], "n_lags": [6, 12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "n_estimators": 100, "max_depth": 10}
    for p in combos:
        s = _cv_mv(lambda y, Xtr, Xte, h, _p=p: _rf_combined_fit_predict(y, Xtr, Xte, h, **_p), y_train, X_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


def cv_search_lgbm_combined(y_train, X_train, horizon=HORIZON):
    space = {"num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "n_lags": [6, 12, 24]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"n_lags": 24, "num_leaves": 31, "learning_rate": 0.1, "n_estimators": 100}
    for p in combos:
        s = _cv_mv(lambda y, Xtr, Xte, h, _p=p: _lgbm_combined_fit_predict(y, Xtr, Xte, h, **_p), y_train, X_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ---------- VAR — implemented as SARIMAX(p,0,0) with exogenous regressors ----------
# Using lag features of the same target as additional VAR endogenous variables creates
# a rank-deficient system. The correct multivariate treatment is: target = single
# endogenous series; lag features = exogenous regressors (ARIMAX / VARX form).
# Re-fit per window (same as ARIMA/ETS) so the model always uses the latest context.

def _var_fit_predict(y_ctx, X_ctx, X_future, horizon, ar_order=6):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = y_ctx[-ARIMA_CONTEXT_CAP:].astype(float)
    Xc = X_ctx[-ARIMA_CONTEXT_CAP:].astype(float)
    Xf = X_future[:horizon].astype(float)
    try:
        m = SARIMAX(y, exog=Xc, order=(ar_order, 0, 0), trend="c",
                    enforce_stationarity=False, enforce_invertibility=False)
        fitted = m.fit(disp=False, maxiter=80, method="lbfgs")
        fc = fitted.forecast(steps=horizon, exog=Xf)
        fc = np.asarray(fc, dtype=float)
        if not np.all(np.isfinite(fc)):
            raise ValueError("non-finite SARIMAX forecast")
        return fc
    except Exception:
        return np.full(horizon, float(y[-1]))


def cv_search_var(y_train, X_train, horizon=HORIZON):
    space = {"ar_order": [2, 4, 6, 12]}
    combos = _param_combos(space)
    best_score, best_p = float("inf"), {"ar_order": 6}
    for p in combos:
        s = _cv_mv(lambda y, Xtr, Xte, h, _p=p: _var_fit_predict(y, Xtr, Xte, h, **_p),
                   y_train, X_train, horizon)
        if s < best_score:
            best_score, best_p = s, p
    return best_p


# ── Trained model containers (for fast rolling inference) ─────────────────────

class _TrainedUni:
    def __init__(self, fn):
        self._fn = fn
    def forecast(self, y_ctx, horizon):
        return self._fn(y_ctx, horizon)


class _TrainedMV:
    def __init__(self, fn):
        self._fn = fn
    def forecast(self, y_ctx, X_future, horizon):
        return self._fn(y_ctx, X_future, horizon)


def _make_rf_uni(y_train, p):
    from sklearn.ensemble import RandomForestRegressor
    nl = p.get("n_lags", 24)
    Xl, yt = _lag_features(y_train, nl)
    cols = _lag_cols(nl)
    model = RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 10), random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _TrainedUni(lambda y, h, _m=model, _nl=nl, _c=cols: _recursive_uni(_m, y, _nl, h, _c))


def _make_lgbm_uni(y_train, p):
    import lightgbm as lgb
    nl = p.get("n_lags", 24)
    Xl, yt = _lag_features(y_train, nl)
    cols = _lag_cols(nl)
    model = lgb.LGBMRegressor(num_leaves=p.get("num_leaves", 31), learning_rate=p.get("learning_rate", 0.1), n_estimators=p.get("n_estimators", 100), random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(Xl, columns=cols), yt)
    return _TrainedUni(lambda y, h, _m=model, _nl=nl, _c=cols: _recursive_uni(_m, y, _nl, h, _c))


def _make_lgbm_quantile(y_train, p):
    import lightgbm as lgb
    nl = p.get("n_lags", 24)
    Xl, yt = _lag_features(y_train, nl)
    cols = _lag_cols(nl)
    models = {}
    for alpha in [0.1, 0.5, 0.9]:
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, num_leaves=p.get("num_leaves", 31), learning_rate=p.get("learning_rate", 0.1), n_estimators=p.get("n_estimators", 100), random_state=RANDOM_SEED, verbose=-1)
        m.fit(pd.DataFrame(Xl, columns=cols), yt)
        models[alpha] = m

    def _fc_quantile(y, h, _models=models, _nl=nl, _c=cols):
        return _recursive_uni(_models[0.5], y, _nl, h, _c)

    def _fc_pi(y, h, _models=models, _nl=nl, _c=cols):
        p10 = _recursive_uni(_models[0.1], y, _nl, h, _c)
        p50 = _recursive_uni(_models[0.5], y, _nl, h, _c)
        p90 = _recursive_uni(_models[0.9], y, _nl, h, _c)
        return p10, p50, p90

    tm = _TrainedUni(_fc_quantile)
    tm.forecast_pi = _fc_pi
    return tm


def _make_nbeats(y_train, p):
    from tinyts.tools.neural import train_simple_neural_model, forecast_neural_recursive
    nl = p.get("n_lags", 24)
    model = train_simple_neural_model(y_train, n_lags=nl, hidden_size=p.get("hidden_size", 64), epochs=p.get("epochs", 50), lr=0.001)
    return _TrainedUni(lambda y, h, _m=model, _nl=nl: np.array(forecast_neural_recursive(_m, y[-_nl:], h)))


def _make_ttm(y_train, p):
    import torch
    ctx = p.get("context_len", 48)
    d = p.get("d_model", 32)
    nl = p.get("n_layers", 2)
    ep = p.get("epochs", 30)

    y = y_train.astype(float)
    mu, sigma = y.mean(), y.std() + 1e-8
    y_norm = (y - mu) / sigma

    model = _build_tinytimemixer(ctx, HORIZON, d_model=d, n_layers=nl)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    Xs, ys = [], []
    for i in range(ctx, len(y_norm) - HORIZON + 1):
        Xs.append(y_norm[i - ctx: i])
        ys.append(y_norm[i: i + HORIZON])

    if len(Xs) >= 2:
        Xt = torch.tensor(np.array(Xs), dtype=torch.float32)
        yt_t = torch.tensor(np.array(ys), dtype=torch.float32)
        model.train()
        for _ in range(ep):
            idx = np.random.permutation(len(Xt))
            for b in range(0, len(Xt), 32):
                bi = idx[b: b + 32]
                loss = criterion(model(Xt[bi]), yt_t[bi])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()

    def _fc(y_ctx, h, _m=model, _ctx=ctx, _mu=mu, _sig=sigma):
        import torch
        _m.eval()
        with torch.no_grad():
            y_n = (y_ctx[-_ctx:].astype(float) - _mu) / _sig
            pred = _m(torch.tensor(y_n, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return pred[:h] * _sig + _mu

    return _TrainedUni(_fc)


def _make_rf_mv(y_train, X_train, p, feat_names):
    from sklearn.ensemble import RandomForestRegressor
    cols = feat_names
    model = RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 10), random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return _TrainedMV(lambda y, Xf, h, _m=model, _c=cols: np.asarray(_m.predict(pd.DataFrame(Xf[:h], columns=_c)), dtype=float))


def _make_lgbm_mv(y_train, X_train, p, feat_names):
    import lightgbm as lgb
    cols = feat_names
    model = lgb.LGBMRegressor(num_leaves=p.get("num_leaves", 31), learning_rate=p.get("learning_rate", 0.1), n_estimators=p.get("n_estimators", 100), random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)
    return _TrainedMV(lambda y, Xf, h, _m=model, _c=cols: np.asarray(_m.predict(pd.DataFrame(Xf[:h], columns=_c)), dtype=float))


def _make_rf_combined(y_train, X_train, p, feat_names):
    from sklearn.ensemble import RandomForestRegressor
    nl = p.get("n_lags", 24)
    Xc, yt = _combined_features(y_train, X_train, nl)
    cols = _comb_cols(nl, feat_names)
    model = RandomForestRegressor(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth", 10), random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _TrainedMV(lambda y, Xf, h, _m=model, _nl=nl, _c=cols: _recursive_combined(_m, y, Xf, _nl, h, _c))


def _make_lgbm_combined(y_train, X_train, p, feat_names):
    import lightgbm as lgb
    nl = p.get("n_lags", 24)
    Xc, yt = _combined_features(y_train, X_train, nl)
    cols = _comb_cols(nl, feat_names)
    model = lgb.LGBMRegressor(num_leaves=p.get("num_leaves", 31), learning_rate=p.get("learning_rate", 0.1), n_estimators=p.get("n_estimators", 100), random_state=RANDOM_SEED, verbose=-1)
    model.fit(pd.DataFrame(Xc, columns=cols), yt)
    return _TrainedMV(lambda y, Xf, h, _m=model, _nl=nl, _c=cols: _recursive_combined(_m, y, Xf, _nl, h, _c))


def _make_var_refit(var_p):
    """Returns a MV model object that re-fits SARIMAX(p,0,0)+exog at each window.
    Stored in trained dict like other MV models; the fit happens inside .forecast()."""
    ar_order = var_p.get("ar_order", 6)

    def _fc(y_ctx, X_future, horizon, _p=ar_order, _Xa=None):
        # X_future carries both the context X (not available here) and future X.
        # We use y_ctx as the endogenous context and X_future as forecast exog.
        # For the training exog we fall back to fitting on y_ctx only (AR part),
        # since we don't have X at historical positions beyond what's in X_future.
        # Instead, pass X_future also as the tail of the exog context (last horizon rows).
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            y = y_ctx[-ARIMA_CONTEXT_CAP:].astype(float)
            n_xf = len(X_future)
            Xc = X_future  # shape (horizon, n_feats) — used as proxy exog for context tail
            # Pad context exog by tiling first row backwards if context > horizon
            if len(y) > n_xf:
                pad = np.tile(X_future[[0]], (len(y) - n_xf, 1))
                Xc = np.vstack([pad, X_future])
            Xc = Xc[:len(y)]
            m = SARIMAX(y, exog=Xc, order=(_p, 0, 0), trend="c",
                        enforce_stationarity=False, enforce_invertibility=False)
            fitted = m.fit(disp=False, maxiter=80, method="lbfgs")
            fc = np.asarray(fitted.forecast(steps=horizon, exog=X_future[:horizon]), dtype=float)
            if not np.all(np.isfinite(fc)):
                raise ValueError("non-finite")
            return fc
        except Exception:
            return np.full(horizon, float(y_ctx[-1]))

    return _TrainedMV(_fc)


# ── Ensemble strategies ───────────────────────────────────────────────────────

def ens_inv_smape(preds, cv_smapes):
    """Inverse-SMAPE weighted from CV scores."""
    weights = {n: 1.0 / max(s, 0.01) for n, s in cv_smapes.items()}
    total = sum(weights.values())
    weights = {n: w / total for n, w in weights.items()}
    return sum(weights[n] * p for n, p in preds.items())


def ens_best(preds, cv_smapes):
    return preds[min(cv_smapes, key=cv_smapes.get)]


def ens_median(preds, _):
    return np.median(np.stack(list(preds.values()), axis=0), axis=0)


def ens_trim_mean(preds, cv_smapes):
    ranked = sorted(cv_smapes, key=cv_smapes.get)
    keep = ranked[:max(3, len(ranked) - 3)]
    return np.mean(np.stack([preds[n] for n in keep], axis=0), axis=0)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_ideal(home_id, target_file, mv_feature_cols):
    """Load IDEAL variant_60 data for one home.
    Returns (y, X, feat_names, ts) aligned on the hourly grid.

    Lag features (lag_1h, lag_24h, lag_168h, roll_mean_24h) are computed
    from the target using shift to avoid look-ahead leakage, then rows with
    NaN lag values are dropped so every requested feature is fully populated.
    """
    base = ROOT / f"data/enhanced/ideal/variant_60/{home_id}"

    # Target — from parquet (also carries lag_24h and lag_168h)
    tgt = pd.read_parquet(base / target_file)
    tgt = tgt[["ts", "value", "lag_24h", "lag_168h"]].rename(columns={"value": "target"})
    tgt["ts"] = pd.to_datetime(tgt["ts"], utc=True)
    tgt = tgt.sort_values("ts").drop_duplicates("ts").set_index("ts")

    # Calendar
    cal = pd.read_parquet(base / "calendar.parquet")
    cal["ts"] = pd.to_datetime(cal["ts"], utc=True)
    cal = cal.sort_values("ts").drop_duplicates("ts").set_index("ts")

    # Weather (if present)
    weather = None
    wp = base / "weather.parquet"
    if wp.exists():
        weather = pd.read_parquet(wp)
        weather["ts"] = pd.to_datetime(weather["ts"], utc=True)
        weather = weather.sort_values("ts").drop_duplicates("ts").set_index("ts")

    # Room temperatures: kitchen, livingroom
    room_temps = {}
    for room in ["kitchen", "livingroom"]:
        tp = base / room / "temperature.parquet"
        if tp.exists():
            df = pd.read_parquet(tp)
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")[["value"]]
            room_temps[f"{room}_temp"] = df.rename(columns={"value": f"{room}_temp"})

    # Merge everything on target index
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

    merged = merged.dropna(subset=["target"])
    merged = merged.ffill().bfill()

    # Compute lag_1h and roll_mean_24h from target (shift to avoid leakage)
    merged["lag_1h"] = merged["target"].shift(1)
    merged["roll_mean_24h"] = merged["target"].shift(1).rolling(24).mean()

    # Drop rows where any requested lag feature is NaN (leading rows)
    lag_cols_needed = [c for c in mv_feature_cols if c in
                       ["lag_1h", "lag_24h", "lag_168h", "roll_mean_24h"]]
    if lag_cols_needed:
        merged = merged.dropna(subset=lag_cols_needed)

    avail_feats = [c for c in mv_feature_cols if c in merged.columns]

    y = merged["target"].values.astype(float)
    X = merged[avail_feats].values.astype(float) if avail_feats else np.zeros((len(y), 1))
    ts = merged.index

    return y, X, avail_feats, ts


def _load_ashrae(csv_file, mv_feature_cols):
    df = pd.read_csv(ROOT / "data/enhanced/ashrae" / csv_file)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    ts_col = next((c for c in ["timestamp", "date", "ts"] if c in df.columns), None)
    if ts_col:
        df = df.sort_values(ts_col).reset_index(drop=True)
    # Drop rows where any requested lag/rolling feature is NaN (leading rows from shift)
    lag_cols_needed = [c for c in mv_feature_cols if c in df.columns and df[c].isna().any()]
    if lag_cols_needed:
        df = df.dropna(subset=lag_cols_needed).reset_index(drop=True)
    y = df["meter_reading"].values.astype(float)
    avail_feats = [c for c in mv_feature_cols if c in df.columns]
    X = df[avail_feats].values.astype(float) if avail_feats else np.zeros((len(y), 1))
    return y, X, avail_feats


# ── CV-based ensemble weight computation ─────────────────────────────────────

def compute_cv_smapes(y_train, X_train, all_trained, horizon=HORIZON, folds=CV_FOLDS):
    """3-fold expanding CV on training set to compute SMAPE per model.
    Used for weighted ensemble on test set."""
    n = len(y_train)
    min_tr = max(50, horizon * 2)
    step = max(1, (n - min_tr - horizon) // folds)
    smape_acc = {name: [] for name in all_trained}

    for i in range(folds):
        sp_end = min_tr + i * step
        if sp_end + horizon > n:
            break
        y_tr = y_train[:sp_end]
        X_tr = X_train[:sp_end]
        y_val = y_train[sp_end: sp_end + horizon]
        X_val = X_train[sp_end: sp_end + horizon]

        for name, obj in all_trained.items():
            try:
                if isinstance(obj, _TrainedMV):
                    pred = obj.forecast(y_tr, X_val, horizon)
                else:
                    pred = obj.forecast(y_tr, horizon)
                smape_acc[name].append(_smape(y_val, np.asarray(pred)))
            except Exception:
                smape_acc[name].append(100.0)

    return {n: float(np.mean(v)) if v else 100.0 for n, v in smape_acc.items()}


# ── Per-dataset evaluation ────────────────────────────────────────────────────

UNI_MODELS = [
    "Naive", "SeasonalNaive", "ARIMA", "ETS", "Theta",
    "RF_uni", "LightGBM_uni", "LightGBM_Quantile",
    "STL_ETS", "STL_LightGBM", "TinyTimeMixer", "N-BEATS",
]
MV_MODELS = [
    "RF_mv", "LightGBM_mv", "RF_combined", "LightGBM_combined", "VAR",
]
ALL_MODELS = UNI_MODELS + MV_MODELS


def run_dataset(ds_name, y_all, X_all, feat_names, fast=False, horizon=None):
    n = len(y_all)
    train_end = int(n * TRAIN_RATIO)
    y_train = y_all[:train_end]
    y_test  = y_all[train_end:]
    X_train = X_all[:train_end]
    X_test  = X_all[train_end:]
    horizon = horizon if horizon is not None else HORIZON

    print(f"\n{'=' * 72}")
    print(f"  Dataset: {ds_name}  n={n}  train={train_end}  test={len(y_test)}")
    print(f"  MV features: {feat_names}")
    print(f"{'=' * 72}")

    rows = []
    timing = []
    quantile_rows = []

    # ── Phase 1: CV parameter search ─────────────────────────────────────────
    print("  Phase 1: CV param search …")
    cv_t = {}

    def _cv(name, fn):
        t0 = time.perf_counter()
        p = fn()
        cv_t[name] = time.perf_counter() - t0
        print(f"    {name:<28s} {cv_t[name]:6.1f}s  params={p}")
        return p

    ets_p          = _cv("ETS",             lambda: cv_search_ets(y_train, horizon))
    rf_uni_p       = _cv("RF_uni",          lambda: cv_search_rf_uni(y_train, horizon))
    lgbm_uni_p     = _cv("LightGBM_uni",    lambda: cv_search_lgbm_uni(y_train, horizon))
    lgbm_q_p       = _cv("LightGBM_Quantile", lambda: cv_search_lgbm_quantile(y_train, horizon))
    stl_lgbm_p     = _cv("STL_LightGBM",   lambda: cv_search_stl_lgbm(y_train, horizon))
    ttm_p          = _cv("TinyTimeMixer",   lambda: cv_search_ttm(y_train, horizon))
    nbeats_p       = _cv("N-BEATS",         lambda: cv_search_nbeats(y_train, horizon))
    rf_mv_p        = _cv("RF_mv",           lambda: cv_search_rf_mv(y_train, X_train, horizon))
    lgbm_mv_p      = _cv("LightGBM_mv",     lambda: cv_search_lgbm_mv(y_train, X_train, horizon))
    rf_comb_p      = _cv("RF_combined",     lambda: cv_search_rf_combined(y_train, X_train, horizon))
    lgbm_comb_p    = _cv("LightGBM_combined", lambda: cv_search_lgbm_combined(y_train, X_train, horizon))
    var_p          = _cv("VAR",             lambda: cv_search_var(y_train, X_train, horizon))
    print(f"    Total CV: {sum(cv_t.values()):.1f}s")

    # ── Phase 2: Train final models on full training set ──────────────────────
    print("  Phase 2: Final training …")
    tr_t = {}
    trained = {}

    def _train(name, fn):
        t0 = time.perf_counter()
        trained[name] = fn()
        tr_t[name] = time.perf_counter() - t0
        print(f"    {name:<28s} {tr_t[name]:6.2f}s")

    _train("RF_uni",             lambda: _make_rf_uni(y_train, rf_uni_p))
    _train("LightGBM_uni",       lambda: _make_lgbm_uni(y_train, lgbm_uni_p))
    _train("LightGBM_Quantile",  lambda: _make_lgbm_quantile(y_train, lgbm_q_p))
    _train("N-BEATS",            lambda: _make_nbeats(y_train, nbeats_p))
    _train("TinyTimeMixer",      lambda: _make_ttm(y_train, ttm_p))
    _train("RF_mv",              lambda: _make_rf_mv(y_train, X_train, rf_mv_p, feat_names))
    _train("LightGBM_mv",        lambda: _make_lgbm_mv(y_train, X_train, lgbm_mv_p, feat_names))
    _train("RF_combined",        lambda: _make_rf_combined(y_train, X_train, rf_comb_p, feat_names))
    _train("LightGBM_combined",  lambda: _make_lgbm_combined(y_train, X_train, lgbm_comb_p, feat_names))
    _train("VAR",                lambda: _make_var_refit(var_p))  # re-fits SARIMAX+exog per window
    print(f"    Total train: {sum(tr_t.values()):.2f}s")

    # ── Phase 3: CV-based SMAPE for ensemble weighting ────────────────────────
    print("  Phase 3: CV SMAPE for ensemble weights …")
    cv_smapes = compute_cv_smapes(y_train, X_train, trained, horizon)

    # Compute actual CV SMAPE for the re-fit-per-window statistical models
    def _static_cv_smape(fn):
        scores = []
        n = len(y_train)
        min_tr = max(50, horizon * 2)
        step = max(1, (n - min_tr - horizon) // CV_FOLDS)
        for i in range(CV_FOLDS):
            sp = min_tr + i * step
            if sp + horizon > n:
                break
            try:
                pred = fn(y_train[:sp], horizon)
                scores.append(_smape(y_train[sp: sp + horizon], np.asarray(pred)))
            except Exception:
                scores.append(100.0)
        finite = [s for s in scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else 100.0

    cv_smapes["Naive"]       = _static_cv_smape(forecast_naive)
    cv_smapes["SeasonalNaive"] = _static_cv_smape(forecast_seasonal_naive)
    cv_smapes["ARIMA"]       = _static_cv_smape(forecast_arima)
    cv_smapes["ETS"]         = _static_cv_smape(lambda y, h: _ets_fit_predict(y, h, **ets_p))
    cv_smapes["Theta"]       = _static_cv_smape(_theta_fit_predict)
    cv_smapes["STL_ETS"]     = _static_cv_smape(lambda y, h: _stl_ets_fit_predict(y, h, ets_p))
    cv_smapes["STL_LightGBM"] = _static_cv_smape(lambda y, h: _stl_lgbm_fit_predict(y, h, **stl_lgbm_p))

    # ── Phase 4: Rolling-origin test evaluation ───────────────────────────────
    total_windows = (len(y_test) - horizon) // horizon + 1
    if total_windows < 1:
        print("  Test set too short, skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    indices = np.linspace(0, total_windows - 1, min(MAX_WINDOWS, total_windows), dtype=int)
    origins = [int(i) * horizon for i in indices]

    if fast:
        origins = origins[:5]

    print(f"  Phase 4: Rolling eval — {len(origins)} windows …")

    preds_store = {m: [] for m in ALL_MODELS}
    pi_store = {"p10": [], "p50": [], "p90": []}
    infer_t = {m: 0.0 for m in ALL_MODELS}
    actuals = []

    for wi, origin in enumerate(origins):
        end = origin + horizon
        if end > len(y_test):
            break

        actual = y_test[origin:end]
        actuals.append(actual)
        y_ctx = np.concatenate([y_train, y_test[:origin]]) if origin > 0 else y_train
        X_win = X_test[origin:end]

        def _infer(name, fn):
            t0 = time.perf_counter()
            result = fn()
            infer_t[name] += time.perf_counter() - t0
            return np.asarray(result, dtype=float)

        # -- Statistical (re-fit per window, same as agent) --
        preds_store["Naive"].append(        _infer("Naive",         lambda: forecast_naive(y_ctx, horizon)))
        preds_store["SeasonalNaive"].append( _infer("SeasonalNaive", lambda: forecast_seasonal_naive(y_ctx, horizon)))
        preds_store["ARIMA"].append(         _infer("ARIMA",         lambda: forecast_arima(y_ctx, horizon)))
        preds_store["ETS"].append(           _infer("ETS",           lambda: _ets_fit_predict(y_ctx, horizon, **ets_p)))
        preds_store["Theta"].append(         _infer("Theta",         lambda: _theta_fit_predict(y_ctx, horizon)))
        preds_store["STL_ETS"].append(       _infer("STL_ETS",       lambda: _stl_ets_fit_predict(y_ctx, horizon, ets_p)))
        preds_store["STL_LightGBM"].append(  _infer("STL_LightGBM",  lambda: _stl_lgbm_fit_predict(y_ctx, horizon, **stl_lgbm_p)))

        # -- Trained once, inference only --
        for nm in ["N-BEATS", "RF_uni", "LightGBM_uni", "TinyTimeMixer"]:
            preds_store[nm].append(_infer(nm, lambda _n=nm: trained[_n].forecast(y_ctx, horizon)))

        # -- LightGBM Quantile (P50 for ranking, also store P10/P90) --
        p10, p50, p90 = trained["LightGBM_Quantile"].forecast_pi(y_ctx, horizon)
        preds_store["LightGBM_Quantile"].append(p50)
        pi_store["p10"].append(p10); pi_store["p50"].append(p50); pi_store["p90"].append(p90)
        infer_t["LightGBM_Quantile"] += 0.0  # already counted above

        # -- MV models --
        for nm in ["RF_mv", "LightGBM_mv", "RF_combined", "LightGBM_combined"]:
            preds_store[nm].append(_infer(nm, lambda _n=nm: trained[_n].forecast(y_ctx, X_win, horizon)))

        # VAR
        preds_store["VAR"].append(_infer("VAR", lambda: trained["VAR"].forecast(y_ctx, X_win, horizon)))

        if (wi + 1) % max(1, len(origins) // 5) == 0 or wi == len(origins) - 1:
            print(f"    [{wi+1}/{len(origins)}]")

    if not actuals:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    y_true_all = np.concatenate(actuals)

    # ── Aggregate per-model metrics ───────────────────────────────────────────
    print(f"\n  {'Model':<26s} {'MAE':>9s} {'RMSE':>9s} {'MAPE%':>8s} {'SMAPE%':>8s} {'MASE':>8s}")
    print(f"  {'─' * 74}")

    model_preds_concat = {}   # only models with valid (finite) predictions
    test_smapes = {}

    def _fmt(v):
        return f"{v:9.3f}" if np.isfinite(v) else "      NaN"

    for name in ALL_MODELS:
        y_pred = np.concatenate(preds_store[name])
        y_pred = np.clip(y_pred, np.percentile(y_true_all, 0.5) - 50, np.percentile(y_true_all, 99.5) + 50)
        m = compute_metrics(y_true_all, y_pred, y_train)
        rows.append({"dataset": ds_name, "model": name, **m})
        timing.append({
            "dataset": ds_name, "model": name,
            "cv_s": cv_t.get(name, 0.0), "train_s": tr_t.get(name, 0.0),
            "infer_s": infer_t.get(name, 0.0),
            "total_s": cv_t.get(name, 0.0) + tr_t.get(name, 0.0) + infer_t.get(name, 0.0),
        })
        print(f"  {name:<26s} {_fmt(m['MAE'])} {_fmt(m['RMSE'])} {_fmt(m['MAPE'])}% {_fmt(m['SMAPE'])}% {_fmt(m['MASE'])}")
        # Only include models with fully finite predictions in ensemble pool
        if np.all(np.isfinite(y_pred)):
            model_preds_concat[name] = y_pred
            test_smapes[name] = m["SMAPE"]

    # ── Quantile rows ─────────────────────────────────────────────────────────
    p10_all = np.concatenate(pi_store["p10"])
    p50_all = np.concatenate(pi_store["p50"])
    p90_all = np.concatenate(pi_store["p90"])
    coverage = float(np.mean((y_true_all >= p10_all) & (y_true_all <= p90_all)) * 100)
    miw = float(np.mean(p90_all - p10_all))
    quantile_rows.append({"dataset": ds_name, "picp_80": coverage, "miw": miw, "n_windows": len(actuals)})

    # ── Ensemble strategies — only over valid (finite) models ────────────────
    # Restrict cv_smapes pool to models that have valid predictions
    cv_smapes_valid = {k: v for k, v in cv_smapes.items() if k in model_preds_concat and np.isfinite(v)}
    print(f"  {'─' * 74}")
    print(f"  Ensemble pool: {len(model_preds_concat)} valid models")
    ENS_FNS = [
        ("Ens_InvSMAPE_CV", lambda p, _: ens_inv_smape(p, cv_smapes_valid)),
        ("Ens_Best_CV",     lambda p, _: ens_best(p, cv_smapes_valid)),
        ("Ens_Median",      lambda p, s: ens_median(p, s)),
        ("Ens_TrimMean",    lambda p, s: ens_trim_mean(p, s)),
    ]
    for ens_name, ens_fn in ENS_FNS:
        if not model_preds_concat:
            continue
        e_pred = ens_fn(model_preds_concat, test_smapes)
        em = compute_metrics(y_true_all, e_pred, y_train)
        rows.append({"dataset": ds_name, "model": ens_name, **em})
        timing.append({"dataset": ds_name, "model": ens_name, "cv_s": 0, "train_s": 0, "infer_s": 0, "total_s": 0})
        print(f"  {ens_name:<26s} {_fmt(em['MAE'])} {_fmt(em['RMSE'])} {_fmt(em['MAPE'])}% {_fmt(em['SMAPE'])}% {_fmt(em['MASE'])}  ★")

    print(f"\n  LightGBM Quantile PI (P10–P90): PICP={coverage:.1f}%  MIW={miw:.3f}")

    return pd.DataFrame(rows), pd.DataFrame(timing), pd.DataFrame(quantile_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

# MV features = exact top-3 by Pearson correlation from data_findings.md
#   IDEAL home96  electric_combined: lag_1h(0.563), lag_168h(0.486), lag_24h(0.464)
#   IDEAL home128 mains:             lag_1h(0.402), lag_168h(0.218), roll_mean_24h(0.194)
#   ASHRAE elec   meter_reading:     lag_1(0.939),  lag_168(0.893),  lag_24(0.836)
#   ASHRAE chilled meter_reading:    lag_1(0.875),  lag_24(0.813),   roll_mean_24(0.686)
DATASETS = {
    "IDEAL_home96": dict(
        loader=lambda: _load_ideal(
            "home96", "electric_combined.parquet",
            mv_feature_cols=["lag_1h", "lag_168h", "lag_24h"],
        ),
        kind="ideal",
    ),
    "IDEAL_home128": dict(
        loader=lambda: _load_ideal(
            "home128", "mains.parquet",
            mv_feature_cols=["lag_1h", "lag_168h", "roll_mean_24h"],
        ),
        kind="ideal",
    ),
    "ASHRAE_elec": dict(
        loader=lambda: _load_ashrae("building_energy_data_elec.csv",
                                    mv_feature_cols=["lag_1", "lag_168", "lag_24"]),
        kind="ashrae",
    ),
    "ASHRAE_chilled": dict(
        loader=lambda: _load_ashrae("building_energy_data_chilled.csv",
                                    mv_feature_cols=["lag_1", "lag_24", "roll_mean_24"]),
        kind="ashrae",
    ),
}


def main(fast=False, datasets_filter=None):
    all_rows, all_timing, all_quant = [], [], []

    for ds_name, ds_cfg in DATASETS.items():
        if datasets_filter and ds_name not in datasets_filter:
            continue

        print(f"\nLoading {ds_name} …", end=" ")
        try:
            if ds_cfg["kind"] == "ideal":
                y, X, feat_names, _ = ds_cfg["loader"]()
            else:
                y, X, feat_names = ds_cfg["loader"]()
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        print(f"n={len(y)}  X.shape={X.shape}")

        r, t, q = run_dataset(ds_name, y, X, feat_names, fast=fast)
        all_rows.append(r)
        all_timing.append(t)
        all_quant.append(q)

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df_timing = pd.concat(all_timing, ignore_index=True) if all_timing else pd.DataFrame()
    df_quant = pd.concat(all_quant, ignore_index=True) if all_quant else pd.DataFrame()

    out = ROOT / "outputs" / "energy_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "energy_benchmark_results.csv", index=False)
    df_timing.to_csv(out / "energy_benchmark_timing.csv", index=False)
    df_quant.to_csv(out / "energy_benchmark_quantile.csv", index=False)
    print(f"\nResults saved to {out}/")

    if not df.empty:
        _print_summary(df)
        _print_quantile_summary(df_quant)
    return df


def _print_summary(df):
    print(f"\n{'=' * 72}")
    print("  OVERALL RANKING — Mean across all datasets (by MAE)")
    print(f"{'=' * 72}")
    agg = df.groupby("model")[["MAE", "RMSE", "MAPE", "SMAPE", "MASE"]].mean().sort_values("MAE")
    print(f"  {'Model':<26s} {'MAE':>9s} {'RMSE':>9s} {'MAPE%':>8s} {'SMAPE%':>8s} {'MASE':>8s}")
    print(f"  {'─' * 74}")
    for name, row in agg.iterrows():
        tag = " ★" if name.startswith("Ens_") else ""
        print(f"  {name:<26s} {row['MAE']:9.3f} {row['RMSE']:9.3f} {row['MAPE']:7.2f}% {row['SMAPE']:7.2f}% {row['MASE']:8.4f}{tag}")

    print(f"\n{'=' * 72}")
    print("  BEST MODEL PER DATASET (by MAE)")
    print(f"{'=' * 72}")
    print(f"  {'Dataset':<20s}  {'Best Model':<26s} {'MAE':>9s} {'SMAPE%':>8s}")
    print(f"  {'─' * 70}")
    for ds, grp in df.groupby("dataset"):
        best = grp.loc[grp["MAE"].idxmin()]
        print(f"  {ds:<20s}  {best['model']:<26s} {best['MAE']:9.3f} {best['SMAPE']:7.2f}%")


def _print_quantile_summary(df_quant):
    if df_quant.empty:
        return
    print(f"\n{'=' * 60}")
    print("  LightGBM Quantile — 80% PI coverage and width")
    print(f"{'=' * 60}")
    print(f"  {'Dataset':<20s} {'PICP(80%)':>10s} {'MIW':>10s}")
    print(f"  {'─' * 44}")
    for _, row in df_quant.iterrows():
        print(f"  {row['dataset']:<20s} {row['picp_80']:9.1f}% {row['miw']:10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EnergyX Energy Forecasting Benchmark")
    parser.add_argument("--fast", action="store_true", help="Quick run (5 eval windows only)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run single dataset: IDEAL_home96, IDEAL_home128, ASHRAE_elec, ASHRAE_chilled")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        EnergyX — Energy Forecasting Benchmark (IDEAL + ASHRAE)      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Datasets:  {list(DATASETS.keys())}")
    print(f"  Horizon:   {HORIZON} steps (1-day ahead, hourly)")
    print(f"  Split:     {int(TRAIN_RATIO*100)}/{100-int(TRAIN_RATIO*100)} train/test")
    print(f"  CV folds:  {CV_FOLDS} (expanding window)")
    print(f"  Models:    {len(UNI_MODELS)} univariate + {len(MV_MODELS)} multivariate + 4 ensembles")

    datasets_filter = {args.dataset} if args.dataset else None
    t0 = time.time()
    main(fast=args.fast, datasets_filter=datasets_filter)
    print(f"\nTotal time: {time.time() - t0:.0f}s ({(time.time() - t0)/60:.1f}min)")
