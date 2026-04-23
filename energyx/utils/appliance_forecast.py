"""Per-appliance forecasting utility.

Trains a lightweight ARIMA + RandomForest ensemble on a single appliance's
power time series from the IDEAL dataset, with rolling-origin cross-validation.

Used by the Monitor tab appliance forecast section.
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def forecast_appliance(
    appl_df: pd.DataFrame,
    sensor_id: str,
    horizon: int = 24,
    cv_folds: int = 3,
) -> Dict[str, Any]:
    """Train ARIMA + RF ensemble on one appliance; return forecast + metadata.

    Args:
        appl_df:   DataFrame with columns [ts, sensor_id, value, ...] filtered
                   to sensor_type == 'appliance_power'.
        sensor_id: Target appliance sensor ID.
        horizon:   Steps to forecast (each step = 1 data frequency unit).
        cv_folds:  Rolling-origin folds for MAPE estimation.

    Returns:
        {
            "status":         "ok" | "error",
            "sensor_id":      str,
            "predictions":    [{"step": int, "value": float}, ...],
            "ensemble_mape":  str,
            "best_model":     str,
            "n_train":        int,
            "frequency":      str,
            "error":          str  (only if status == "error"),
        }
    """
    try:
        series = _prepare_series(appl_df, sensor_id)
    except Exception as e:
        return {"status": "error", "sensor_id": sensor_id, "error": str(e)}

    if len(series) < max(20, horizon * 2):
        return {
            "status": "error",
            "sensor_id": sensor_id,
            "error": f"Not enough data: {len(series)} rows (need ≥{max(20, horizon*2)}).",
        }

    freq = _infer_freq(series)
    values = series.values.astype(float)

    # ── ARIMA forecast ────────────────────────────────────────────────────────
    arima_preds, arima_mape = _arima_forecast(values, horizon, cv_folds)

    # ── RandomForest forecast ─────────────────────────────────────────────────
    rf_preds, rf_mape = _rf_forecast(values, horizon, cv_folds)

    # ── Ensemble (inverse-MAPE weights) ───────────────────────────────────────
    arima_w, rf_w, best = _ensemble_weights(arima_mape, rf_mape)
    ensemble = [
        arima_w * a + rf_w * r
        for a, r in zip(arima_preds, rf_preds)
    ]
    ens_mape = arima_w * arima_mape + rf_w * rf_mape

    predictions = [
        {"step": i + 1, "value": round(max(0.0, v), 2)}
        for i, v in enumerate(ensemble)
    ]

    return {
        "status":        "ok",
        "sensor_id":     sensor_id,
        "predictions":   predictions,
        "ensemble_mape": f"{ens_mape:.3f}",
        "best_model":    best,
        "n_train":       len(values),
        "frequency":     freq,
        "arima_mape":    f"{arima_mape:.3f}",
        "rf_mape":       f"{rf_mape:.3f}",
    }


# ── helpers ────────────────────────────────────────────────────────────────────

def _prepare_series(appl_df: pd.DataFrame, sensor_id: str) -> pd.Series:
    sub = appl_df[appl_df["sensor_id"] == sensor_id][["ts", "value"]].copy()
    if sub.empty:
        raise ValueError(f"No rows for sensor_id={sensor_id}")
    sub["ts"] = pd.to_datetime(sub["ts"])
    sub = sub.set_index("ts").sort_index()["value"]
    # Resample to 1-hour mean (smooth noisy appliance data)
    return sub.resample("1h").mean().fillna(method="ffill").fillna(0)


def _infer_freq(series: pd.Series) -> str:
    if series.index.freq is not None:
        return str(series.index.freq)
    if len(series) >= 2:
        delta = (series.index[1] - series.index[0]).total_seconds() / 60
        return f"~{delta:.0f}min"
    return "unknown"


def _rolling_origin_splits(values: np.ndarray, horizon: int, n_folds: int):
    """Yield (train, test) index pairs using rolling-origin CV."""
    min_train = max(20, 2 * horizon)
    n = len(values)
    step = max(1, (n - min_train - horizon) // n_folds)
    splits = []
    for k in range(n_folds):
        train_end = min_train + k * step
        test_end  = train_end + horizon
        if test_end > n:
            break
        splits.append((train_end, test_end))
    return splits


def _arima_forecast(
    values: np.ndarray, horizon: int, cv_folds: int
) -> tuple[list, float]:
    """ARIMA(1,1,1) forecast; fallback to naive if statsmodels unavailable."""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        splits = _rolling_origin_splits(values, horizon, cv_folds)
        mape_scores = []
        for train_end, test_end in splits:
            train = values[:train_end]
            test  = values[train_end:test_end]
            try:
                model = ARIMA(train, order=(1, 1, 1)).fit()
                preds = model.forecast(steps=len(test))
                mask = test > 0
                if mask.any():
                    mape_scores.append(
                        np.mean(np.abs((test[mask] - preds[mask]) / test[mask]))
                    )
            except Exception:
                pass

        mape = float(np.mean(mape_scores)) if mape_scores else 0.5

        # Final model on full series
        model_final = ARIMA(values, order=(1, 1, 1)).fit()
        preds_final = list(model_final.forecast(steps=horizon))
        return [max(0.0, float(p)) for p in preds_final], mape

    except ImportError:
        # Naive seasonal fallback (repeat last 24h)
        period = min(24, len(values))
        tail = values[-period:]
        preds = [float(tail[i % period]) for i in range(horizon)]
        return preds, 0.5


def _rf_forecast(
    values: np.ndarray, horizon: int, cv_folds: int
) -> tuple[list, float]:
    """RandomForest with lag features; fallback to naive."""
    try:
        from sklearn.ensemble import RandomForestRegressor

        lags = min(24, len(values) // 4)

        def _make_features(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            X, y = [], []
            for i in range(lags, len(arr)):
                X.append(arr[i - lags:i])
                y.append(arr[i])
            return np.array(X), np.array(y)

        splits = _rolling_origin_splits(values, horizon, cv_folds)
        mape_scores = []
        for train_end, test_end in splits:
            train_vals = values[:train_end]
            test_vals  = values[train_end:test_end]
            X_tr, y_tr = _make_features(train_vals)
            if len(X_tr) < 5:
                continue
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_tr, y_tr)
            # Recursive multi-step
            buf = list(train_vals[-lags:])
            preds = []
            for _ in range(len(test_vals)):
                p = float(rf.predict([buf[-lags:]])[0])
                preds.append(max(0.0, p))
                buf.append(p)
            preds_arr = np.array(preds)
            mask = test_vals > 0
            if mask.any():
                mape_scores.append(
                    np.mean(np.abs((test_vals[mask] - preds_arr[mask]) / test_vals[mask]))
                )

        mape = float(np.mean(mape_scores)) if mape_scores else 0.5

        # Final model
        X_all, y_all = _make_features(values)
        rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_final.fit(X_all, y_all)
        buf = list(values[-lags:])
        preds_final = []
        for _ in range(horizon):
            p = float(rf_final.predict([buf[-lags:]])[0])
            preds_final.append(max(0.0, p))
            buf.append(p)
        return preds_final, mape

    except ImportError:
        period = min(24, len(values))
        tail = values[-period:]
        preds = [float(tail[i % period]) for i in range(horizon)]
        return preds, 0.5


def _ensemble_weights(
    arima_mape: float, rf_mape: float
) -> tuple[float, float, str]:
    """Inverse-MAPE ensemble weights (same as TinyTSAgent)."""
    eps = 1e-6
    inv_a = 1.0 / (arima_mape + eps)
    inv_r = 1.0 / (rf_mape + eps)
    total = inv_a + inv_r
    w_a = inv_a / total
    w_r = inv_r / total
    best = "ARIMA" if arima_mape <= rf_mape else "RandomForest"
    return w_a, w_r, best
