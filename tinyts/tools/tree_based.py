"""Tree-based forecasting models as LangChain tools."""

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain.tools import tool
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Sparse lag indices (in minutes at 1-minute resolution)
# ---------------------------------------------------------------------------
# Instead of a dense window of 720 consecutive lags, we pick 12 specific
# "jump" offsets that cover short-term momentum and the key daily anchors.
# Each column is just y[t - lag] — a single array index, essentially free.
#
#   1–5 min  : very short-term autocorrelation / transient state
#   15–30 min: appliance usage cycles
#   60 min   : hourly pattern
#   120 min  : 2-hour pattern (common for heating/cooling cycles)
#   360 min  : 6-hour pattern
#   720 min  : 12-hour anchor  (same time yesterday morning / night)
#   1440 min : 24-hour anchor  (same time yesterday — strongest seasonality)
DEFAULT_LAG_INDICES: List[int] = [1, 2, 3, 5, 15, 30, 60, 120, 360, 720, 1440]

# Fourier periods: daily (1440 min) and weekly (10080 min)
_FOURIER_PERIODS: Tuple[int, ...] = (1440, 10080)


def create_lagged_features_sparse(
    y: np.ndarray,
    lag_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a feature matrix using specific lag offsets — not a dense window.

    For each sample i (starting from max(lag_indices)), each column is
    y[i - lag] for lag in lag_indices.  This is a set of O(1) numpy slice
    operations — far cheaper and more memory-efficient than a dense
    (N × max_lag) sliding-window matrix.

    Args:
        y:           1-D time series (1-minute resolution).
        lag_indices: Lag offsets in minutes.  Defaults to DEFAULT_LAG_INDICES.

    Returns:
        X:        (n - max_lag, len(lag_indices)) feature matrix
        y_target: (n - max_lag,) target vector aligned with X
    """
    if lag_indices is None:
        lag_indices = DEFAULT_LAG_INDICES
    y = np.asarray(y, dtype=float)
    max_lag = max(lag_indices)
    n = len(y)
    n_samples = n - max_lag
    # Column k = y[max_lag - lag_k : max_lag - lag_k + n_samples]
    # Pure numpy slicing — one O(1) view per lag, zero copies needed.
    X = np.column_stack([y[max_lag - lag: max_lag - lag + n_samples] for lag in lag_indices])
    y_target = y[max_lag:]
    return X, y_target


def add_fourier_features(
    X: np.ndarray,
    n_samples: int,
    t_offset: int = 0,
    periods: Tuple[int, ...] = _FOURIER_PERIODS,
) -> np.ndarray:
    """Append sin/cos Fourier terms to a feature matrix.

    Each period T contributes two columns: sin(2π·t/T) and cos(2π·t/T),
    where t = t_offset … t_offset + n_samples - 1.
    Using a sin+cos pair (not just sin) avoids phase ambiguity — together
    they uniquely encode any position within the cycle.

    t_offset should be the absolute minute-index of the first row in X so
    that the phase aligns correctly between training and recursive forecast.

    Args:
        X:         (n_samples, n_features) existing feature matrix.
        n_samples: Number of rows.
        t_offset:  Absolute minute index of the first row.
        periods:   Periodicities in minutes.

    Returns:
        (n_samples, n_features + 2·len(periods)) augmented matrix.
    """
    t = np.arange(t_offset, t_offset + n_samples, dtype=float)
    fourier_cols = []
    for period in periods:
        fourier_cols.append(np.sin(2.0 * np.pi * t / period))
        fourier_cols.append(np.cos(2.0 * np.pi * t / period))
    return np.hstack([X, np.column_stack(fourier_cols)])


def lag_col_names_sparse(lag_indices: Optional[List[int]] = None) -> List[str]:
    """Column names for a sparse lag feature matrix, e.g. 'lag_720m'."""
    if lag_indices is None:
        lag_indices = DEFAULT_LAG_INDICES
    return [f"lag_{lag}m" for lag in lag_indices]


def fourier_col_names(periods: Tuple[int, ...] = _FOURIER_PERIODS) -> List[str]:
    """Column names for Fourier features, e.g. 'sin_24h', 'cos_24h'."""
    cols: List[str] = []
    for p in periods:
        label = f"{p // 60}h" if p % 60 == 0 else f"{p}m"
        cols.extend([f"sin_{label}", f"cos_{label}"])
    return cols


def all_feature_col_names(
    lag_indices: Optional[List[int]] = None,
    periods: Tuple[int, ...] = _FOURIER_PERIODS,
) -> List[str]:
    """Combined lag + Fourier column names (matches the cache column order)."""
    return lag_col_names_sparse(lag_indices) + fourier_col_names(periods)


def forecast_recursive_sparse(
    model,
    recent_history: np.ndarray,
    horizon: int,
    lag_indices: List[int],
    col_names: Optional[List[str]] = None,
    t_start: int = 0,
    periods: Tuple[int, ...] = _FOURIER_PERIODS,
) -> List[float]:
    """Recursive forecast using sparse lags + Fourier features.

    Maintains a ring buffer of max(lag_indices) recent values so any lag
    index is an O(1) lookup at each step.  Fourier terms are computed from
    the absolute time index, keeping phase consistent with training.

    Args:
        model:          Fitted sklearn / LightGBM model.
        recent_history: At least max(lag_indices) most-recent observed values.
        horizon:        Number of steps to forecast.
        lag_indices:    Sparse lag offsets used during training.
        col_names:      Full column names (lag + Fourier) — passed to model.
        t_start:        Absolute minute index of the first forecast step.
                        = max(lag_indices) + n_training_samples
        periods:        Fourier periods matching training.
    """
    max_lag = max(lag_indices)
    history = list(recent_history[-max_lag:])

    preds: List[float] = []
    for step in range(horizon):
        lag_feats = [history[-lag] for lag in lag_indices]
        row = np.array([lag_feats])

        if periods:
            t = float(t_start + step)
            fourier_feats = []
            for period in periods:
                fourier_feats.extend([
                    np.sin(2.0 * np.pi * t / period),
                    np.cos(2.0 * np.pi * t / period),
                ])
            row = np.hstack([row, np.array([fourier_feats])])

        if col_names is not None:
            row = pd.DataFrame(row, columns=col_names)

        pred = float(model.predict(row)[0])
        preds.append(pred)
        history.append(pred)

    return preds


# ---------------------------------------------------------------------------
# Legacy dense-window helpers (kept for @tool fallback path compatibility)
# ---------------------------------------------------------------------------

def create_lagged_features(y: np.ndarray, n_lags: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Dense lag matrix via stride tricks (legacy — prefer create_lagged_features_sparse)."""
    from numpy.lib.stride_tricks import sliding_window_view
    y = np.asarray(y, dtype=float)
    X = sliding_window_view(y, n_lags)[:-1].copy()
    y_target = y[n_lags:]
    return X, y_target


def _lag_col_names(n_lags: int) -> List[str]:
    return [f"lag_{i+1}" for i in range(n_lags)]


def forecast_recursive(model, last_values: np.ndarray, horizon: int, col_names: List[str] = None) -> List[float]:
    """Generate recursive forecasts.

    Args:
        model: Trained model with predict method
        last_values: Last n_lags values from training data
        horizon: Forecast horizon
        col_names: Feature column names (avoids sklearn warning)

    Returns:
        List of predictions
    """
    predictions = []
    current = last_values.copy()

    for _ in range(horizon):
        row = pd.DataFrame([current], columns=col_names) if col_names else current.reshape(1, -1)
        pred = model.predict(row)[0]
        predictions.append(float(pred))
        current = np.append(current[1:], pred)

    return predictions


def _fit_rf_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    last_window: np.ndarray,
    horizon: int,
    params: dict,
    lag_indices: Optional[List[int]] = None,
) -> Tuple[List[float], dict]:
    """Fit a Random Forest on a pre-computed feature matrix (no JSON overhead).

    When lag_indices is provided the matrix is assumed to contain sparse lags
    followed by Fourier columns (as produced by create_lagged_features_sparse +
    add_fourier_features).  forecast_recursive_sparse is then used so that the
    Fourier phase stays consistent across training and recursive forecast steps.

    Args:
        X_train:     (n_samples, n_features) feature matrix.
        y_train:     (n_samples,) target values.
        last_window: At least max(lag_indices) most-recent raw values for
                     recursive forecasting.
        horizon:     Steps to forecast.
        params:      Hyperparameter dict (n_estimators, max_depth).
        lag_indices: Sparse lag offsets (None → legacy dense path).

    Returns:
        (predictions, metadata_dict)
    """
    use_sparse = lag_indices is not None
    cols = all_feature_col_names(lag_indices) if use_sparse else _lag_col_names(X_train.shape[1])
    model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 50),
        max_depth=params.get("max_depth", 10),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)

    if use_sparse:
        # t_start: first forecast step's absolute minute index
        t_start = max(lag_indices) + len(X_train)
        preds = forecast_recursive_sparse(model, last_window, horizon, lag_indices,
                                          col_names=cols, t_start=t_start)
    else:
        preds = forecast_recursive(model, last_window, horizon, col_names=cols)

    meta = {
        "lag_indices": lag_indices if use_sparse else list(range(1, X_train.shape[1] + 1)),
        "n_estimators": params.get("n_estimators", 50),
        "max_depth": params.get("max_depth", 10),
        "feature_importance": model.feature_importances_.tolist(),
    }
    return preds, meta


def _fit_lgbm_numpy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    last_window: np.ndarray,
    horizon: int,
    params: dict,
    lag_indices: Optional[List[int]] = None,
) -> Tuple[List[float], dict]:
    """Fit LightGBM on a pre-computed feature matrix (no JSON overhead).

    When lag_indices is provided the matrix is assumed to contain sparse lags
    followed by Fourier columns.  See _fit_rf_numpy for full argument docs.
    """
    use_sparse = lag_indices is not None
    cols = all_feature_col_names(lag_indices) if use_sparse else _lag_col_names(X_train.shape[1])
    model = lgb.LGBMRegressor(
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.1),
        n_estimators=params.get("n_estimators", 50),
        random_state=42,
        verbose=-1,
    )
    model.fit(pd.DataFrame(X_train, columns=cols), y_train)

    if use_sparse:
        t_start = max(lag_indices) + len(X_train)
        preds = forecast_recursive_sparse(model, last_window, horizon, lag_indices,
                                          col_names=cols, t_start=t_start)
    else:
        preds = forecast_recursive(model, last_window, horizon, col_names=cols)

    meta = {
        "lag_indices": lag_indices if use_sparse else list(range(1, X_train.shape[1] + 1)),
        "num_leaves": params.get("num_leaves", 31),
        "learning_rate": params.get("learning_rate", 0.1),
        "n_estimators": params.get("n_estimators", 50),
        "feature_importance": model.feature_importances_.tolist(),
    }
    return preds, meta


@tool
def train_random_forest_forecaster(
    train_data: str,
    horizon: int,
    n_estimators: int = 100,
    max_depth: int = 10,
    **kwargs
) -> str:
    """Train a Random Forest forecaster using sparse lags + Fourier features.

    Features: DEFAULT_LAG_INDICES sparse lags (1–1440 min) plus sin/cos
    Fourier terms for daily (1440 min) and weekly (10080 min) seasonality.

    Args:
        train_data:   JSON string of training data (list of floats)
        horizon:      Forecast horizon
        n_estimators: Number of trees (default: 100)
        max_depth:    Maximum tree depth (default: 10)

    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    lag_indices = DEFAULT_LAG_INDICES
    max_lag = max(lag_indices)
    try:
        X, y = create_lagged_features_sparse(y_train, lag_indices)
        X = add_fourier_features(X, len(X), t_offset=max_lag)
        params = {"n_estimators": n_estimators, "max_depth": max_depth}
        preds, meta = _fit_rf_numpy(X, y, y_train[-max_lag:], horizon, params, lag_indices)
        result = {"model_name": "RandomForest", "predictions": preds, "metadata": meta}
    except Exception as e:
        result = {
            "model_name": "RandomForest",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {"error": str(e), "fallback": "naive"},
        }
    return json.dumps(result)


@tool
def train_lightgbm_forecaster(
    train_data: str,
    horizon: int,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    **kwargs
) -> str:
    """Train a LightGBM forecaster using sparse lags + Fourier features.

    Features: DEFAULT_LAG_INDICES sparse lags (1–1440 min) plus sin/cos
    Fourier terms for daily (1440 min) and weekly (10080 min) seasonality.

    Args:
        train_data:    JSON string of training data (list of floats)
        horizon:       Forecast horizon
        num_leaves:    Number of leaves (default: 31)
        learning_rate: Learning rate (default: 0.1)
        n_estimators:  Number of boosting rounds (default: 100)

    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    lag_indices = DEFAULT_LAG_INDICES
    max_lag = max(lag_indices)
    try:
        X, y = create_lagged_features_sparse(y_train, lag_indices)
        X = add_fourier_features(X, len(X), t_offset=max_lag)
        params = {"num_leaves": num_leaves, "learning_rate": learning_rate, "n_estimators": n_estimators}
        preds, meta = _fit_lgbm_numpy(X, y, y_train[-max_lag:], horizon, params, lag_indices)
        result = {"model_name": "LightGBM", "predictions": preds, "metadata": meta}
    except Exception as e:
        result = {
            "model_name": "LightGBM",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {"error": str(e), "fallback": "naive"},
        }
    return json.dumps(result)


@tool
def train_random_forest_multivariate(
    train_features: str,
    train_target: str,
    test_features: str,
    horizon: int,
    n_estimators: int = 100,
    max_depth: int = 10,
    **kwargs,
) -> str:
    """Train Random Forest with explicit feature matrix (multivariate).

    Args:
        train_features: JSON string of 2D feature matrix (list of lists)
        train_target: JSON string of target values (list of floats)
        test_features: JSON string of 2D test feature matrix
        horizon: Forecast horizon
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum tree depth (default: 10)

    Returns:
        JSON string with predictions, feature_importance, and metadata
    """
    X_train = np.array(json.loads(train_features))
    y_train = np.array(json.loads(train_target))
    X_test = np.array(json.loads(test_features))
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)

    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_df, y_train)

        predictions = model.predict(X_test_df.iloc[:horizon]).tolist()
        fi = model.feature_importances_.tolist()

        result = {
            "model_name": "RandomForest",
            "predictions": predictions,
            "metadata": {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "feature_importance": fi,
                "multivariate": True,
            },
        }
    except Exception as e:
        result = {
            "model_name": "RandomForest",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {"error": str(e), "fallback": "naive", "multivariate": True},
        }

    return json.dumps(result)


@tool
def train_lightgbm_multivariate(
    train_features: str,
    train_target: str,
    test_features: str,
    horizon: int,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    **kwargs,
) -> str:
    """Train LightGBM with explicit feature matrix (multivariate).

    Args:
        train_features: JSON string of 2D feature matrix (list of lists)
        train_target: JSON string of target values (list of floats)
        test_features: JSON string of 2D test feature matrix
        horizon: Forecast horizon
        num_leaves: Number of leaves (default: 31)
        learning_rate: Learning rate (default: 0.1)
        n_estimators: Number of boosting rounds (default: 100)

    Returns:
        JSON string with predictions, feature_importance, and metadata
    """
    X_train = np.array(json.loads(train_features))
    y_train = np.array(json.loads(train_target))
    X_test = np.array(json.loads(test_features))
    cols = [f"f_{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=cols)
    X_test_df = pd.DataFrame(X_test, columns=cols)

    try:
        model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train_df, y_train)

        predictions = model.predict(X_test_df.iloc[:horizon]).tolist()
        fi = model.feature_importances_.tolist()

        result = {
            "model_name": "LightGBM",
            "predictions": predictions,
            "metadata": {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "feature_importance": fi,
                "multivariate": True,
            },
        }
    except Exception as e:
        result = {
            "model_name": "LightGBM",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {"error": str(e), "fallback": "naive", "multivariate": True},
        }

    return json.dumps(result)

