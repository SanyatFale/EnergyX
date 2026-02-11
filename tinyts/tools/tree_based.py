"""Tree-based forecasting models as LangChain tools."""

import json
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from langchain.tools import tool
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def create_lagged_features(y: np.ndarray, n_lags: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Create lagged features for supervised learning.

    Args:
        y: Time series data
        n_lags: Number of lags to create

    Returns:
        X: Feature matrix (lagged values)
        y_target: Target values
    """
    X, y_target = [], []

    for i in range(n_lags, len(y)):
        X.append(y[i-n_lags:i])
        y_target.append(y[i])

    return np.array(X), np.array(y_target)


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


@tool
def train_random_forest_forecaster(
    train_data: str,
    horizon: int,
    n_lags: int = 12,
    n_estimators: int = 100,
    max_depth: int = 10,
    **kwargs
) -> str:
    """Train a Random Forest forecaster.
    
    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        n_lags: Number of lagged features (default: 12)
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum tree depth (default: 10)
        
    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    
    try:
        X, y = create_lagged_features(y_train, n_lags)
        cols = _lag_col_names(n_lags)
        X_df = pd.DataFrame(X, columns=cols)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_df, y)

        last_values = y_train[-n_lags:]
        predictions = forecast_recursive(model, last_values, horizon, col_names=cols)

        result = {
            "model_name": "RandomForest",
            "predictions": predictions,
            "metadata": {
                "n_lags": n_lags,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "feature_importance": model.feature_importances_.tolist(),
            }
        }
    except Exception as e:
        result = {
            "model_name": "RandomForest",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {
                "error": str(e),
                "fallback": "naive",
            }
        }

    return json.dumps(result)


@tool
def train_lightgbm_forecaster(
    train_data: str,
    horizon: int,
    n_lags: int = 12,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    **kwargs
) -> str:
    """Train a LightGBM forecaster.

    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        n_lags: Number of lagged features (default: 12)
        num_leaves: Number of leaves (default: 31)
        learning_rate: Learning rate (default: 0.1)
        n_estimators: Number of boosting rounds (default: 100)

    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))

    try:
        X, y = create_lagged_features(y_train, n_lags)
        cols = _lag_col_names(n_lags)
        X_df = pd.DataFrame(X, columns=cols)

        model = lgb.LGBMRegressor(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_df, y)

        last_values = y_train[-n_lags:]
        predictions = forecast_recursive(model, last_values, horizon, col_names=cols)

        result = {
            "model_name": "LightGBM",
            "predictions": predictions,
            "metadata": {
                "n_lags": n_lags,
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "feature_importance": model.feature_importances_.tolist(),
            }
        }
    except Exception as e:
        result = {
            "model_name": "LightGBM",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {
                "error": str(e),
                "fallback": "naive",
            }
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

