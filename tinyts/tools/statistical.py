"""Statistical forecasting models as LangChain tools."""

import json
import warnings
from typing import Any, Dict, List

import numpy as np
from langchain.tools import tool
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning


@tool
def train_naive_forecaster(
    train_data: str,
    horizon: int,
    **kwargs
) -> str:
    """Train a naive forecaster (last value persistence).
    
    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        
    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    
    # Naive forecast: repeat last value
    last_value = y_train[-1]
    predictions = [last_value] * horizon
    
    result = {
        "model_name": "Naive",
        "predictions": predictions,
        "metadata": {
            "last_value": float(last_value),
            "horizon": horizon,
        }
    }
    
    return json.dumps(result)


@tool
def train_seasonal_naive_forecaster(
    train_data: str,
    horizon: int,
    seasonal_period: int = 12,
    **kwargs
) -> str:
    """Train a seasonal naive forecaster.
    
    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        seasonal_period: Seasonal period (default: 12)
        
    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))
    
    # Seasonal naive: repeat last seasonal cycle
    predictions = []
    for i in range(horizon):
        idx = -(seasonal_period - (i % seasonal_period))
        if abs(idx) <= len(y_train):
            predictions.append(float(y_train[idx]))
        else:
            predictions.append(float(y_train[-1]))
    
    result = {
        "model_name": "SeasonalNaive",
        "predictions": predictions,
        "metadata": {
            "seasonal_period": seasonal_period,
            "horizon": horizon,
        }
    }
    
    return json.dumps(result)


@tool
def train_arima_forecaster(
    train_data: str,
    horizon: int,
    p: int = 1,
    d: int = 1,
    q: int = 1,
    **kwargs
) -> str:
    """Train an ARIMA forecaster using auto_arima for optimal order selection.

    Uses pmdarima's auto_arima to automatically select best (p,d,q) order
    by minimizing AIC. Falls back to manual ARIMA(p,d,q) if auto_arima fails.
    The p, d, q parameters are ignored when auto_arima succeeds.

    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        p: AR order for manual fallback (default: 1)
        d: Differencing order for manual fallback (default: 1)
        q: MA order for manual fallback (default: 1)

    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)

            # Try auto_arima first
            try:
                import pmdarima as pm
                auto_model = pm.auto_arima(
                    y_train,
                    start_p=0, max_p=2,
                    start_q=0, max_q=2,
                    start_d=0, max_d=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                fitted = auto_model
                order = auto_model.order
                p_fit, d_fit, q_fit = order
                forecast = fitted.predict(n_periods=horizon)
                predictions = forecast.tolist()

                result = {
                    "model_name": f"ARIMA({p_fit},{d_fit},{q_fit})",
                    "predictions": predictions,
                    "metadata": {
                        "aic": float(fitted.aic()),
                        "p": p_fit,
                        "d": d_fit,
                        "q": q_fit,
                        "horizon": horizon,
                        "method": "auto_arima",
                    }
                }
            except Exception:
                # Fallback to manual ARIMA with provided (p,d,q)
                model = ARIMA(y_train, order=(p, d, q))
                fitted = model.fit()
                forecast = fitted.forecast(steps=horizon)
                predictions = forecast.tolist()

                result = {
                    "model_name": f"ARIMA({p},{d},{q})",
                    "predictions": predictions,
                    "metadata": {
                        "aic": float(fitted.aic),
                        "bic": float(fitted.bic),
                        "p": p,
                        "d": d,
                        "q": q,
                        "horizon": horizon,
                        "method": "manual",
                    }
                }
    except Exception as e:
        result = {
            "model_name": f"ARIMA({p},{d},{q})",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {
                "error": str(e),
                "fallback": "naive",
            }
        }

    return json.dumps(result)


@tool
def train_exponential_smoothing(
    train_data: str,
    horizon: int,
    seasonal_periods: int = 12,
    trend: str = "add",
    seasonal: str = "add",
    **kwargs
) -> str:
    """Train an Exponential Smoothing (ETS) forecaster.
    
    Args:
        train_data: JSON string of training data (list of floats)
        horizon: Forecast horizon
        seasonal_periods: Seasonal period (default: 12)
        trend: Trend component ('add', 'mul', or None)
        seasonal: Seasonal component ('add', 'mul', or None)
        
    Returns:
        JSON string with predictions and metadata
    """
    y_train = np.array(json.loads(train_data))

    try:
        # Fit ETS model with warning suppression
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            warnings.filterwarnings('ignore', category=ConvergenceWarning)

            model = ExponentialSmoothing(
                y_train,
                seasonal_periods=seasonal_periods,
                trend=trend if trend != "none" else None,
                seasonal=seasonal if seasonal != "none" else None,
            )
            fitted = model.fit()
        
        # Generate forecasts
        forecast = fitted.forecast(steps=horizon)
        predictions = forecast.tolist()
        
        result = {
            "model_name": f"ETS(trend={trend},seasonal={seasonal})",
            "predictions": predictions,
            "metadata": {
                "aic": float(fitted.aic),
                "seasonal_periods": seasonal_periods,
                "trend": trend,
                "seasonal": seasonal,
            }
        }
    except Exception as e:
        result = {
            "model_name": f"ETS(trend={trend},seasonal={seasonal})",
            "predictions": [float(y_train[-1])] * horizon,
            "metadata": {
                "error": str(e),
                "fallback": "naive",
            }
        }
    
    return json.dumps(result)

