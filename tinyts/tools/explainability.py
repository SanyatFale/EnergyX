"""Explainability computation utilities.
Provides statistical summaries,
STL decomposition, lag contributions, feature importance, SHAP values,
feature correlations, and anomaly explanation helpers.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL


# ---------------------------------------------------------------------------
# Univariate forecast explanation
# ---------------------------------------------------------------------------

def compute_statistical_summary(
    train: pd.Series,
    test: pd.Series,
    periods_per_day: int = 24,
) -> Dict[str, Any]:
    """Historical statistics and recent trend."""
    day_p = periods_per_day
    week_p = periods_per_day * 7

    if len(train) >= day_p * 2:
        recent_trend = (
            "upward"
            if train.iloc[-day_p:].mean() > train.iloc[-day_p * 2 : -day_p].mean()
            else "downward"
        )
    else:
        recent_trend = "stable"

    return {
        "train_mean": float(train.mean()),
        "train_std": float(train.std()),
        "train_min": float(train.min()),
        "train_max": float(train.max()),
        "test_mean": float(test.mean()) if len(test) > 0 else None,
        "test_std": float(test.std()) if len(test) > 0 else None,
        "recent_trend": recent_trend,
        "recent_7d_avg": float(train.iloc[-week_p:].mean()) if len(train) > week_p else float(train.mean()),
        "recent_1d_avg": float(train.iloc[-day_p:].mean()) if len(train) > day_p else float(train.mean()),
    }


def compute_decomposition_metrics(
    series: pd.Series,
    seasonal_period: int = 24,
) -> Optional[Dict[str, Any]]:
    """STL decomposition -> trend/seasonal strength."""
    try:
        if len(series) < 2 * seasonal_period:
            return None
        stl = STL(series, period=seasonal_period, robust=True).fit()
        return {
            "trend_strength": float(
                max(0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
            ),
            "seasonal_strength": float(
                max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
            ),
        }
    except Exception:
        return None


def compute_lag_contributions(
    series: pd.Series,
    lags: List[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Autocorrelation at various lags."""
    if lags is None:
        lags = [1, 6, 12, 24]
    contributions: Dict[str, Dict[str, Any]] = {}
    for lag in lags:
        if len(series) > lag:
            corr = series.autocorr(lag=lag)
            contributions[f"lag_{lag}"] = {
                "correlation": float(corr) if not np.isnan(corr) else 0.0,
                "value": float(series.iloc[-lag]),
            }
    return contributions


def get_model_specific_explanation(model_name: str, context: Dict) -> str:
    """One-liner describing a model's forecasting logic."""
    explanations = {
        "Naive": f"Used last observed value ({context.get('last_value', 0):.2f}) as forecast.",
        "SeasonalNaive": f"Used values from same time {context.get('seasonal_period', 24) // 24} day(s) ago.",
        "ARIMA": "Used autoregressive integrated moving average model fitting historical patterns.",
        "LightGBM": "Used gradient boosting on lag features.",
        "RandomForest": "Used ensemble of decision trees on lag features.",
    }
    return explanations.get(model_name, f"Used {model_name} model.")


# ---------------------------------------------------------------------------
# Multivariate forecast explanation
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: Any,
    feature_cols: List[str],
    model_name: str,
) -> Dict[str, float]:
    """Extract feature importance from tree-based or linear models."""
    try:
        if hasattr(model, "feature_importances_"):
            return {
                col: float(imp)
                for col, imp in zip(feature_cols, model.feature_importances_)
            }
        if hasattr(model, "coef_"):
            return {
                col: float(c) for col, c in zip(feature_cols, model.coef_)
            }
    except Exception:
        pass
    return {}


def compute_shap_values(
    model: Any,
    X_test: np.ndarray,
    feature_cols: List[str],
    model_name: str,
) -> Dict[str, float]:
    """SHAP values for tree-based models (optional dependency)."""
    try:
        import shap

        if model_name in ("RandomForest", "LightGBM", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_test[:min(10, len(X_test))])
            mean_abs = np.abs(sv).mean(axis=0)
            return {col: float(v) for col, v in zip(feature_cols, mean_abs)}
    except ImportError:
        pass
    except Exception:
        pass
    return {}


def compute_feature_correlations(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: List[str],
) -> Dict[str, float]:
    """Pearson correlation between each feature and target."""
    correlations = {}
    for col in feature_cols:
        if col in X.columns:
            corr = X[col].corr(y)
            correlations[col] = float(corr) if not np.isnan(corr) else 0.0
    return correlations


# ---------------------------------------------------------------------------
# Anomaly explanation helpers
# ---------------------------------------------------------------------------

def compute_zscore_explanation(
    series: pd.Series,
    anomaly_indices: pd.Index,
) -> Dict[str, Any]:
    """Z-score context for flagged anomalies."""
    mean = float(series.mean())
    std_val = float(series.std()) + 1e-8

    samples = []
    for idx in anomaly_indices[:5]:
        if idx in series.index:
            val = series.loc[idx]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            val = float(val)
            zscore = (val - mean) / std_val
            samples.append({
                "timestamp": str(idx),
                "value": val,
                "zscore": zscore,
                "deviation_pct": (val - mean) / mean * 100 if mean != 0 else 0,
            })

    return {
        "mean": mean,
        "std": std_val,
        "threshold": 3.0,
        "sample_explanations": samples,
    }


def compute_iqr_explanation(series: pd.Series) -> Dict[str, Any]:
    """IQR-based anomaly bounds."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "expected_range": f"{lower:.2f} - {upper:.2f}",
    }


def compute_stl_anomaly_explanation(
    series: pd.Series,
    seasonal_period: int = 24,
) -> Dict[str, Any]:
    """STL residual stats for anomaly explanation."""
    try:
        if len(series) < 2 * seasonal_period:
            return {}
        stl = STL(series, period=seasonal_period, robust=True).fit()
        resid_std = float(stl.resid.std())
        return {
            "trend_mean": float(stl.trend.mean()),
            "seasonal_amplitude": float(stl.seasonal.max() - stl.seasonal.min()),
            "residual_std": resid_std,
            "anomaly_threshold": 3 * resid_std,
        }
    except Exception:
        return {}


def compute_method_agreement(method_counts: Dict[str, int], n_methods: int) -> Dict[str, Any]:
    """Summarise agreement across anomaly detection methods."""
    total_flags = sum(method_counts.values())
    avg = total_flags / max(n_methods, 1)
    return {
        "method_counts": method_counts,
        "average_agreement": avg,
        "total_methods": n_methods,
    }


def compute_percentile_context(
    series: pd.Series,
    anomaly_values: List[float],
) -> Dict[str, Any]:
    """Percentile ranking for anomalous values."""
    pcts = {}
    for val in anomaly_values[:5]:
        pcts[str(val)] = float(stats.percentileofscore(series, val))
    return {
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "anomaly_percentiles": pcts,
    }
