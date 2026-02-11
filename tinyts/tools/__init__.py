"""LangChain tools for time series models."""

from tinyts.tools.statistical import (
    train_naive_forecaster,
    train_seasonal_naive_forecaster,
    train_arima_forecaster,
    train_exponential_smoothing,
)
from tinyts.tools.tree_based import (
    train_random_forest_forecaster,
    train_lightgbm_forecaster,
    train_random_forest_multivariate,
    train_lightgbm_multivariate,
)
from tinyts.tools.neural import (
    train_tiny_time_mixer,
    train_nbeats,
)
from tinyts.tools.anomaly import (
    run_isolation_forest,
    run_statistical_anomaly_detection,
    run_modified_zscore,
    run_rolling_anomaly,
    run_iqr_anomaly,
    run_stl_anomaly,
    run_dbscan_anomaly,
    run_anomaly_ensemble,
)
from tinyts.tools.ensemble import (
    compute_ensemble_weights,
)

__all__ = [
    # Statistical
    "train_naive_forecaster",
    "train_seasonal_naive_forecaster",
    "train_arima_forecaster",
    "train_exponential_smoothing",
    # Tree-based (univariate)
    "train_random_forest_forecaster",
    "train_lightgbm_forecaster",
    # Tree-based (multivariate)
    "train_random_forest_multivariate",
    "train_lightgbm_multivariate",
    # Neural
    "train_tiny_time_mixer",
    "train_nbeats",
    # Anomaly detection (7-method ensemble)
    "run_isolation_forest",
    "run_statistical_anomaly_detection",
    "run_modified_zscore",
    "run_rolling_anomaly",
    "run_iqr_anomaly",
    "run_stl_anomaly",
    "run_dbscan_anomaly",
    "run_anomaly_ensemble",
    # Ensemble weights
    "compute_ensemble_weights",
]
