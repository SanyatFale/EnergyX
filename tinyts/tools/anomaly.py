"""Anomaly detection tools - 7-method ensemble.

Ported from OwnSolarCast anomaly.py. Includes:
Z-score, Modified Z-score (MAD), Rolling statistics, IQR,
STL decomposition, Isolation Forest, DBSCAN, and ensemble voting.
"""

import json
from typing import List

import numpy as np
from langchain.tools import tool
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


@tool
def run_isolation_forest(
    data: str,
    contamination: float = 0.02,
    n_estimators: int = 100,
    **kwargs
) -> str:
    """Run Isolation Forest for anomaly detection.

    Args:
        data: JSON string of data (list of floats)
        contamination: Expected proportion of outliers (default: 0.02)
        n_estimators: Number of trees (default: 100)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data)).reshape(-1, 1)

    try:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        labels = model.fit_predict(y_scaled)
        scores = model.score_samples(y_scaled)
        anomaly_scores = -scores

        result = {
            "model_name": "IsolationForest",
            "anomaly_labels": labels.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "n_anomalies": int((labels == -1).sum()),
            "metadata": {"contamination": contamination, "n_estimators": n_estimators},
        }
    except Exception as e:
        result = {
            "model_name": "IsolationForest",
            "anomaly_labels": [1] * len(y),
            "anomaly_scores": [0.0] * len(y),
            "n_anomalies": 0,
            "metadata": {"error": str(e)},
        }

    return json.dumps(result)


@tool
def run_statistical_anomaly_detection(
    data: str,
    threshold: float = 3.0,
    **kwargs
) -> str:
    """Run Z-score based anomaly detection.

    Args:
        data: JSON string of data (list of floats)
        threshold: Z-score threshold (default: 3.0)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data))
    mean = np.mean(y)
    std = np.std(y)

    if std == 0:
        z_scores = np.zeros_like(y)
    else:
        z_scores = np.abs((y - mean) / std)

    labels = np.where(z_scores > threshold, -1, 1)

    result = {
        "model_name": "ZScore",
        "anomaly_labels": labels.tolist(),
        "anomaly_scores": z_scores.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "metadata": {"threshold": threshold, "mean": float(mean), "std": float(std)},
    }
    return json.dumps(result)


@tool
def run_modified_zscore(
    data: str,
    threshold: float = 3.5,
    **kwargs
) -> str:
    """Run Modified Z-score (MAD-based) anomaly detection.

    More robust to outliers than standard Z-score.
    Uses Median Absolute Deviation instead of standard deviation.

    Args:
        data: JSON string of data (list of floats)
        threshold: Modified Z-score threshold (default: 3.5)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data))
    median = np.median(y)
    mad = np.median(np.abs(y - median))

    if mad == 0:
        modified_z = np.zeros_like(y)
    else:
        modified_z = 0.6745 * np.abs(y - median) / mad

    # Adaptive threshold: if fixed threshold flags >5% of data,
    # use the 97.5th percentile of scores as threshold instead
    max_contamination = 0.05
    labels = np.where(modified_z > threshold, -1, 1)
    contamination_rate = (labels == -1).sum() / len(y)

    if contamination_rate > max_contamination and len(y) > 0:
        adaptive_threshold = float(np.percentile(modified_z, (1 - max_contamination) * 100))
        # Ensure adaptive threshold is at least the original
        threshold = max(threshold, adaptive_threshold)
        labels = np.where(modified_z > threshold, -1, 1)

    result = {
        "model_name": "ModifiedZScore",
        "anomaly_labels": labels.tolist(),
        "anomaly_scores": modified_z.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "metadata": {"threshold": threshold, "median": float(median), "mad": float(mad),
                      "adaptive": contamination_rate > max_contamination},
    }
    return json.dumps(result)


@tool
def run_rolling_anomaly(
    data: str,
    window: int = 24,
    sigma_k: float = 3.0,
    **kwargs
) -> str:
    """Run rolling statistics anomaly detection.

    Flags points outside rolling_mean +/- k * rolling_std.

    Args:
        data: JSON string of data (list of floats)
        window: Rolling window size (default: 24)
        sigma_k: Number of standard deviations (default: 3.0)

    Returns:
        JSON string with anomaly scores and labels
    """
    import pandas as pd

    y = np.array(json.loads(data))
    s = pd.Series(y)

    rolling_mean = s.rolling(window=window, min_periods=1).mean()
    rolling_std = s.rolling(window=window, min_periods=1).std().fillna(0)

    deviation = np.abs(y - rolling_mean.values)
    threshold_values = sigma_k * rolling_std.values
    threshold_values = np.where(threshold_values == 0, np.inf, threshold_values)

    scores = deviation / threshold_values
    labels = np.where(scores > 1.0, -1, 1)

    result = {
        "model_name": "RollingStats",
        "anomaly_labels": labels.tolist(),
        "anomaly_scores": scores.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "metadata": {"window": window, "sigma_k": sigma_k},
    }
    return json.dumps(result)


@tool
def run_iqr_anomaly(
    data: str,
    k: float = 1.5,
    **kwargs
) -> str:
    """Run IQR-based anomaly detection.

    Args:
        data: JSON string of data (list of floats)
        k: IQR multiplier (default: 1.5)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data))
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    labels = np.where((y < lower) | (y > upper), -1, 1)

    # Score: distance from bounds normalized by IQR
    if iqr == 0:
        scores = np.zeros_like(y, dtype=float)
    else:
        scores = np.clip(np.maximum(lower - y, y - upper), 0, None) / iqr

    result = {
        "model_name": "IQR",
        "anomaly_labels": labels.tolist(),
        "anomaly_scores": scores.tolist(),
        "n_anomalies": int((labels == -1).sum()),
        "metadata": {"k": k, "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                      "lower_bound": float(lower), "upper_bound": float(upper)},
    }
    return json.dumps(result)


@tool
def run_stl_anomaly(
    data: str,
    period: int = 24,
    threshold_sigma: float = 3.0,
    **kwargs
) -> str:
    """Run STL decomposition-based anomaly detection.

    Decomposes into trend + seasonal + residual.
    Flags points where |residual| > threshold * std(residual).

    Args:
        data: JSON string of data (list of floats)
        period: Seasonal period (default: 24)
        threshold_sigma: Sigma threshold for residuals (default: 3.0)

    Returns:
        JSON string with anomaly scores and labels
    """
    from statsmodels.tsa.seasonal import STL

    y = np.array(json.loads(data))

    try:
        if len(y) < 2 * period:
            # Not enough data for STL
            result = {
                "model_name": "STL",
                "anomaly_labels": [1] * len(y),
                "anomaly_scores": [0.0] * len(y),
                "n_anomalies": 0,
                "metadata": {"error": "insufficient data for STL"},
            }
            return json.dumps(result)

        stl = STL(y, period=period, robust=True)
        decomposition = stl.fit()
        residuals = decomposition.resid

        residual_std = np.std(residuals)
        if residual_std == 0:
            scores = np.zeros_like(residuals)
        else:
            scores = np.abs(residuals) / residual_std

        labels = np.where(scores > threshold_sigma, -1, 1)

        result = {
            "model_name": "STL",
            "anomaly_labels": labels.tolist(),
            "anomaly_scores": scores.tolist(),
            "n_anomalies": int((labels == -1).sum()),
            "metadata": {"period": period, "residual_std": float(residual_std)},
        }
    except Exception as e:
        result = {
            "model_name": "STL",
            "anomaly_labels": [1] * len(y),
            "anomaly_scores": [0.0] * len(y),
            "n_anomalies": 0,
            "metadata": {"error": str(e)},
        }

    return json.dumps(result)


@tool
def run_dbscan_anomaly(
    data: str,
    eps: float = 0.5,
    min_samples: int = 5,
    **kwargs
) -> str:
    """Run DBSCAN density-based anomaly detection.

    Points labeled as noise (-1) by DBSCAN are treated as anomalies.

    Args:
        data: JSON string of data (list of floats)
        eps: DBSCAN eps parameter (default: 0.5)
        min_samples: DBSCAN min_samples (default: 5)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data)).reshape(-1, 1)

    try:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)

        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(y_scaled)

        # Noise points (label=-1) are anomalies
        labels = np.where(cluster_labels == -1, -1, 1)
        # Score: 1 for anomalies, 0 for normal
        scores = np.where(cluster_labels == -1, 1.0, 0.0)

        result = {
            "model_name": "DBSCAN",
            "anomaly_labels": labels.tolist(),
            "anomaly_scores": scores.tolist(),
            "n_anomalies": int((labels == -1).sum()),
            "metadata": {"eps": eps, "min_samples": min_samples,
                          "n_clusters": int(len(set(cluster_labels) - {-1}))},
        }
    except Exception as e:
        result = {
            "model_name": "DBSCAN",
            "anomaly_labels": [1] * len(y),
            "anomaly_scores": [0.0] * len(y),
            "n_anomalies": 0,
            "metadata": {"error": str(e)},
        }

    return json.dumps(result)


@tool
def run_anomaly_ensemble(
    data: str,
    min_votes: int = 3,
    **kwargs
) -> str:
    """Run all 7 anomaly detection methods and return majority-voted results.

    Ported from OwnSolarCast anomaly.py ensemble voting pattern.
    A point is flagged as anomaly if >= min_votes methods agree.

    Args:
        data: JSON string of data (list of floats)
        min_votes: Minimum methods that must agree (default: 3 of 7)

    Returns:
        JSON string with ensemble results, per-method counts, and voted labels
    """
    # Run all 7 methods
    methods = [
        ("ZScore", run_statistical_anomaly_detection),
        ("ModifiedZScore", run_modified_zscore),
        ("RollingStats", run_rolling_anomaly),
        ("IQR", run_iqr_anomaly),
        ("STL", run_stl_anomaly),
        ("IsolationForest", run_isolation_forest),
        ("DBSCAN", run_dbscan_anomaly),
    ]

    all_labels = []
    method_counts = {}

    for name, method_tool in methods:
        try:
            result_json = method_tool.invoke({"data": data})
            result = json.loads(result_json)
            labels = np.array(result["anomaly_labels"])
            all_labels.append(labels)
            method_counts[name] = int((labels == -1).sum())
        except Exception as e:
            y_len = len(json.loads(data))
            all_labels.append(np.ones(y_len))
            method_counts[name] = 0

    # Ensemble voting
    all_labels_array = np.array(all_labels)
    anomaly_votes = (all_labels_array == -1).sum(axis=0)
    ensemble_labels = np.where(anomaly_votes >= min_votes, -1, 1)

    # Compute per-anomaly-point agreement: how many methods voted for each anomaly
    anomaly_point_votes = anomaly_votes[ensemble_labels == -1]
    avg_agreement = float(np.mean(anomaly_point_votes)) if len(anomaly_point_votes) > 0 else 0.0
    # Safety: agreement can never exceed number of methods
    avg_agreement = min(avg_agreement, float(len(methods)))

    result = {
        "model_name": "Ensemble",
        "anomaly_labels": ensemble_labels.tolist(),
        "anomaly_scores": anomaly_votes.tolist(),
        "n_anomalies": int((ensemble_labels == -1).sum()),
        "method_counts": method_counts,
        "metadata": {
            "min_votes": min_votes,
            "n_methods": len(methods),
            "average_agreement": avg_agreement,
        },
    }
    return json.dumps(result)
