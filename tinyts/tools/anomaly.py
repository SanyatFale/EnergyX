"""Anomaly detection tools - 7-method ensemble.

anomaly.py. Includes:
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
        data: JSON string of data — list of floats (univariate) or
              list of lists (multivariate, each row = [target, feat1, ...])
        contamination: Expected proportion of outliers (default: 0.02)
        n_estimators: Number of trees (default: 100)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data))
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]

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
            "metadata": {"contamination": contamination, "n_estimators": n_estimators,
                          "n_features": int(y.shape[1])},
        }
    except Exception as e:
        result = {
            "model_name": "IsolationForest",
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
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
                      "adaptive": bool(contamination_rate > max_contamination)},
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
        data: JSON string of data — list of floats (univariate) or
              list of lists (multivariate, each row = [target, feat1, ...])
        eps: DBSCAN eps parameter (default: 0.5)
        min_samples: DBSCAN min_samples (default: 5)

    Returns:
        JSON string with anomaly scores and labels
    """
    y = np.array(json.loads(data))
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]

    _DBSCAN_MAX_N = 50_000  # O(n²) memory — subsample above this

    try:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y)

        if n <= _DBSCAN_MAX_N:
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(y_scaled)
        else:
            # Subsample + propagate via nearest core point
            from sklearn.neighbors import BallTree
            rng = np.random.RandomState(42)
            idx = rng.choice(n, _DBSCAN_MAX_N, replace=False)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(y_scaled[idx])
            sub_labels = db.labels_
            core_mask = np.zeros(_DBSCAN_MAX_N, dtype=bool)
            core_mask[db.core_sample_indices_] = True
            core_pts = y_scaled[idx][core_mask]
            core_lbls = sub_labels[core_mask]
            if len(core_pts) == 0:
                cluster_labels = -np.ones(n, dtype=int)
            else:
                tree = BallTree(core_pts)
                dists, inds = tree.query(y_scaled, k=1)
                cluster_labels = np.where(dists.ravel() <= eps,
                                          core_lbls[inds.ravel()], -1)

        # Noise points (label=-1) are anomalies
        labels = np.where(cluster_labels == -1, -1, 1)
        # Score: 1 for anomalies, 0 for normal
        scores = np.where(cluster_labels == -1, 1.0, 0.0)

        n_clusters = int(len(set(cluster_labels.tolist()) - {-1}))
        result = {
            "model_name": "DBSCAN",
            "anomaly_labels": labels.tolist(),
            "anomaly_scores": scores.tolist(),
            "n_anomalies": int((labels == -1).sum()),
            "metadata": {"eps": eps, "min_samples": min_samples,
                          "n_clusters": n_clusters,
                          "n_features": int(y.shape[1]),
                          "subsampled": n > _DBSCAN_MAX_N},
        }
    except Exception as e:
        result = {
            "model_name": "DBSCAN",
            "anomaly_labels": [1] * n,
            "anomaly_scores": [0.0] * n,
            "n_anomalies": 0,
            "metadata": {"error": str(e)},
        }

    return json.dumps(result)


def _per_channel_union(method_tool, channels: List[np.ndarray]) -> np.ndarray:
    """Run a univariate anomaly method on each channel, union the results.

    A point is flagged anomalous if ANY channel flags it.
    Returns labels array of length n with -1 (anomaly) or 1 (normal).
    """
    n = len(channels[0])
    union_flags = np.zeros(n, dtype=bool)
    for ch in channels:
        try:
            res = json.loads(method_tool.invoke({"data": json.dumps(ch.tolist())}))
            ch_labels = np.array(res["anomaly_labels"])
            union_flags |= (ch_labels == -1)
        except Exception:
            pass  # channel failed — skip, don't flag
    return np.where(union_flags, -1, 1)


@tool
def run_anomaly_ensemble(
    data: str,
    features: str = "",
    min_votes: int = 3,
    **kwargs
) -> str:
    """Run all 7 anomaly detection methods and return majority-voted results.

    Supports both univariate and multivariate modes.

    **Univariate** (features=""):  all 7 methods run on target only.
    **Multivariate** (features=JSON 2D array):
      - IsolationForest & DBSCAN run on the full [target + features] matrix.
      - Z-score, MAD, Rolling, IQR, STL run per-channel (target + each
        feature column) and union the results (flagged in ANY channel →
        anomalous for that method).
      - Ensemble voting across 7 methods is unchanged.

    Args:
        data: JSON string of target values (list of floats)
        features: JSON string of feature matrix (list of lists). Empty
                  string for univariate mode.
        min_votes: Minimum methods that must agree (default: 3 of 7)

    Returns:
        JSON string with ensemble results, per-method counts, and voted labels
    """
    y_target = np.array(json.loads(data))
    n = len(y_target)

    # Parse optional feature matrix
    feat_matrix = None
    is_mv = False
    if features and features.strip() and features.strip() != "[]":
        try:
            feat_matrix = np.array(json.loads(features))
            if feat_matrix.ndim == 2 and feat_matrix.shape[0] == n and feat_matrix.shape[1] > 0:
                is_mv = True
            else:
                feat_matrix = None
        except Exception:
            feat_matrix = None

    # Per-channel methods (univariate internally)
    per_channel_methods = [
        ("ZScore", run_statistical_anomaly_detection),
        ("ModifiedZScore", run_modified_zscore),
        ("RollingStats", run_rolling_anomaly),
        ("IQR", run_iqr_anomaly),
        ("STL", run_stl_anomaly),
    ]
    # Natively multivariate methods
    mv_methods = [
        ("IsolationForest", run_isolation_forest),
        ("DBSCAN", run_dbscan_anomaly),
    ]

    all_labels = []
    method_counts = {}

    _MAX_CHANNELS = 10  # Cap per-channel union to target + top-K by variance

    if is_mv:
        # --- Multivariate mode ---
        # Build channel list: target + top-K most variable features
        n_feat = feat_matrix.shape[1]
        if n_feat <= _MAX_CHANNELS - 1:
            channels = [y_target] + [feat_matrix[:, i] for i in range(n_feat)]
        else:
            variances = np.var(feat_matrix, axis=0)
            top_k = np.argsort(variances)[::-1][:_MAX_CHANNELS - 1]
            channels = [y_target] + [feat_matrix[:, i] for i in top_k]

        # Per-channel methods: run on each channel, union results
        for name, method_tool in per_channel_methods:
            try:
                labels = _per_channel_union(method_tool, channels)
                all_labels.append(labels)
                method_counts[name] = int((labels == -1).sum())
            except Exception:
                all_labels.append(np.ones(n))
                method_counts[name] = 0

        # Multivariate methods: pass combined [target | features] matrix
        combined = np.hstack([y_target.reshape(-1, 1), feat_matrix])
        combined_json = json.dumps(combined.tolist())
        for name, method_tool in mv_methods:
            try:
                result_json = method_tool.invoke({"data": combined_json})
                result = json.loads(result_json)
                labels = np.array(result["anomaly_labels"])
                all_labels.append(labels)
                method_counts[name] = int((labels == -1).sum())
            except Exception:
                all_labels.append(np.ones(n))
                method_counts[name] = 0
    else:
        # --- Univariate mode (unchanged) ---
        all_methods = per_channel_methods + mv_methods
        for name, method_tool in all_methods:
            try:
                result_json = method_tool.invoke({"data": data})
                result = json.loads(result_json)
                labels = np.array(result["anomaly_labels"])
                all_labels.append(labels)
                method_counts[name] = int((labels == -1).sum())
            except Exception:
                all_labels.append(np.ones(n))
                method_counts[name] = 0

    # Ensemble voting
    all_labels_array = np.array(all_labels)
    anomaly_votes = (all_labels_array == -1).sum(axis=0)
    ensemble_labels = np.where(anomaly_votes >= min_votes, -1, 1)

    # Compute per-anomaly-point agreement
    anomaly_point_votes = anomaly_votes[ensemble_labels == -1]
    avg_agreement = float(np.mean(anomaly_point_votes)) if len(anomaly_point_votes) > 0 else 0.0
    avg_agreement = min(avg_agreement, float(len(per_channel_methods) + len(mv_methods)))

    result = {
        "model_name": "Ensemble",
        "anomaly_labels": ensemble_labels.tolist(),
        "anomaly_scores": anomaly_votes.tolist(),
        "n_anomalies": int((ensemble_labels == -1).sum()),
        "method_counts": method_counts,
        "metadata": {
            "min_votes": min_votes,
            "n_methods": len(per_channel_methods) + len(mv_methods),
            "average_agreement": avg_agreement,
            "multivariate": is_mv,
            "n_features": int(feat_matrix.shape[1]) if is_mv else 0,
        },
    }
    return json.dumps(result)
