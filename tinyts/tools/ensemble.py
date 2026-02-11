"""Ensemble tools for combining model predictions."""

import json
from typing import Dict, List

import numpy as np
from langchain.tools import tool


@tool
def compute_ensemble_weights(
    model_scores: str,
    method: str = "inverse_error",
    **kwargs
) -> str:
    """Compute ensemble weights based on validation scores.
    
    Args:
        model_scores: JSON string mapping model names to validation scores (lower is better)
        method: Weighting method ('inverse_error', 'softmax', 'uniform')
        
    Returns:
        JSON string with model weights
    """
    scores_dict = json.loads(model_scores)
    
    model_names = list(scores_dict.keys())
    scores = np.array(list(scores_dict.values()))
    
    if method == "inverse_error":
        # Weight inversely proportional to error
        # Replace inf/nan with large finite value so they get near-zero weight
        safe_scores = np.where(np.isfinite(scores), scores, 1e10)
        weights = 1.0 / (safe_scores + 1e-6)
        w_sum = weights.sum()
        weights = weights / w_sum if w_sum > 0 else np.ones(len(scores)) / len(scores)
    
    elif method == "softmax":
        # Softmax of negative scores (lower score = higher weight)
        safe_scores = np.where(np.isfinite(scores), scores, 1e10)
        shifted = -safe_scores - np.max(-safe_scores)  # numerical stability
        exp_scores = np.exp(shifted)
        weights = exp_scores / exp_scores.sum()
    
    elif method == "uniform":
        # Equal weights
        weights = np.ones(len(scores)) / len(scores)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create weight dictionary
    weight_dict = {name: float(w) for name, w in zip(model_names, weights)}
    
    result = {
        "method": method,
        "weights": weight_dict,
        "metadata": {
            "n_models": len(model_names),
            "input_scores": scores_dict,
        }
    }
    
    return json.dumps(result)


@tool
def combine_predictions(
    predictions_dict: str,
    weights_dict: str,
    **kwargs
) -> str:
    """Combine predictions from multiple models using weights.
    
    Args:
        predictions_dict: JSON string mapping model names to predictions
        weights_dict: JSON string mapping model names to weights
        
    Returns:
        JSON string with combined predictions
    """
    predictions = json.loads(predictions_dict)
    weights = json.loads(weights_dict)
    
    # Ensure all models have weights
    model_names = list(predictions.keys())
    
    # Stack predictions
    pred_arrays = [np.array(predictions[name]) for name in model_names]
    weight_values = [weights.get(name, 1.0 / len(model_names)) for name in model_names]
    
    # Normalize weights
    weight_values = np.array(weight_values)
    weight_values = weight_values / weight_values.sum()
    
    # Weighted average
    combined = np.zeros_like(pred_arrays[0])
    for pred, weight in zip(pred_arrays, weight_values):
        combined += weight * pred
    
    result = {
        "combined_predictions": combined.tolist(),
        "metadata": {
            "n_models": len(model_names),
            "models": model_names,
            "weights": {name: float(w) for name, w in zip(model_names, weight_values)},
        }
    }
    
    return json.dumps(result)


@tool
def compute_median_ensemble(
    predictions_dict: str,
    **kwargs
) -> str:
    """Compute median ensemble (robust to outliers).
    
    Args:
        predictions_dict: JSON string mapping model names to predictions
        
    Returns:
        JSON string with median predictions
    """
    predictions = json.loads(predictions_dict)
    
    # Stack predictions
    pred_arrays = [np.array(preds) for preds in predictions.values()]
    pred_matrix = np.stack(pred_arrays, axis=0)
    
    # Compute median
    median_preds = np.median(pred_matrix, axis=0)
    
    result = {
        "median_predictions": median_preds.tolist(),
        "metadata": {
            "n_models": len(predictions),
            "models": list(predictions.keys()),
        }
    }
    
    return json.dumps(result)

