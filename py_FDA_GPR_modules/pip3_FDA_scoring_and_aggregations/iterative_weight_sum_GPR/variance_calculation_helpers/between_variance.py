"""
Between-variance and total variance helpers for Summary GPR.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .within_variance import compute_within_model_variances


def compute_between_model_variances(y_array: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    """Average squared deviation from mean for each model (σ_between^2 per model)."""
    n_points = y_array.shape[1]
    D_i = np.sum((y_array - y_mean) ** 2, axis=1) / n_points
    return D_i


def compute_total_model_variance(
    y_array: np.ndarray,
    S_array: np.ndarray,
    y_mean: np.ndarray,
    *,
    include_within: bool = True,
    include_between: bool = True,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Curve-level total variance V_i = D_i + σ_within^2 + ε."""
    num_models = y_array.shape[0]

    if include_between:
        D_i = compute_between_model_variances(y_array, y_mean)
    else:
        D_i = np.zeros(num_models)

    if include_within:
        sigma_within = compute_within_model_variances(S_array)
    else:
        sigma_within = np.zeros(num_models)

    V_i = D_i + sigma_within + epsilon
    return V_i


def compute_pointwise_variance(
    y_array: np.ndarray,
    S_array: np.ndarray,
    y_mean: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Pointwise variance per model and point: (y_i-ȳ)^2 + s_i^2 + ε."""
    V = (y_array - y_mean) ** 2 + S_array ** 2 + epsilon
    return V


def compute_final_variance(
    y_array: np.ndarray,
    S_array: np.ndarray,
    weights: np.ndarray,
    y_mean: np.ndarray,
    *,
    include_within: bool = True,
    include_between: bool = True,
) -> np.ndarray:
    """Total aggregated std across points combining between and within components."""
    W = weights[:, np.newaxis] if weights.ndim == 1 else weights

    sig2_between = (W * (y_array - y_mean) ** 2).sum(axis=0) if include_between else 0.0
    sig2_within = (W * S_array ** 2).sum(axis=0) if include_within else 0.0

    sigma_total = np.sqrt(sig2_between + sig2_within)
    return sigma_total


def compute_weighted_mean(y_array: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean prediction ȳ = Σ_i w_i y_i."""
    W = weights[:, np.newaxis] if weights.ndim == 1 else weights
    return (W * y_array).sum(axis=0)
