"""
Curve-level weight optimization helpers for Summary GPR.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from ..variance_calculation_helpers.between_variance import (
    compute_between_model_variances,
    compute_weighted_mean,
)


@dataclass
class CurvewiseWeightResult:
    weights: np.ndarray
    weight_history: List[np.ndarray]
    curve_history: List[np.ndarray]
    n_iterations: int
    converged: bool
    final_delta: float


def iterative_weight_optimization(
    y_array: np.ndarray,
    S_array: np.ndarray,
    *,
    include_within: bool = True,
    include_between: bool = True,
    epsilon: float = 1e-12,
    convergence_tol: float = 1e-6,
    max_iterations: Optional[int] = None,
    return_history: bool = False,
    verbose: bool = True,
) -> CurvewiseWeightResult:
    """
    Iteratively optimize curve-level weights to minimize total variance.

    Parameters
    ----------
    y_array : (R, N) array
        Per-curve predictions (R curves, N grid points).
    S_array : (R, N) array
        Per-curve GPR posterior std at each grid point.
        The within-curve variance for curve r is computed directly as:
            sigma_within_r^2 = (1/N) sum_j S_r(j)^2
        No external binning or pre-averaging is applied.
    include_within : bool
        Include within-curve variance (individual GPR posterior).
    include_between : bool
        Include between-curve variance (deviation from weighted mean).
    epsilon : float
        Numerical jitter.
    convergence_tol : float
        RMS weight-change threshold for convergence.
    max_iterations : int or None
        Max iterations (None = unlimited).
    return_history : bool
        Store weight and curve history.
    verbose : bool
        Print convergence info.
    """
    num_models, n_points = y_array.shape
    weights = np.ones(num_models) / num_models

    weight_history: List[np.ndarray] = []
    curve_history: List[np.ndarray] = []

    iteration = 0
    converged = False
    final_delta = 0.0

    # Within-curve variance from individual GPR posterior (no binning):
    # sigma_within_r^2 = (1/N) sum_j S_r(j)^2
    if include_within:
        sigma_within_sq = np.sum(S_array ** 2, axis=1) / n_points  # (R,)
    else:
        sigma_within_sq = np.zeros(num_models)

    while True:
        iteration += 1

        weighted_mean = compute_weighted_mean(y_array, weights)

        if return_history:
            weight_history.append(weights.copy())
            curve_history.append(weighted_mean.copy())

        # Between-curve variance w.r.t. current weighted mean:
        # D_r = (1/N) sum_j (y_r(j) - y_bar(j))^2
        if include_between:
            D_i = compute_between_model_variances(y_array, weighted_mean)
        else:
            D_i = np.zeros(num_models)

        # Total per-curve variance: V_r = D_r + sigma_within_r^2
        V_i = D_i + sigma_within_sq + epsilon

        weights_new = 1.0 / V_i
        weights_new /= np.sum(weights_new)

        final_delta = np.linalg.norm(weights_new - weights) / np.sqrt(weights_new.size)
        if final_delta < convergence_tol:
            converged = True
            if verbose:
                print(f"  Converged after {iteration} iterations (delta_rms={final_delta:.2e})")
            weights = weights_new
            break

        if max_iterations is not None and iteration >= max_iterations:
            if verbose:
                print(f"  Max iterations ({max_iterations}) reached")
            weights = weights_new
            break

        weights = weights_new.copy()

    return CurvewiseWeightResult(
        weights=weights,
        weight_history=weight_history,
        curve_history=curve_history,
        n_iterations=iteration,
        converged=converged,
        final_delta=final_delta,
    )


def compute_weights_equal(n_models: int, n_points: int, scope: str = "curve") -> np.ndarray:
    """Create uniform weights (curve-level or pointwise)."""
    if scope.lower() == "curve":
        return np.full(n_models, 1.0 / n_models)
    return np.full((n_models, n_points), 1.0 / n_models)
