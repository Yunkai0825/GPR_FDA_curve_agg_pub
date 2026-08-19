# iterative_weight_sum_GPR/iterative_gpr_core.py
"""
Core Iterative Weighted-Sum GPR aggregation function.

This module provides the main compute_summary_gpr() function that
ties together variance scoring and weight optimization.

For lower-level functions, see:
- variance_calculation_helpers/: Variance computation functions
- weight_calculation_helpers/: Weight refinement algorithms

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

# Import ScalingInfo from pip1
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo

# Import from submodules (relative within iterative_weight_sum_GPR)
from .variance_calculation_helpers.between_variance import (
    compute_final_variance,
    compute_weighted_mean,
)
from .weight_calculation_helpers.curvewise_weight_helpers import (
    iterative_weight_optimization,
    compute_weights_equal,
)
from .weight_calculation_helpers.pointwise_weight_helpers import pointwise_weight_optimization


def _scale_multiplier(scaling: ScalingInfo) -> float:
    """Return multiplicative scale to map normalized covariance to real scale."""
    method = scaling.method
    if method in ("peak", "middle_average", "divide", "identity", ""):
        return float(scaling.params.get("factor", 1.0))
    if method == "standardize":
        return float(scaling.params.get("std", 1.0))
    if method == "minmax":
        min_val = float(scaling.params.get("min_val", 0.0))
        max_val = float(scaling.params.get("max_val", 1.0))
        feature_range = scaling.params.get("feature_range", (0, 1))
        return float((max_val - min_val) / (feature_range[1] - feature_range[0]))
    # Log and other non-linear transforms are not linear; leave unchanged
    return 1.0


@dataclass
class SummaryGPRCoreResult:
    """
    Result from compute_summary_gpr.
    
    Attributes
    ----------
    x_pred : np.ndarray
        X prediction values (transformed space).
    y_mean : np.ndarray
        Aggregated mean prediction (original scale).
    y_mean_norm : np.ndarray
        Aggregated mean prediction in normalized scale.
    y_std_real : np.ndarray
        Aggregated standard deviation using real-scale variance.
    y_std_norm : np.ndarray
        Aggregated standard deviation using normalized-scale variance.
    weights : np.ndarray
        Final model weights.
    weight_history : List[np.ndarray]
        Weight evolution during optimization.
    curve_history : List[np.ndarray]
        Summary curve evolution during optimization.
    y_scaling : ScalingInfo
        Aggregated Y-axis scaling info (weighted combination of input scalings).
    n_models : int
        Number of models aggregated.
    n_points : int
        Number of prediction points.
    """
    x_pred: np.ndarray
    y_mean: np.ndarray
    y_mean_norm: np.ndarray
    y_std_real: np.ndarray
    y_std_norm: np.ndarray
    weights: np.ndarray
    weight_history: List[np.ndarray]
    curve_history: List[np.ndarray]
    y_scaling: ScalingInfo
    n_models: int
    n_points: int


def _extract_rescaling_factors(y_scalings: List[ScalingInfo]) -> np.ndarray:
    """
    Extract rescaling factors from ScalingInfo objects.
    
    Parameters
    ----------
    y_scalings : List[ScalingInfo]
        Y-axis scaling info objects.
        
    Returns
    -------
    np.ndarray
        Array of scaling factors.
    """
    factors = []
    for scaling in y_scalings:
        factor = scaling.params.get("factor", 1.0)
        factors.append(factor)
    return np.array(factors, dtype=float)


def _compute_aggregated_scaling(
    y_scalings: List[ScalingInfo],
    weights: np.ndarray,
) -> ScalingInfo:
    """
    Compute aggregated ScalingInfo from weighted combination of input scalings.
    
    For linear scalings (divide_by_factor, standardize), computes weighted
    average of parameters. For non-linear scalings (log), uses the same
    method with weighted average parameters.
    
    Parameters
    ----------
    y_scalings : List[ScalingInfo]
        Input scaling info objects.
    weights : np.ndarray
        Model weights (will be normalized to sum to 1).
        
    Returns
    -------
    ScalingInfo
        Aggregated scaling info.
        
    Notes
    -----
    The algorithm assumes all input scalings use the same method.
    For mixed methods, falls back to weighted average factor with
    the first scaling's method.
    """
    if len(y_scalings) == 0:
        return ScalingInfo.identity()
    
    if len(y_scalings) == 1:
        return y_scalings[0]
    
    # Normalize weights
    w = weights / weights.sum()
    
    # Get the primary method (assume all same)
    method = y_scalings[0].method
    
    # Check if all scalings use the same method
    all_same = all(s.method == method for s in y_scalings)
    
    if method in ("peak", "middle_average", "divide") or not all_same:
        # Linear division scaling: weighted average of factors
        factors = _extract_rescaling_factors(y_scalings)
        agg_factor = float(np.dot(w, factors))
        return ScalingInfo.divide_by_factor(agg_factor, method_name=method)
    
    elif method == "standardize":
        # Standardization: weighted average of means and stds
        means = np.array([s.params.get("mean", 0.0) for s in y_scalings])
        stds = np.array([s.params.get("std", 1.0) for s in y_scalings])
        agg_mean = float(np.dot(w, means))
        agg_std = float(np.dot(w, stds))
        return ScalingInfo.standardize(agg_mean, agg_std)
    
    elif method == "minmax":
        # Min-max: weighted average of min and max
        mins = np.array([s.params.get("min_val", 0.0) for s in y_scalings])
        maxs = np.array([s.params.get("max_val", 1.0) for s in y_scalings])
        agg_min = float(np.dot(w, mins))
        agg_max = float(np.dot(w, maxs))
        feature_range = y_scalings[0].params.get("feature_range", (0, 1))
        return ScalingInfo.minmax(agg_min, agg_max, feature_range)
    
    elif method.startswith("log"):
        # Log transform: all should have same shift, just use first
        shift = y_scalings[0].params.get("shift", 1e-9)
        base = y_scalings[0].params.get("base", "log10")
        return ScalingInfo.log_transform(shift=shift, base=base)
    
    elif method == "identity":
        return ScalingInfo.identity()
    
    else:
        # Fallback: use weighted factor approach
        factors = _extract_rescaling_factors(y_scalings)
        agg_factor = float(np.dot(w, factors))
        return ScalingInfo.divide_by_factor(agg_factor, method_name=method)


def compute_summary_gpr(
    y_array: np.ndarray,
    S_array: np.ndarray,
    x_pred: np.ndarray,
    y_scalings: List[ScalingInfo],
    *,
    weight_mode: str = "iterative",
    weight_scope: str = "curve",
    include_within: bool = True,
    include_between: bool = True,
    variance_scale: str = "real",
    normalization_summary: bool = True,
    epsilon: float = 1e-12,
    convergence_tol: float = 1e-6,
    max_iterations: Optional[int] = None,
    verbose: bool = True,
) -> SummaryGPRCoreResult:
    """
    Compute summary GPR from individual predictions.
    
    This function:
    1. Optionally normalizes predictions by rescaling factors
    2. Optimizes weights using variance-based scoring
    3. Computes weighted mean and variance
    4. Returns results in original scale
    
    Parameters
    ----------
    y_array : np.ndarray
        Predictions, shape (n_models, n_points), in original scale.
    S_array : np.ndarray
        Standard deviations, shape (n_models, n_points), in original scale.
    x_pred : np.ndarray
        X prediction values, shape (n_points,).
    y_scalings : List[ScalingInfo]
        Per-model Y-axis scaling information objects.
    weight_mode : str
        "equal" or "iterative".
    weight_scope : str
        "curve" or "point".
    include_within : bool
        Include within-model variance in weight optimization.
    include_between : bool
        Include between-model variance in weight optimization.
    variance_scale : str
        "real" or "normalised".
    normalization_summary : bool
        Whether to work in normalized space during optimization.
    epsilon : float
        Small constant for numerical stability.
    convergence_tol : float
        Convergence tolerance for weight optimization.
    max_iterations : int, optional
        Maximum iterations for optimization (None = unlimited).
    verbose : bool
        Print optimization progress.
        
    Returns
    -------
    SummaryGPRCoreResult
        Summary GPR result with predictions, weights, and diagnostics.
    """
    n_models, n_points = y_array.shape
    
    # Always compute normalized versions (needed for variance in both modes)
    y_norm = np.array([y_scalings[i].transform(y_array[i]) for i in range(n_models)])
    S_norm = np.array([y_scalings[i].transform_std(S_array[i]) for i in range(n_models)])

    # Select optimization space
    if normalization_summary:
        y_opt = y_norm
        S_opt = S_norm
    else:
        y_opt = y_array
        S_opt = S_array
    
    # -------------------------------------------------------------------------
    # Weight Optimization
    # -------------------------------------------------------------------------
    weight_history: List[np.ndarray] = []
    curve_history: List[np.ndarray] = []
    
    if weight_mode.lower() == "equal":
        weights = compute_weights_equal(n_models, n_points, weight_scope)
    
    elif weight_mode.lower() == "iterative":
        if weight_scope.lower() == "curve":
            # Curve-level optimization: between-curve deviation + within-curve GPR variance
            # No pre-binning — pass full S_opt (R, N) directly
            opt_result = iterative_weight_optimization(
                y_opt, S_opt,
                include_within=include_within,
                include_between=include_between,
                epsilon=epsilon,
                convergence_tol=convergence_tol,
                max_iterations=max_iterations,
                return_history=True,
                verbose=verbose,
            )
            weights = opt_result.weights
            weight_history = opt_result.weight_history
            curve_history = opt_result.curve_history
            
        else:  # point
            # Pointwise optimization
            opt_result = pointwise_weight_optimization(
                y_opt, S_opt,
                epsilon=epsilon,
                convergence_tol=convergence_tol,
                max_iterations=max_iterations or 10000,
                return_history=True,
                verbose=verbose,
            )
            weights = opt_result.weights
            weight_history = opt_result.weight_history
            curve_history = opt_result.curve_history
    
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    
    # -------------------------------------------------------------------------
    # Compute Aggregated Prediction
    # -------------------------------------------------------------------------
    
    # Get weight vector for per-curve scaling factors
    if weights.ndim == 1:
        w_vec = weights
    else:
        w_vec = weights.mean(axis=1)
        w_vec /= w_vec.sum()

    agg_scaling = _compute_aggregated_scaling(y_scalings, w_vec)

    # Aggregated prediction
    # Principle: aggregate in the primary space, then transform the
    # aggregated mean to the other space using the aggregated scale
    # factor — NOT per-curve rescale then aggregate (which collapses
    # to identical results as real-space aggregation).
    if normalization_summary:
        # Aggregate in normalized space, transform back with agg scale
        y_agg_norm = compute_weighted_mean(y_norm, weights)
        y_agg_real = agg_scaling.inverse_transform(y_agg_norm)
    else:
        # Aggregate in real space, transform to normalized with agg scale
        y_agg_real = compute_weighted_mean(y_array, weights)
        y_agg_norm = agg_scaling.transform(y_agg_real)

    y_mean_norm = y_agg_norm
    
    # -------------------------------------------------------------------------
    # Compute Variance
    # -------------------------------------------------------------------------
    
    # Normalized-space variance (shape-focused, units of normalized scale)
    y_std_norm = compute_final_variance(
        y_norm, S_norm, weights, y_agg_norm,
        include_within=include_within,
        include_between=include_between,
    )

    # Real-space variance
    y_std_real = compute_final_variance(
        y_array, S_array, weights, y_agg_real,
        include_within=include_within,
        include_between=include_between,
    )

    return SummaryGPRCoreResult(
        x_pred=x_pred,
        y_mean=y_agg_real,
        y_mean_norm=y_mean_norm,
        y_std_real=y_std_real,
        y_std_norm=y_std_norm,
        weights=weights,
        weight_history=weight_history,
        curve_history=curve_history,
        y_scaling=agg_scaling,
        n_models=n_models,
        n_points=n_points,
    )
