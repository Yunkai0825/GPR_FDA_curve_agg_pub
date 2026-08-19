# pip1_datapreprocessing/preprc_downsampling.py
"""
Downsampling functions for electrochemical transient data.

This module provides various strategies for reducing the number of data points
while preserving important features of the signal. Downsampling is essential for:
- Reducing computational cost in GPR fitting
- Creating uniform density in log-time space
- Preserving critical features (peaks, transitions, steady-state)

Available Strategies:
---------------------
- uniform_bins: Equal-sized bins in transformed x-space (default)
- adaptive: Density-based sampling that preserves high-curvature regions
- importance_weighted: Samples more densely where signal changes rapidly

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal, Union
from dataclasses import dataclass


# =============================================================================
# Downsampling Result Container
# =============================================================================

@dataclass
class DownsampleResult:
    """
    Container for downsampling results with metadata.
    
    Attributes
    ----------
    df : pd.DataFrame
        Downsampled DataFrame.
    indices : np.ndarray
        Original indices of selected points.
    method : str
        Name of the downsampling method used.
    original_count : int
        Number of points before downsampling.
    downsampled_count : int
        Number of points after downsampling.
    reduction_ratio : float
        Ratio of points removed (1 - downsampled/original).
    """
    df: pd.DataFrame
    indices: np.ndarray
    method: str
    original_count: int
    downsampled_count: int
    
    @property
    def reduction_ratio(self) -> float:
        """Fraction of points removed."""
        if self.original_count == 0:
            return 0.0
        return 1.0 - (self.downsampled_count / self.original_count)
    
    def __repr__(self) -> str:
        return (
            f"DownsampleResult(method='{self.method}', "
            f"{self.original_count} -> {self.downsampled_count} points, "
            f"reduction={self.reduction_ratio:.1%})"
        )


# =============================================================================
# Core Downsampling Functions
# =============================================================================

def downsample_uniform_bins(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    selection: Literal["middle", "first", "last", "random"] = "middle",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Downsample using uniform bins in x-space.
    
    Divides the x-range into equal-sized bins and selects one point per bin.
    This is the default method, equivalent to the original `downsample_data`.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with transformed x column.
    max_points : int
        Maximum number of points to keep.
    x_col : str
        Name of the x column to use for binning.
    selection : str
        How to select point within each bin:
        - 'middle': Select the middle point (default)
        - 'first': Select the first point
        - 'last': Select the last point
        - 'random': Select a random point
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (downsampled DataFrame, indices of selected points)
        
    Example
    -------
    >>> df_ds, indices = downsample_uniform_bins(df, max_points=500)
    >>> print(f"Reduced from {len(df)} to {len(df_ds)} points")
    """
    df = df.copy()
    num_bins = min(max_points, len(df))
    
    if num_bins >= len(df):
        # No downsampling needed
        return df, df.index.to_numpy()
    
    x_values = np.asarray(df[x_col].values)
    bin_edges = np.linspace(x_values.min(), x_values.max(), num_bins + 1)
    df['_bin_ds'] = np.digitize(x_values, bin_edges) - 1
    df['_bin_ds'] = np.clip(df['_bin_ds'], 0, num_bins - 1)
    
    # Selection strategy
    if selection == "middle":
        selector = lambda x: x.iloc[len(x) // 2].name
    elif selection == "first":
        selector = lambda x: x.iloc[0].name
    elif selection == "last":
        selector = lambda x: x.iloc[-1].name
    elif selection == "random":
        selector = lambda x: x.iloc[np.random.randint(len(x))].name
    else:
        raise ValueError(f"Unknown selection method: {selection}")
    
    # Group and select
    try:
        downsampled_indices = df.groupby('_bin_ds').apply(
            selector, include_groups=False  # type: ignore[call-overload]
        )
    except TypeError:
        # Fallback for older pandas without include_groups
        downsampled_indices = df.groupby('_bin_ds').apply(selector)
    
    df_downsampled = df.loc[downsampled_indices.values].drop(columns=['_bin_ds'])
    return df_downsampled, downsampled_indices.values


def downsample_adaptive(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    curvature_weight: float = 0.5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Adaptive downsampling that preserves high-curvature regions.
    
    This method computes local curvature (second derivative) and samples
    more densely in regions where the signal is changing rapidly. This
    preserves important features like peaks and transitions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    max_points : int
        Maximum number of points to keep.
    x_col : str
        Name of the x column.
    y_col : str
        Name of the y column.
    curvature_weight : float
        Weight for curvature-based sampling (0 to 1).
        - 0: Pure uniform sampling
        - 1: Pure curvature-based sampling
        - 0.5: Equal weight (default)
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (downsampled DataFrame, indices of selected points)
        
    Example
    -------
    >>> df_ds, indices = downsample_adaptive(df, max_points=500, curvature_weight=0.7)
    """
    if len(df) <= max_points:
        return df.copy(), df.index.to_numpy()
    
    df = df.copy().reset_index(drop=True)
    x = df[x_col].values
    y = df[y_col].values
    n = len(x)
    
    # Compute local curvature (approximate second derivative)
    # Use central differences with boundary handling
    curvature = np.zeros(n)
    if n > 2:
        # Central difference for interior points
        dx = np.diff(x)
        dy = np.diff(y)
        
        # Avoid division by zero
        dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
        
        # First derivative
        dydx = dy / dx
        
        # Second derivative (curvature proxy)
        for i in range(1, n - 1):
            if i < len(dydx):
                curvature[i] = abs(dydx[i] - dydx[i-1]) if i > 0 and i < len(dydx) else 0
        
        # Handle boundaries
        curvature[0] = curvature[1] if n > 1 else 0
        curvature[-1] = curvature[-2] if n > 1 else 0
    
    # Normalize curvature to [0, 1]
    curv_max = curvature.max()
    if curv_max > 0:
        curvature_norm = curvature / curv_max
    else:
        curvature_norm = np.zeros(n)
    
    # Create importance weights: combination of uniform and curvature
    uniform_weight = np.ones(n) / n
    curvature_prob = curvature_norm / (curvature_norm.sum() + 1e-12)
    
    importance = (1 - curvature_weight) * uniform_weight + curvature_weight * curvature_prob
    importance = importance / importance.sum()  # Normalize to probability
    
    # Sample without replacement
    selected_indices = np.random.choice(
        n, size=min(max_points, n), replace=False, p=importance
    )
    selected_indices = np.sort(selected_indices)  # Keep original order
    
    df_downsampled = df.iloc[selected_indices]
    return df_downsampled, selected_indices


def downsample_gradient_preserving(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    min_density_factor: float = 0.3,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Gradient-preserving downsampling for transient signals.
    
    This method ensures higher sampling density in regions where the 
    gradient (dy/dx) is large, which is important for capturing the
    initial transient behavior in electrochemical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    max_points : int
        Maximum number of points to keep.
    x_col : str
        Name of the x column.
    y_col : str
        Name of the y column.
    min_density_factor : float
        Minimum density as fraction of max density (0 to 1).
        Higher values make sampling more uniform.
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (downsampled DataFrame, indices of selected points)
        
    Example
    -------
    >>> # Preserve early transient with high gradient
    >>> df_ds, indices = downsample_gradient_preserving(df, max_points=500)
    """
    if len(df) <= max_points:
        return df.copy(), df.index.to_numpy()
    
    df = df.copy().reset_index(drop=True)
    x = df[x_col].values
    y = df[y_col].values
    n = len(x)
    
    # Compute absolute gradient
    gradient = np.zeros(n)
    if n > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
        grad = np.abs(dy / dx)
        
        # Extend to all points
        gradient[:-1] = grad
        gradient[-1] = grad[-1] if len(grad) > 0 else 0
    
    # Normalize and apply minimum density
    grad_max = gradient.max()
    if grad_max > 0:
        gradient_norm = gradient / grad_max
    else:
        gradient_norm = np.ones(n) / n
    
    # Apply minimum density floor
    gradient_norm = np.maximum(gradient_norm, min_density_factor)
    
    # Convert to probability
    importance = gradient_norm / gradient_norm.sum()
    
    # Sample without replacement
    selected_indices = np.random.choice(
        n, size=min(max_points, n), replace=False, p=importance
    )
    selected_indices = np.sort(selected_indices)
    
    df_downsampled = df.iloc[selected_indices]
    return df_downsampled, selected_indices


def downsample_feature_aware(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    peak_fraction: float = 0.2,
    transition_fraction: float = 0.3,
    tail_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Feature-aware downsampling optimized for electrochemical transients.
    
    This method divides the signal into three regions based on x-position:
    1. Peak region (early x): High density for capturing initial transient
    2. Transition region (middle x): Medium density for decay
    3. Tail region (late x): Lower density for steady-state
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    max_points : int
        Total maximum number of points to keep.
    x_col : str
        Name of the x column.
    y_col : str
        Name of the y column.
    peak_fraction : float
        Fraction of x-range considered as peak region (0 to 1).
    transition_fraction : float
        Fraction of x-range considered as transition (0 to 1).
    tail_fraction : float
        Fraction of x-range considered as tail (0 to 1).
        Note: peak + transition + tail should equal 1.0
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (downsampled DataFrame, indices of selected points)
        
    Example
    -------
    >>> # More points in early transient, fewer in steady-state tail
    >>> df_ds, indices = downsample_feature_aware(
    ...     df, max_points=500,
    ...     peak_fraction=0.15, transition_fraction=0.35, tail_fraction=0.5
    ... )
    """
    if len(df) <= max_points:
        return df.copy(), df.index.to_numpy()
    
    df = df.copy().reset_index(drop=True)
    x = df[x_col].values
    n = len(x)
    
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    
    # Define region boundaries
    peak_end = x_min + peak_fraction * x_range
    transition_end = x_min + (peak_fraction + transition_fraction) * x_range
    
    # Assign points to regions
    peak_mask = x <= peak_end
    transition_mask = (x > peak_end) & (x <= transition_end)
    tail_mask = x > transition_end
    
    # Count points in each region
    n_peak = peak_mask.sum()
    n_transition = transition_mask.sum()
    n_tail = tail_mask.sum()
    
    # Allocate points proportionally with higher density in peak region
    # Peak gets 40% of points, transition 35%, tail 25%
    peak_points = int(0.40 * max_points)
    transition_points = int(0.35 * max_points)
    tail_points = max_points - peak_points - transition_points
    
    # Clamp to available points
    peak_points = min(peak_points, n_peak)
    transition_points = min(transition_points, n_transition)
    tail_points = min(tail_points, n_tail)
    
    selected_indices = []
    
    # Sample from each region
    for mask, n_select in [
        (peak_mask, peak_points),
        (transition_mask, transition_points),
        (tail_mask, tail_points)
    ]:
        region_indices = np.where(mask)[0]
        if len(region_indices) > 0 and n_select > 0:
            if len(region_indices) <= n_select:
                selected_indices.extend(region_indices)
            else:
                # Uniform sampling within region
                step = len(region_indices) / n_select
                sampled = [region_indices[int(i * step)] for i in range(n_select)]
                selected_indices.extend(sampled)
    
    selected_indices = np.array(sorted(selected_indices))
    df_downsampled = df.iloc[selected_indices]
    return df_downsampled, selected_indices


# =============================================================================
# High-Level Downsampling Interface
# =============================================================================

DownsampleMethod = Literal["uniform", "adaptive", "gradient", "feature_aware"]


def downsample_data(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    method: DownsampleMethod = "uniform",
    **kwargs,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Downsample DataFrame using the specified method.
    
    This is the main entry point for downsampling. It dispatches to the
    appropriate method based on the `method` parameter.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    max_points : int
        Maximum number of points to keep.
    x_col : str
        Name of the x column (default: 'x_transformed').
    y_col : str
        Name of the y column (default: 'y_transformed').
    method : str
        Downsampling method:
        - 'uniform': Equal-sized bins (default, backward compatible)
        - 'adaptive': Curvature-aware sampling
        - 'gradient': Gradient-preserving sampling
        - 'feature_aware': Region-based sampling for transients
    **kwargs
        Additional arguments passed to the specific method.
        
    Returns
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (downsampled DataFrame, indices of selected points)
        
    Example
    -------
    >>> # Default uniform binning (backward compatible)
    >>> df_ds, indices = downsample_data(df, max_points=500)
    >>> 
    >>> # Adaptive sampling for complex signals
    >>> df_ds, indices = downsample_data(
    ...     df, max_points=500, method='adaptive', curvature_weight=0.7
    ... )
    >>> 
    >>> # Feature-aware for electrochemical transients
    >>> df_ds, indices = downsample_data(
    ...     df, max_points=500, method='feature_aware',
    ...     peak_fraction=0.2, transition_fraction=0.3
    ... )
    """
    if method == "uniform":
        return downsample_uniform_bins(df, max_points, x_col, **kwargs)
    elif method == "adaptive":
        return downsample_adaptive(df, max_points, x_col, y_col, **kwargs)
    elif method == "gradient":
        return downsample_gradient_preserving(df, max_points, x_col, y_col, **kwargs)
    elif method == "feature_aware":
        return downsample_feature_aware(df, max_points, x_col, y_col, **kwargs)
    else:
        raise ValueError(
            f"Unknown downsampling method: {method}. "
            f"Choose from: uniform, adaptive, gradient, feature_aware"
        )


def downsample_with_result(
    df: pd.DataFrame,
    max_points: int,
    x_col: str = "x_transformed",
    y_col: str = "y_transformed",
    method: DownsampleMethod = "uniform",
    **kwargs,
) -> DownsampleResult:
    """
    Downsample DataFrame and return detailed result object.
    
    Same as `downsample_data` but returns a `DownsampleResult` object
    with additional metadata about the downsampling operation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x and y columns.
    max_points : int
        Maximum number of points to keep.
    x_col : str
        Name of the x column.
    y_col : str
        Name of the y column.
    method : str
        Downsampling method.
    **kwargs
        Additional arguments passed to the specific method.
        
    Returns
    -------
    DownsampleResult
        Object containing downsampled data and metadata.
        
    Example
    -------
    >>> result = downsample_with_result(df, max_points=500, method='adaptive')
    >>> print(result)  # Shows reduction statistics
    >>> df_ds = result.df
    """
    original_count = len(df)
    df_ds, indices = downsample_data(df, max_points, x_col, y_col, method, **kwargs)
    
    return DownsampleResult(
        df=df_ds,
        indices=indices,
        method=method,
        original_count=original_count,
        downsampled_count=len(df_ds),
    )
