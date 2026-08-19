# pip3_FDA_scoring_and_aggregations/summary_gpr_plotting.py
"""
Plotting utilities for Summary GPR.

This module provides:
- Generic base plot functions (plot_summary_gpr, plot_weight_convergence, etc.)
  used by all methods
- Comparison plots (baseline vs operator fusion vs FGPR)
- CSV loading utilities

Method-specific CSV wrappers live in each method folder:
- iterative_weight_sum_GPR/iterative_plot_helpers.py
- operator_fusion_noweight/operator_fusion_plot_helpers.py
- functional_GPR/fgpr_plot_helpers.py

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union
from scipy.stats import norm

from .summary_gpr_loader import IndividualGPRData
from ..pip0_dataloading.filename_parser import format_group_key_title

# Alias for internal use (keeps existing code working)
_format_group_key_title = format_group_key_title


# =============================================================================
# CSV Loading Utilities
# =============================================================================

def load_summary_gpr_csv(csv_path: Path) -> pd.DataFrame:
    """Load Summary_GPR CSV."""
    return pd.read_csv(csv_path)

def load_weight_history_csv(csv_path: Path) -> pd.DataFrame:
    """Load weight history CSV."""
    return pd.read_csv(csv_path)

def load_curve_history_csv(csv_path: Path) -> pd.DataFrame:
    """Load curve history CSV."""
    return pd.read_csv(csv_path)

def load_converged_weights_csv(csv_path: Path) -> pd.DataFrame:
    """Load converged weights CSV."""
    return pd.read_csv(csv_path)


# =============================================================================
# Core Plotting Functions (take data directly for in-pipeline use)
# =============================================================================

def plot_summary_gpr(
    x_display: np.ndarray,
    y_mean: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    output_path: Path,
    *,
    individual_curves: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    group_key: str = "unknown",
    confidence_level: float = 0.75,
    individual_alpha: float = 0.20,
    min_time_cap: float = 0.01,
    max_time_cap: float = 1e4,
    x_axis_label: str = "X_label",
    y_axis_label: str = "Y_label",
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for summary GPR with optional individual curves overlay.
    
    Parameters
    ----------
    x_display : np.ndarray
        Time values in seconds (original scale, not log-transformed).
    y_mean : np.ndarray
        Mean prediction values.
    y_lower : np.ndarray
        Lower confidence bound.
    y_upper : np.ndarray
        Upper confidence bound.
    output_path : Path
        Full path for output file (including filename).
    individual_curves : List[Tuple[np.ndarray, np.ndarray]], optional
        List of (x, y) tuples for individual curves overlay.
    group_key : str
        Group identifier for title.
    confidence_level : float
        Confidence interval level (0-1). Used for legend text.
    individual_alpha : float
        Transparency for individual curves.
    min_time_cap : float
        Minimum time for display (seconds).
    max_time_cap : float
        Maximum time for display (seconds).
    x_axis_label : str
        Label for x-axis.
    y_axis_label : str
        Label for y-axis.
    figsize : Tuple[int, int]
        Figure size.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Plot individual curves if provided
    if individual_curves:
        for x_ind, y_ind in individual_curves:
            plt.plot(x_ind, y_ind, color='gray', alpha=individual_alpha, linewidth=1)

    # Auto-derive max_time_cap from data when not explicitly set
    if max_time_cap is None:
        max_time_cap = float(x_display.max())
        if individual_curves:
            for x_ind, _ in individual_curves:
                max_time_cap = max(max_time_cap, float(np.max(x_ind)))
        max_time_cap *= 1.05  # 5% padding
    
    # Plot summary with confidence band
    plt.plot(x_display, y_mean, 'b-', linewidth=2, label='Summary GPR')
    plt.fill_between(
        x_display, y_lower, y_upper,
        color='blue', alpha=0.2,
        label=f'{int(confidence_level * 100)}% Confidence Interval'
    )
    
    # Format plot - create title from group_key
    # Parse group_key format like "potential=-1.95" or "pH=7.4|temp=25"
    plt.title(f'Summary GPR: {_format_group_key_title(group_key)}')
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xscale('log')
    plt.xlim([min_time_cap, max_time_cap])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved summary GPR plot to {output_path}")
    
    return output_path


def plot_summary_comparison(
    *,
    x_real: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_lower: np.ndarray,
    baseline_upper: np.ndarray,
    operator_mean: np.ndarray,
    operator_lower: np.ndarray,
    operator_upper: np.ndarray,
    output_directory: Path,
    group_key: str,
    x_axis_label: str,
    y_axis_label: str,
    min_time_cap: float,
    max_time_cap: float,
    individual_curves: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    individual_alpha: float = 0.1,
    fgpr_mean: Optional[np.ndarray] = None,
    fgpr_lower: Optional[np.ndarray] = None,
    fgpr_upper: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """Overlay baseline summary vs operator-fusion vs FGPR band on the same axes."""
    if baseline_mean is None or operator_mean is None:
        return None

    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Summary_GPR_{safe_key}_comparison.png'

    plt.figure(figsize=(10, 6))

    if individual_curves:
        for x_ind, y_ind in individual_curves:
            plt.plot(x_ind, y_ind, color='gray', alpha=individual_alpha, linewidth=1)

    # Auto-derive max_time_cap from data when not explicitly set
    if max_time_cap is None:
        max_time_cap = float(x_real.max())
        if individual_curves:
            for x_ind, _ in individual_curves:
                max_time_cap = max(max_time_cap, float(np.max(x_ind)))
        max_time_cap *= 1.05  # 5% padding

    plt.plot(x_real, baseline_mean, color='tab:blue', linewidth=2, label='Summary GPR')
    plt.fill_between(
        x_real, baseline_lower, baseline_upper,
        color='tab:blue', alpha=0.18, label='Summary CI'
    )

    plt.plot(x_real, operator_mean, color='tab:orange', linewidth=2, label='Operator fusion')
    plt.fill_between(
        x_real, operator_lower, operator_upper,
        color='tab:orange', alpha=0.18, label='Operator CI'
    )

    if fgpr_mean is not None and fgpr_lower is not None and fgpr_upper is not None:
        plt.plot(x_real, fgpr_mean, color='tab:green', linewidth=2, label='FGPR')
        plt.fill_between(
            x_real, fgpr_lower, fgpr_upper,
            color='tab:green', alpha=0.18, label='FGPR CI'
        )

    title_parts = ['Summary vs Operator Fusion']
    if fgpr_mean is not None:
        title_parts[0] = 'Summary vs Operator vs FGPR'
    plt.title(f'{title_parts[0]}: {_format_group_key_title(group_key)}')
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xscale('log')
    plt.xlim([min_time_cap, max_time_cap])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved comparison plot to {output_path}")

    return output_path


def plot_summary_comparison_normalized(
    *,
    x_real: np.ndarray,
    baseline_mean_norm: np.ndarray,
    baseline_lower_norm: np.ndarray,
    baseline_upper_norm: np.ndarray,
    operator_mean_norm: Optional[np.ndarray] = None,
    operator_lower_norm: Optional[np.ndarray] = None,
    operator_upper_norm: Optional[np.ndarray] = None,
    fgpr_mean_norm: Optional[np.ndarray] = None,
    fgpr_lower_norm: Optional[np.ndarray] = None,
    fgpr_upper_norm: Optional[np.ndarray] = None,
    output_directory: Path,
    group_key: str,
    x_axis_label: str,
    y_axis_label: str,
    min_time_cap: float,
    max_time_cap: float,
    individual_curves_norm: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    individual_alpha: float = 0.25,
    verbose: bool = True,
) -> Optional[Path]:
    """Comparison plot of all methods in normalized space with individual GPR curves."""
    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Summary_GPR_{safe_key}_comparison_normalized.png'

    plt.figure(figsize=(10, 6))

    # Individual GPR curves in normalized space
    if individual_curves_norm:
        for idx, (x_ind, y_ind) in enumerate(individual_curves_norm):
            lbl = 'Individual GPR' if idx == 0 else None
            plt.plot(x_ind, y_ind, color='gray', alpha=individual_alpha, linewidth=1, label=lbl)

    # Auto-derive max_time_cap from data when not explicitly set
    if max_time_cap is None:
        max_time_cap = float(x_real.max())
        if individual_curves_norm:
            for x_ind, _ in individual_curves_norm:
                max_time_cap = max(max_time_cap, float(np.max(x_ind)))
        max_time_cap *= 1.05  # 5% padding

    # Baseline (Summary GPR)
    plt.plot(x_real, baseline_mean_norm, color='tab:blue', linewidth=2, label='Summary GPR')
    plt.fill_between(
        x_real, baseline_lower_norm, baseline_upper_norm,
        color='tab:blue', alpha=0.18, label='Summary CI'
    )

    # Operator fusion
    if operator_mean_norm is not None and operator_lower_norm is not None:
        plt.plot(x_real, operator_mean_norm, color='tab:orange', linewidth=2, label='Operator fusion')
        plt.fill_between(
            x_real, operator_lower_norm, operator_upper_norm,
            color='tab:orange', alpha=0.18, label='Operator CI'
        )

    # FGPR
    if fgpr_mean_norm is not None and fgpr_lower_norm is not None:
        plt.plot(x_real, fgpr_mean_norm, color='tab:green', linewidth=2, label='FGPR')
        plt.fill_between(
            x_real, fgpr_lower_norm, fgpr_upper_norm,
            color='tab:green', alpha=0.18, label='FGPR CI'
        )

    n_methods = 1 + (operator_mean_norm is not None) + (fgpr_mean_norm is not None)
    if n_methods == 3:
        title_tag = 'Summary vs Operator vs FGPR'
    elif operator_mean_norm is not None:
        title_tag = 'Summary vs Operator Fusion'
    else:
        title_tag = 'Summary GPR'
    plt.title(f'{title_tag} (normalized): {_format_group_key_title(group_key)}')
    plt.xlabel(x_axis_label)
    plt.ylabel(f'{y_axis_label} [normalized]')
    plt.xscale('log')
    plt.xlim([min_time_cap, max_time_cap])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved normalized comparison plot to {output_path}")

    return output_path


def plot_weight_convergence(
    iterations: np.ndarray,
    weight_history: np.ndarray,
    output_path: Path,
    *,
    group_key: str = "unknown",
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for weight convergence.
    
    Parameters
    ----------
    iterations : np.ndarray
        Iteration numbers (1D array).
    weight_history : np.ndarray
        Weight values, shape (n_iterations, n_models).
    output_path : Path
        Full path for output file.
    group_key : str
        Group identifier for title.
    figsize : Tuple[int, int]
        Figure size.
    verbose : bool
        Print status.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    n_models = weight_history.shape[1] if weight_history.ndim > 1 else 1
    
    if weight_history.ndim == 1:
        plt.plot(iterations, weight_history, '-o', markersize=3, alpha=0.7)
    else:
        for i in range(n_models):
            plt.plot(iterations, weight_history[:, i], '-o', markersize=3, alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Weight')
    plt.title(f'Weight Convergence: {group_key}')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved weight convergence plot to {output_path}")
    
    return output_path


def plot_summary_curve_iterations(
    x_display: np.ndarray,
    curve_history: np.ndarray,
    output_path: Path,
    *,
    group_key: str = "unknown",
    min_time_cap: float = 0.01,
    max_time_cap: float = 1e4,
    x_axis_label: str = "X_label",
    y_axis_label: str = "Y_label",
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for summary curve iteration history.
    
    Parameters
    ----------
    x_display : np.ndarray
        Time values in seconds (original scale).
    curve_history : np.ndarray
        Curve values, shape (n_points, n_iterations).
    output_path : Path
        Full path for output file.
    group_key : str
        Group identifier for title.
    min_time_cap : float
        Minimum time for x-axis.
    max_time_cap : float
        Maximum time for x-axis.
    x_axis_label : str
        Label for x-axis.
    y_axis_label : str
        Label for y-axis.
    figsize : Tuple[int, int]
        Figure size.
    verbose : bool
        Print status.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    n_iterations = curve_history.shape[1] if curve_history.ndim > 1 else 1
    
    plt.figure(figsize=figsize)
    
    if curve_history.ndim == 1:
        plt.plot(x_display, curve_history, 'k-', linewidth=2, label='Final Summary GPR')
    else:
        for i in range(n_iterations):
            summary_curve = curve_history[:, i]
            if i == n_iterations - 1:
                plt.plot(x_display, summary_curve, 'k-', linewidth=2, label='Final Summary GPR')
            else:
                plt.plot(x_display, summary_curve, label=f'Iteration {i+1}', alpha=0.3)
    
    # Format title from group_key
    plt.title(f'Summary GPR Iterations: {_format_group_key_title(group_key)}')
    
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.xscale('log')
    plt.xlim([min_time_cap, max_time_cap])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved iterations plot to {output_path}")
    
    return output_path


def plot_weight_distribution(
    weights: np.ndarray,
    output_path: Path,
    *,
    group_key: str = "unknown",
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for weight distribution histogram.
    
    Parameters
    ----------
    weights : np.ndarray
        Converged weight values.
    output_path : Path
        Full path for output file.
    group_key : str
        Group identifier for title.
    figsize : Tuple[int, int]
        Figure size.
    verbose : bool
        Print status.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    plt.hist(weights, bins=min(10, len(weights)), color='steelblue', edgecolor='k')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    
    # Format title from group_key
    plt.title(f'Weight Distribution: {_format_group_key_title(group_key)}')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if verbose:
        print(f"Saved weight distribution plot to {output_path}")
    
    return output_path


# =============================================================================
# Backward-compatible re-exports from iterative_plot_helpers
# =============================================================================
# CSV wrapper functions have moved to iterative_weight_sum_GPR/iterative_plot_helpers.py
# These re-exports keep existing imports working.

def plot_summary_gpr_from_csv(*args, **kwargs):
    """Re-export: see iterative_weight_sum_GPR.iterative_plot_helpers."""
    from .iterative_weight_sum_GPR.iterative_plot_helpers import plot_summary_gpr_from_csv as _f
    return _f(*args, **kwargs)

def plot_weight_convergence_from_csv(*args, **kwargs):
    """Re-export: see iterative_weight_sum_GPR.iterative_plot_helpers."""
    from .iterative_weight_sum_GPR.iterative_plot_helpers import plot_weight_convergence_from_csv as _f
    return _f(*args, **kwargs)

def plot_summary_curve_iterations_from_csv(*args, **kwargs):
    """Re-export: see iterative_weight_sum_GPR.iterative_plot_helpers."""
    from .iterative_weight_sum_GPR.iterative_plot_helpers import plot_summary_curve_iterations_from_csv as _f
    return _f(*args, **kwargs)

def plot_weight_distribution_from_csv(*args, **kwargs):
    """Re-export: see iterative_weight_sum_GPR.iterative_plot_helpers."""
    from .iterative_weight_sum_GPR.iterative_plot_helpers import plot_weight_distribution_from_csv as _f
    return _f(*args, **kwargs)