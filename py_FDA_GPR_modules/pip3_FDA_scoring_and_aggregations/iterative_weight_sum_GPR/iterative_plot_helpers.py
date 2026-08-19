# iterative_weight_sum_GPR/iterative_plot_helpers.py
"""
Plot helpers for Iterative Weighted-Sum GPR aggregation.

Handles:
- Summary GPR plot (mean + CI band) from CSV
- Weight distribution histogram from CSV
- Weight convergence from CSV
- Curve iteration history from CSV

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

# Import generic base plot functions from parent plotting module
from ..summary_gpr_plotting import (
    load_summary_gpr_csv,
    load_weight_history_csv,
    load_curve_history_csv,
    load_converged_weights_csv,
    plot_summary_gpr,
    plot_weight_convergence,
    plot_weight_distribution,
    plot_summary_curve_iterations,
)
from ..summary_gpr_loader import IndividualGPRData
from scipy.interpolate import interp1d
from scipy.stats import norm as sp_norm

if TYPE_CHECKING:
    from ..summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult

try:
    from ...pip0_dataloading.filename_parser import format_group_key_title
except Exception:
    def format_group_key_title(key: str) -> str:
        return key


# =============================================================================
# CSV Wrapper Functions (moved from summary_gpr_plotting.py)
# =============================================================================

def plot_summary_gpr_from_csv(
    summary_csv_path: Path,
    output_directory: Path,
    *,
    individual_gpr_csvs: Optional[List[Path]] = None,
    group_key: Optional[str] = None,
    confidence_level: float = 0.75,
    plot_individuals: bool = True,
    individual_alpha: float = 0.20,
    min_time_cap: float = 0.01,
    max_time_cap: float = 1e4,
    x_axis_label: str = "X_label",
    y_axis_label: str = "Y_label",
    x_is_log: bool = True,
    verbose: bool = True,
) -> Path:
    """
    CSV wrapper: Plot summary GPR with optional individual curves overlay.
    
    Loads data from CSV files and calls the core plot_summary_gpr function.
    Also generates a variance comparison plot (real vs normalized) if both
    variance types are present in the CSV.
    """
    summary_csv_path = Path(summary_csv_path)

    # Load summary data
    df = load_summary_gpr_csv(summary_csv_path)

    # Extract group_key from filename if not provided
    if group_key is None:
        stem = summary_csv_path.stem
        if stem.startswith('Summary_GPR_'):
            group_key = stem[len('Summary_GPR_'):]
        else:
            group_key = stem

    # Extract data as numpy arrays
    x_display = df['x_real'].to_numpy()
    y_mean = df['y_real'].to_numpy()
    if 'Upper_CI_real' in df.columns and 'Lower_CI_real' in df.columns:
        y_upper = df['Upper_CI_real'].to_numpy()
        y_lower = df['Lower_CI_real'].to_numpy()
    elif 'Upper_CI_normalised' in df.columns and 'Lower_CI_normalised' in df.columns:
        y_upper = df['Upper_CI_normalised'].to_numpy()
        y_lower = df['Lower_CI_normalised'].to_numpy()
    else:
        y_upper = np.full_like(y_mean, np.nan, dtype=float)
        y_lower = np.full_like(y_mean, np.nan, dtype=float)

    # Load individual curves from CSVs if provided
    individual_curves: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    if plot_individuals and individual_gpr_csvs:
        individual_curves = []
        for ind_csv in individual_gpr_csvs:
            try:
                ind_df = pd.read_csv(ind_csv)
                if 'x_pred_transformed' in ind_df.columns and 'y_pred' in ind_df.columns:
                    x_ind = np.exp(ind_df['x_pred_transformed'].to_numpy()) if x_is_log else ind_df['x_pred_transformed'].to_numpy()
                    y_ind = ind_df['y_pred'].to_numpy()
                    individual_curves.append((np.asarray(x_ind), np.asarray(y_ind)))
            except Exception:
                pass  # Skip if can't read

    # Build output path
    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Summary_GPR_{safe_key}.png'

    # Call core function
    main_plot_path = plot_summary_gpr(
        x_display=x_display,
        y_mean=y_mean,
        y_lower=y_lower,
        y_upper=y_upper,
        output_path=output_path,
        individual_curves=individual_curves,
        group_key=group_key,
        confidence_level=confidence_level,
        individual_alpha=individual_alpha,
        min_time_cap=min_time_cap,
        max_time_cap=max_time_cap,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        verbose=verbose,
    )

    # Optional variance comparison plot (real vs normalized) if available
    has_real = {'Upper_CI_real', 'Lower_CI_real'}.issubset(df.columns)
    has_norm = {'Upper_CI_normalised', 'Lower_CI_normalised'}.issubset(df.columns)
    if has_real and has_norm:
        y_upper_real = np.asarray(df['Upper_CI_real'].to_numpy())
        y_lower_real = np.asarray(df['Lower_CI_real'].to_numpy())
        y_upper_norm = np.asarray(df['Upper_CI_normalised'].to_numpy())
        y_lower_norm = np.asarray(df['Lower_CI_normalised'].to_numpy())

        if not (np.all(np.isnan(y_upper_real)) and np.all(np.isnan(y_upper_norm))):
            compare_path = Path(output_directory) / f'Summary_GPR_{safe_key}_variance_compare.png'
            plt.figure(figsize=(10, 6))

            # Plot individual curves if provided
            if individual_curves:
                for x_ind, y_ind in individual_curves:
                    plt.plot(x_ind, y_ind, color='gray', alpha=individual_alpha, linewidth=1)

            # Mean curve
            plt.plot(x_display, y_mean, 'b-', linewidth=2, label='Summary GPR')

            # Real variance band
            plt.fill_between(
                x_display, y_lower_real, y_upper_real,
                color='orange', alpha=0.20, label='CI (real variance)'
            )

            # Normalized variance band
            plt.fill_between(
                x_display, y_lower_norm, y_upper_norm,
                color='green', alpha=0.20, label='CI (normalized variance)'
            )

            # Format title from group_key
            plt.title(f'Summary GPR Variance Compare: {format_group_key_title(group_key)}')

            plt.xlabel(x_axis_label)
            plt.ylabel(y_axis_label)
            plt.xscale('log')
            plt.xlim([min_time_cap, max_time_cap])
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(compare_path, dpi=300, bbox_inches='tight')
            plt.close()

            if verbose:
                print(f"Saved variance comparison plot to {compare_path}")

    return main_plot_path


def plot_weight_convergence_from_csv(
    csv_path: Path,
    output_directory: Path,
    *,
    group_key: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Optional[Path]:
    """
    CSV wrapper: Plot weight convergence from Weight_History CSV.
    
    Loads data from CSV and calls the core plot_weight_convergence function.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        if verbose:
            print(f"Weight history CSV not found: {csv_path}")
        return None

    df = load_weight_history_csv(csv_path)

    if len(df) < 2:
        return None

    # Extract group_key from filename if not provided
    if group_key is None:
        stem = csv_path.stem
        if stem.startswith('Weight_History_'):
            group_key = stem[len('Weight_History_'):]
        else:
            group_key = stem

    # Extract data as numpy arrays
    iterations = df['iteration'].to_numpy()
    weight_cols = [c for c in df.columns if c.startswith('weight_')]
    weight_history = df[weight_cols].to_numpy()  # Shape: (n_iterations, n_models)

    # Build output path
    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Weight_Convergence_{safe_key}.png'

    # Call core function
    return plot_weight_convergence(
        iterations=iterations,
        weight_history=weight_history,
        output_path=output_path,
        group_key=group_key,
        figsize=figsize,
        verbose=verbose,
    )


def plot_summary_curve_iterations_from_csv(
    csv_path: Path,
    output_directory: Path,
    *,
    group_key: Optional[str] = None,
    min_time_cap: float = 0.01,
    max_time_cap: float = 1e4,
    x_axis_label: str = "X_label",
    y_axis_label: str = "Y_label",
    x_is_log: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Optional[Path]:
    """
    CSV wrapper: Plot summary curve iterations from Curve_History CSV.
    
    Loads data from CSV and calls the core plot_summary_curve_iterations function.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        if verbose:
            print(f"Curve history CSV not found: {csv_path}")
        return None

    df = load_curve_history_csv(csv_path)

    # Extract group_key from filename if not provided
    if group_key is None:
        stem = csv_path.stem
        if stem.startswith('Curve_History_'):
            group_key = stem[len('Curve_History_'):]
        else:
            group_key = stem

    # Get x values and convert from log if needed
    x_transformed = df['x_transformed'].to_numpy()
    x_display = np.exp(x_transformed) if x_is_log else x_transformed

    # Get iteration columns
    iter_cols = [c for c in df.columns if c.startswith('iter_')]
    if len(iter_cols) < 1:
        return None

    curve_history = df[iter_cols].to_numpy()  # Shape: (n_points, n_iterations)

    # Build output path
    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Summary_GPR_Iterations_{safe_key}.png'

    # Call core function
    return plot_summary_curve_iterations(
        x_display=x_display,
        curve_history=curve_history,
        output_path=output_path,
        group_key=group_key,
        min_time_cap=min_time_cap,
        max_time_cap=max_time_cap,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        figsize=figsize,
        verbose=verbose,
    )


def plot_weight_distribution_from_csv(
    csv_path: Path,
    output_directory: Path,
    *,
    group_key: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    verbose: bool = True,
) -> Path:
    """
    CSV wrapper: Plot weight distribution from Converged_Weights CSV.
    
    Loads data from CSV and calls the core plot_weight_distribution function.
    """
    csv_path = Path(csv_path)

    df = load_converged_weights_csv(csv_path)

    # Extract group_key from filename if not provided
    if group_key is None:
        stem = csv_path.stem
        if stem.startswith('Converged_Weights_'):
            group_key = stem[len('Converged_Weights_'):]
        else:
            group_key = stem

    # Get weights - look for 'converged_weight' or 'weight' column
    if 'converged_weight' in df.columns:
        weights = df['converged_weight'].to_numpy()
    elif 'weight' in df.columns:
        weights = df['weight'].to_numpy()
    else:
        raise ValueError(f"No 'converged_weight' or 'weight' column found in {csv_path}")

    # Build output path
    safe_key = group_key.replace('|', '_').replace('=', '_').replace(' ', '_')
    output_path = Path(output_directory) / f'Weight_Distribution_{safe_key}.png'

    # Call core function
    return plot_weight_distribution(
        weights=weights,
        output_path=output_path,
        group_key=group_key,
        figsize=figsize,
        verbose=verbose,
    )


# =====================================================================
# Iterative Sigma Calibration Plot
# =====================================================================

def plot_iterative_sigma_calibration(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[List[IndividualGPRData]] = None,
    *,
    x_axis_label: str = "Time (s)",
    y_axis_label: str = "Current (A/cm²)",
    verbose: bool = True,
    dpi: int = 200,
) -> Optional[Path]:
    """
    Iterative sigma calibration: deviation scatter with 1σ/2σ bands.

    Produces a two-panel figure:
      Left:  real scale
      Right: normalized scale

    Companion CSV is ``Sigma_Calibration_{key}.csv`` produced by
    ``export_iterative_sigma_calibration_csv``.
    """
    if result.y_std_real is None or gprs is None or len(gprs) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original
    mean_r = result.y_mean
    std_r = result.y_std_real

    # Interpolate individual curves (real)
    y_real_stack = []
    for gpr in gprs:
        f_y = interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred, kind='linear', fill_value='extrapolate',
        )
        y_real_stack.append(f_y(x))
    y_real_arr = np.vstack(y_real_stack)

    # Coverage (real)
    dev_real = y_real_arr - mean_r
    cov1_r = float(np.mean((dev_real >= -std_r) & (dev_real <= std_r)))
    cov2_r = float(np.mean((dev_real >= -2 * std_r) & (dev_real <= 2 * std_r)))

    # --- Determine layout ---
    mean_n = getattr(result, 'y_mean_norm', None)
    std_n = getattr(result, 'y_std_norm', None)
    has_norm = mean_n is not None and std_n is not None
    n_panels = 2 if has_norm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    try:
        title_str = format_group_key_title(result.group_key)
    except Exception:
        title_str = result.group_key

    # ----- Panel 1: Real -----
    ax0 = axes[0]
    ax0.axhline(0, color='k', linewidth=1.0, linestyle='--')
    ax0.fill_between(x, -2 * std_r, 2 * std_r, color='C0', alpha=0.12, label='Iterative 2σ')
    ax0.fill_between(x, -std_r, std_r, color='C0', alpha=0.25, label='Iterative 1σ')
    ax0.scatter(np.tile(x, len(y_real_stack)), dev_real.flatten(),
                s=6, alpha=0.08, color='gray', label='Samples')
    ax0.set_xscale('log')
    ax0.set_xlabel(x_axis_label)
    ax0.set_ylabel(f"Deviation ({y_axis_label})")
    ax0.set_title(f"Iterative σ calibration (real): {title_str}")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
    txt0 = [f"Coverage (real):", f"  1σ: {cov1_r*100:.1f}%", f"  2σ: {cov2_r*100:.1f}%"]
    ax0.text(0.02, 0.98, "\n".join(txt0), transform=ax0.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax0.legend(loc='best', fontsize='small')

    # ----- Panel 2: Normalized -----
    if has_norm:
        y_norm_stack = []
        for gpr in gprs:
            y_norm = getattr(gpr, 'y_pred_normalized', None)
            if y_norm is None:
                break
            f_y = interp1d(
                gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
                y_norm, kind='linear', fill_value='extrapolate',
            )
            y_norm_stack.append(f_y(x))

        if len(y_norm_stack) == len(gprs):
            y_norm_arr = np.vstack(y_norm_stack)
            dev_norm = y_norm_arr - mean_n
            cov1_n = float(np.mean((dev_norm >= -std_n) & (dev_norm <= std_n)))
            cov2_n = float(np.mean((dev_norm >= -2 * std_n) & (dev_norm <= 2 * std_n)))

            ax1 = axes[1]
            ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
            ax1.fill_between(x, -2 * std_n, 2 * std_n, color='C1', alpha=0.12, label='Iterative 2σ (norm)')
            ax1.fill_between(x, -std_n, std_n, color='C1', alpha=0.25, label='Iterative 1σ (norm)')
            ax1.scatter(np.tile(x, len(y_norm_stack)), dev_norm.flatten(),
                        s=6, alpha=0.08, color='gray', label='Samples (norm)')
            ax1.set_xscale('log')
            ax1.set_xlabel(x_axis_label)
            ax1.set_ylabel(f"Deviation ({y_axis_label}, norm)")
            ax1.set_title(f"Iterative σ calibration (norm): {title_str}")
            ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
            txt1 = [f"Coverage (norm):", f"  1σ: {cov1_n*100:.1f}%", f"  2σ: {cov2_n*100:.1f}%"]
            ax1.text(0.02, 0.98, "\n".join(txt1), transform=ax1.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            ax1.legend(loc='best', fontsize='small')

    output_path = output_dir / f'Sigma_Calibration_{safe_key}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved iterative sigma calibration to {output_path}")
    return output_path
