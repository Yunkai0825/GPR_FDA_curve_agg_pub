# pip2_individual_gpr/gpr_utilities.py
"""
Utility functions for Individual GPR processing.

This module provides helper functions for:
- Grouping curves by potential
- Saving summary/skipped results
- Plotting individual GPR fits

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Callable, Tuple
from collections import defaultdict

from .gpr_config import ExportCfg


def group_curves_by_primary_key(
    curves: List[Any],
    primary_key: str = "",
    round_digits: int = 2,
) -> Dict[Any, List[Any]]:
    """
    Group curves by their primary grouping key value.
    
    Works with both PreprocessedCurve objects (via group_flags) and raw dictionaries.
    
    Parameters
    ----------
    curves : List[Any]
        List of curves (PreprocessedCurve or dict with group_flags).
    primary_key : str
        Name of the primary key to group by. If empty, uses first key in group_flags.
    round_digits : int
        Number of decimal places for rounding numeric values.
        
    Returns
    -------
    Dict[Any, List[Any]]
        Curves grouped by primary key value.
        
    Example
    -------
    >>> from py_FDA_GPR_modules.pip2_individual_gpr import group_curves_by_primary_key
    >>> grouped = group_curves_by_primary_key(preprocessed.curves, primary_key="potential")
    >>> for value, curves in grouped.items():
    ...     print(f"{value}: {len(curves)} curves")
    """
    curves_by_key: Dict[Any, List[Any]] = defaultdict(list)
    
    for curve in curves:
        # Get group_flags
        if hasattr(curve, 'group_flags'):
            group_flags = curve.group_flags
        else:
            group_flags = curve.get('group_flags', {})
        
        # Determine primary key value
        if primary_key and primary_key in group_flags:
            value = group_flags[primary_key]
        elif group_flags:
            # Use first key's value
            value = next(iter(group_flags.values()))
        else:
            value = None
        
        # Round numeric values
        if isinstance(value, (int, float)):
            value = round(value, round_digits)
        
        curves_by_key[value].append(curve)
    
    return dict(curves_by_key)


def group_curves_by_key(
    curves: List[Any],
    key_extractor: Callable[[Any], str],
) -> Dict[str, List[Any]]:
    """
    Group curves by a generic key extracted via a callable.
    
    Parameters
    ----------
    curves : List[Any]
        List of curve objects.
    key_extractor : Callable[[Any], str]
        Function to extract grouping key from curve.
        
    Returns
    -------
    Dict[str, List[Any]]
        Curves grouped by extracted key.
        
    Example
    -------
    >>> def get_group_key(curve):
    ...     return f"potential={curve.group_flags.get('potential')}"
    >>> grouped = group_curves_by_key(curves, get_group_key)
    """
    curves_by_key: Dict[str, List[Any]] = defaultdict(list)
    for curve in curves:
        key = key_extractor(curve)
        curves_by_key[key].append(curve)
    return dict(curves_by_key)


def save_skipped_samples_summary(
    skipped_samples: List[Dict[str, Any]],
    total_processed: int,
    total_skipped: int,
    output_directory: Union[Path, str],
    verbose: bool = True,
) -> None:
    """
    Save a summary of skipped samples and reasons.
    
    Creates two files:
    - Skipped_Samples_Details.csv: Full details of each skipped sample
    - Skipped_Samples_Summary.csv: Counts by skip reason
    
    Parameters
    ----------
    skipped_samples : List[Dict]
        List of skipped sample info dicts with keys:
        'sample_id', 'potential', 'reason'.
    total_processed : int
        Total number of samples processed.
    total_skipped : int
        Total number of samples skipped.
    output_directory : Path or str
        Directory to save CSV files.
    verbose : bool
        Whether to print status messages.
        
    Example
    -------
    >>> save_skipped_samples_summary(
    ...     gpr_result.skipped,
    ...     total_processed=100,
    ...     total_skipped=5,
    ...     output_directory="/path/to/output"
    ... )
    """
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    if skipped_samples:
        # Save detailed skipped samples
        skipped_df = pd.DataFrame(skipped_samples)
        details_file = output_dir / 'Skipped_Samples_Details.csv'
        skipped_df.to_csv(details_file, index=False)
        if verbose:
            print(f"Saved details of skipped samples to {details_file}")
        
        # Save summary by reason
        skip_reasons = skipped_df['reason'].value_counts().reset_index()
        skip_reasons.columns = ['Reason', 'Count']
        summary_file = output_dir / 'Skipped_Samples_Summary.csv'
        skip_reasons.to_csv(summary_file, index=False)
        if verbose:
            print(f"Saved summary of skipped samples to {summary_file}")
    else:
        if verbose:
            print("No samples were skipped.")
    
    if verbose:
        print(f"\nTotal number of processed samples: {total_processed}")
        print(f"Total number of skipped samples: {total_skipped}")


def plot_individual_gpr(
    gpr_result: Any,
    original_data_df: pd.DataFrame,
    output_directory: Union[Path, str],
    *,
    x_col: str = 'x_transformed',
    y_col: str = 'y_transformed',
    x_scaling: Optional[Any] = None,
    y_scaling: Optional[Any] = None,
    validation_x: Optional[np.ndarray] = None,
    validation_y: Optional[np.ndarray] = None,
    export_cfg: Optional[ExportCfg] = None,
    show_plot: bool = False,
) -> Optional[Path]:
    """
    Plot comprehensive GPR diagnostics including covariance analysis.
    
    All values are plotted in user-defined normalized space (NOT StandardScaler space).
    
    Creates a 3x2 figure with:
    1. GPR prediction with all data points (training + validation)
    2. Full covariance matrix heatmap (in normalized space)
    3. Correlation decay with distance from midpoint
    4. Posterior variance along curve (in normalized space)
    5. Standardized residuals (residuals / predicted std)
    6. Pointwise uncertainty comparison (in normalized space)
    
    Parameters
    ----------
    gpr_result : GPRFitResult
        Complete GPR fitting result with posterior.
    original_data_df : pd.DataFrame
        DataFrame with training data (columns: x_col, y_col).
    output_directory : Path or str
        Directory to save the plot.
    x_col, y_col : str
        Column names in original_data_df for x and y.
    x_scaling, y_scaling : ScalingInfo, optional
        Scaling info for axis labels.
    validation_x, validation_y : np.ndarray, optional
        Validation data points (if separate from training).
    export_cfg : ExportCfg, optional
        Export configuration (for DPI, etc.).
    show_plot : bool
        Whether to display interactively.
        
    Returns
    -------
    Optional[Path]
        Path to saved plot, or None if plotting failed.
    """
    if gpr_result.posterior is None:
        print(f"No posterior available for {gpr_result.sample_id}")
        return None
    
    cfg = export_cfg or ExportCfg()
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    posterior = gpr_result.posterior
    sample_id = gpr_result.sample_id
    
    # Get prediction data in NORMALIZED space (not StandardScaler space!)
    # This is the user's preprocessing normalization (e.g., middle_average, peak)
    x_pred = gpr_result.x_pred_transformed.flatten()
    y_pred = posterior.get_mean_normalized()  # Undo StandardScaler
    y_std = posterior.get_std_normalized()     # Undo StandardScaler
    
    # Get training data - already in normalized space
    x_train = original_data_df[x_col].values if x_col in original_data_df.columns else np.array([])
    y_train = original_data_df[y_col].values if y_col in original_data_df.columns else np.array([])
    
    # Build dynamic y-label based on y_scaling
    if y_scaling is not None and hasattr(y_scaling, 'method'):
        y_label = f"Normalized ({y_scaling.method})"
    elif cfg.y_transform_method:
        y_label = f"Normalized ({cfg.y_transform_method})"
    else:
        y_label = "Normalized current"
    
    # Build dynamic x-label based on x_scaling
    if x_scaling is not None and hasattr(x_scaling, 'method'):
        x_label = f"{cfg.x_col_name} ({x_scaling.method})"
    elif cfg.x_transform_method:
        x_label = f"{cfg.x_col_name} ({cfg.x_transform_method})"
    else:
        x_label = "log(time)"
    
    # Create figure with 4 rows, 2 columns (expanded for true calibration plot)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # =========================================================================
    # Plot 1: GPR Prediction with ALL data points (top-left)
    # =========================================================================
    ax1 = axes[0, 0]
    
    # Plot GPR prediction (in normalized space)
    ax1.plot(x_pred, y_pred, 'b-', label='Mean', linewidth=2)
    ax1.fill_between(x_pred, y_pred - 2*y_std, y_pred + 2*y_std, 
                      alpha=0.3, color='blue', label='±2σ')
    
    # Plot training data (also in normalized space)
    if len(x_train) > 0:
        ax1.scatter(x_train, y_train, c='black', s=15, alpha=0.6, 
                   label=f'Training ({len(x_train)} pts)', zorder=5)
    
    # Plot validation data if provided
    if validation_x is not None and validation_y is not None:
        ax1.scatter(validation_x, validation_y, c='red', s=20, alpha=0.7, 
                   marker='x', label=f'Validation ({len(validation_x)} pts)', zorder=6)
    
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title('GPR Prediction with Data Points')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Show optimized kernel hyperparameters (σ_f, ℓ, ν, σ_m) for quick reference
    hyperparams = getattr(gpr_result, 'hyperparams', {}) or {}

    def _first_present(keys):
        for k in keys:
            if k in hyperparams:
                try:
                    return float(hyperparams[k])
                except Exception:
                    return hyperparams[k]
        return None

    sigma_f = _first_present(['k1__k1__constant_value', 'k1__constant_value', 'constant_value'])
    length_scale = _first_present(['k1__k2__length_scale', 'k1__length_scale', 'length_scale'])
    nu = _first_present(['k1__k2__nu', 'k1__nu', 'nu'])
    sigma_m = _first_present(['k2__noise_level', 'noise_level'])

    lines = ["Hyperparams (opt)"]
    if sigma_f is not None:
        lines.append(f"σ_f: {sigma_f:.3g}")
    if length_scale is not None:
        lines.append(f"ℓ: {length_scale:.3g}")
    if nu is not None:
        try:
            lines.append(f"ν: {nu:g}")
        except Exception:
            lines.append(f"ν: {nu}")
    if sigma_m is not None:
        lines.append(f"σ_m: {sigma_m:.3g}")

    if len(lines) > 1:
        ax1.text(
            0.98,
            0.02,
            "\n".join(lines),
            transform=ax1.transAxes,
            fontsize=8,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            family='monospace',
        )
    
    # =========================================================================
    # Plot 2: Covariance Matrix Heatmap (top-right)
    # =========================================================================
    ax2 = axes[0, 1]
    
    # Get FULL covariance in normalized space (user-defined, not StandardScaler)
    cov = posterior.get_covariance_normalized()
    if cov is not None:
        im = ax2.imshow(cov, cmap='viridis', aspect='equal')
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label(f'Covariance ({y_label}²)')
        ax2.set_title(f'Full Covariance Matrix ({cov.shape[0]}×{cov.shape[1]})')
        ax2.set_xlabel('Point index')
        ax2.set_ylabel('Point index')
    else:
        ax2.text(0.5, 0.5, 'No covariance\nstored', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Covariance Matrix (not available)')
    
    # =========================================================================
    # Plot 3: Correlation Decay with Distance (middle-left)
    # =========================================================================
    ax3 = axes[1, 0]
    
    if cov is not None:
        mid_idx = len(x_pred) // 2
        std_all = np.sqrt(np.diag(cov))
        correlations = cov[mid_idx, :] / (std_all[mid_idx] * std_all + 1e-10)
        distances = np.abs(x_pred - x_pred[mid_idx])
        
        ax3.scatter(distances, correlations, alpha=0.5, s=15, c='steelblue')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel(f'Distance from midpoint ({x_label})')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Correlation Decay with Distance')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No covariance', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
    
    # =========================================================================
    # Plot 4: Posterior Variance Along Curve (middle-right)
    # =========================================================================
    ax4 = axes[1, 1]
    
    variance = y_std ** 2
    ax4.plot(x_pred, variance, 'g-', linewidth=2)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel(f'Variance ({y_label}²)')
    ax4.set_title('Posterior Variance Along Curve')
    ax4.grid(True, alpha=0.3)
    
    # Debug helper: print calibration stats to console for quick sanity checks
    def _debug_calibration(label: str, std_residuals: np.ndarray, pct1: float, pct2: float):
        if std_residuals is None or len(std_residuals) == 0:
            return
        # Basic moments to ensure we are seeing reasonable numbers
        res = std_residuals
        print(
            f"[CALIB DBG] {sample_id} | {label} | n={len(res)} | "
            f"within1={pct1:.2f}% within2={pct2:.2f}% | "
            f"res/sig mean={np.mean(res):.4f} std={np.std(res):.4f} "
            f"min={np.min(res):.3f} max={np.max(res):.3f} "
            f"median={np.median(res):.4f} mad={np.median(np.abs(res - np.median(res))):.4f}"
        )

    # =========================================================================
    # Plot 5: Standardized Residuals at Training Points (bottom-left)
    # NOTE: This uses the REFITTED GPR model which was trained on all data.
    # At training points, latent variance ≈ 0, so σ ≈ √(obs_noise).
    # This shows model fit quality, not true predictive calibration.
    # =========================================================================
    ax5 = axes[2, 0]
    
    # Store residuals for use in plot 6
    residuals = None
    
    if len(x_train) > 0 and gpr_result.gpr_model is not None:
        # Predict at training points
        scaler_X = gpr_result.scaler_X
        scaler_y = gpr_result.scaler_y
        
        if scaler_X is not None and scaler_y is not None:
            X_train_scaled = scaler_X.transform(x_train.reshape(-1, 1))
            y_train_pred_scaled, y_train_std_scaled = gpr_result.gpr_model.predict(
                X_train_scaled, return_std=True
            )
            
            # Convert back to normalized (not original) space
            # sklearn's predict() returns posterior std of latent f(x).
            # Only add observation noise alpha (sigma_f^2), NOT WhiteKernel.
            # sklearn predict(return_std=True) already includes WhiteKernel
            # (sigma_m^2) through the kernel diagonal K(X*,X*).
            # Adding WhiteKernel here would double-count model noise.
            obs_noise = float(getattr(gpr_result.gpr_model, 'alpha', 0.0) or 0.0)

            y_train_pred = y_train_pred_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]
            y_train_std = y_train_std_scaled * scaler_y.scale_[0]
            # NOTE: obs_noise is alpha = sigma_f^2 (observation noise VARIANCE).
            # To convert from scaled-y space to normalized-y space, multiply by scale^2.
            if obs_noise and obs_noise > 0.0:
                y_train_std = np.sqrt(y_train_std ** 2 + obs_noise * (scaler_y.scale_[0] ** 2))
            
            # Compute standardized residuals: (y_observed - y_predicted) / posterior_std
            residuals = y_train - y_train_pred
            standardized_residuals = residuals / (y_train_std + 1e-10)
            
            # Scatter plot with color based on magnitude
            scatter = ax5.scatter(x_train, standardized_residuals, 
                                  c=np.abs(standardized_residuals), cmap='RdYlGn_r',
                                  s=20, alpha=0.7, vmin=0, vmax=3)
            ax5.axhline(y=0, color='k', linestyle='-', alpha=0.8)
            ax5.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='±1σ')
            ax5.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
            ax5.axhline(y=2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
            ax5.axhline(y=-2, color='orange', linestyle=':', alpha=0.5)
            
            # Calculate counts and percentages for calibration statistics
            n_total = len(standardized_residuals)
            n_within_1sig = np.sum(np.abs(standardized_residuals) <= 1)
            n_within_2sig = np.sum(np.abs(standardized_residuals) <= 2)
            n_outside_1sig = n_total - n_within_1sig
            n_outside_2sig = n_total - n_within_2sig
            
            pct_within_1sig = n_within_1sig / n_total * 100
            pct_within_2sig = n_within_2sig / n_total * 100
            pct_outside_1sig = n_outside_1sig / n_total * 100
            pct_outside_2sig = n_outside_2sig / n_total * 100
            
            # Expected values for well-calibrated Gaussian: 68.27% within ±1σ, 95.45% within ±2σ
            expected_1sig = 68.27
            expected_2sig = 95.45
            
            ax5.set_xlabel(x_label)
            ax5.set_ylabel('Standardized Residual (r/σ)')
            ax5.set_title(f'Refitted Model Residuals ({pct_within_1sig:.0f}% in ±1σ, {pct_within_2sig:.0f}% in ±2σ)')
            ax5.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax5, label='|residual/σ|')
            
            # Add annotation box with calibration statistics
            stats_text = (
                f"Training points: {n_total}\n"
                f"─────────────────\n"
                f"Within ±1σ: {n_within_1sig}/{n_total} ({pct_within_1sig:.1f}%)\n"
                f"  Expected: {expected_1sig:.1f}%\n"
                f"Outside ±1σ: {n_outside_1sig}/{n_total} ({pct_outside_1sig:.1f}%)\n"
                f"─────────────────\n"
                f"Within ±2σ: {n_within_2sig}/{n_total} ({pct_within_2sig:.1f}%)\n"
                f"  Expected: {expected_2sig:.1f}%\n"
                f"Outside ±2σ: {n_outside_2sig}/{n_total} ({pct_outside_2sig:.1f}%)"
            )
            
            # Position the text box in upper right
            props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
            ax5.text(0.98, 0.98, stats_text, transform=ax5.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props, family='monospace')

            # Console debug for training residuals
            _debug_calibration(
                label="Train pts (refit)",
                std_residuals=standardized_residuals,
                pct1=pct_within_1sig,
                pct2=pct_within_2sig,
            )
    else:
        ax5.text(0.5, 0.5, 'No training data', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=14)
    
    # =========================================================================
    # Plot 6: Pointwise Uncertainty (Std) Along Curve (bottom-right)
    # =========================================================================
    ax6 = axes[2, 1]
    
    # Plot predicted std
    ax6.plot(x_pred, y_std, 'purple', linewidth=2, label='Predicted σ')
    
    # If we have validation residuals, show actual local error
    if len(x_train) > 0 and gpr_result.gpr_model is not None and scaler_X is not None:
        # Bin training residuals and compute local std
        n_bins = min(20, len(x_train) // 3)
        if n_bins >= 3:
            bin_edges = np.linspace(x_train.min(), x_train.max(), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            local_stds = []
            
            for i in range(n_bins):
                mask = (x_train >= bin_edges[i]) & (x_train < bin_edges[i+1])
                if np.sum(mask) > 1:
                    local_stds.append(np.std(residuals[mask]))
                else:
                    local_stds.append(np.nan)
            
            # Plot binned empirical std
            valid_mask = ~np.isnan(local_stds)
            ax6.scatter(bin_centers[valid_mask], np.array(local_stds)[valid_mask], 
                       c='orange', s=50, marker='s', alpha=0.8, 
                       label='Empirical σ (binned)', zorder=5)
    
    ax6.set_xlabel(x_label)
    ax6.set_ylabel(f'Standard Deviation ({y_label})')
    ax6.set_title('Pointwise Uncertainty Comparison')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 7: TRUE Predictive Calibration (using training-only GPR at validation points)
    # This is the CORRECT calibration metric:
    # - Uses the training-only GPR (hasn't seen validation data)
    # - Predicts at validation points (out-of-sample)
    # - The uncertainty is the TRUE predictive uncertainty
    # =========================================================================
    ax7 = axes[3, 0]
    
    # Get training-only GPR and validation data from gpr_result
    gpr_train = getattr(gpr_result, 'gpr_model_train', None)
    x_val_data = getattr(gpr_result, 'x_val', None)
    y_val_data = getattr(gpr_result, 'y_val', None)
    
    # Use passed validation data if not stored in result
    if x_val_data is None and validation_x is not None:
        x_val_data = validation_x
    if y_val_data is None and validation_y is not None:
        y_val_data = validation_y
    
    if (gpr_train is not None and 
        x_val_data is not None and len(x_val_data) > 0 and
        y_val_data is not None and len(y_val_data) > 0):
        
        scaler_X = gpr_result.scaler_X
        scaler_y = gpr_result.scaler_y
        
        if scaler_X is not None and scaler_y is not None:
            # Predict at VALIDATION points using TRAINING-ONLY GPR
            X_val_scaled = scaler_X.transform(x_val_data.reshape(-1, 1))
            y_val_pred_scaled, y_val_std_scaled = gpr_train.predict(X_val_scaled, return_std=True)
            
            # Only add alpha (sigma_f^2), NOT WhiteKernel noise.
            # sklearn predict(return_std=True) already includes WhiteKernel
            # through K(X*,X*). See theory Eqn 2.19a-b.
            obs_noise = float(getattr(gpr_train, 'alpha', 0.0) or 0.0)
            
            # Convert to normalized space
            y_val_pred = y_val_pred_scaled * scaler_y.scale_[0] + scaler_y.mean_[0]
            y_val_std = y_val_std_scaled * scaler_y.scale_[0]
            # obs_noise is alpha = sigma_f^2 (VARIANCE), multiply by scale^2 to convert
            if obs_noise and obs_noise > 0.0:
                y_val_std = np.sqrt(y_val_std ** 2 + obs_noise * (scaler_y.scale_[0] ** 2))
            
            # Compute TRUE standardized residuals (out-of-sample!)
            true_residuals = y_val_data - y_val_pred
            true_std_residuals = true_residuals / (y_val_std + 1e-10)
            
            # Scatter plot
            scatter7 = ax7.scatter(x_val_data, true_std_residuals, 
                                   c=np.abs(true_std_residuals), cmap='RdYlGn_r',
                                   s=25, alpha=0.7, vmin=0, vmax=3)
            ax7.axhline(y=0, color='k', linestyle='-', alpha=0.8)
            ax7.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='±1σ')
            ax7.axhline(y=-1, color='r', linestyle='--', alpha=0.5)
            ax7.axhline(y=2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
            ax7.axhline(y=-2, color='orange', linestyle=':', alpha=0.5)
            
            # Calculate TRUE calibration statistics
            n_total_true = len(true_std_residuals)
            n_within_1sig_true = np.sum(np.abs(true_std_residuals) <= 1)
            n_within_2sig_true = np.sum(np.abs(true_std_residuals) <= 2)
            n_outside_1sig_true = n_total_true - n_within_1sig_true
            n_outside_2sig_true = n_total_true - n_within_2sig_true
            
            pct_within_1sig_true = n_within_1sig_true / n_total_true * 100
            pct_within_2sig_true = n_within_2sig_true / n_total_true * 100
            pct_outside_1sig_true = n_outside_1sig_true / n_total_true * 100
            pct_outside_2sig_true = n_outside_2sig_true / n_total_true * 100
            
            expected_1sig = 68.27
            expected_2sig = 95.45
            
            ax7.set_xlabel(x_label)
            ax7.set_ylabel('Standardized Residual (r/σ)')
            ax7.set_title(f'TRUE Calibration: Validation Data ({pct_within_1sig_true:.0f}% in ±1σ, {pct_within_2sig_true:.0f}% in ±2σ)')
            ax7.grid(True, alpha=0.3)
            plt.colorbar(scatter7, ax=ax7, label='|residual/σ|')
            
            # Add annotation box
            stats_text_true = (
                f"Validation points: {n_total_true}\n"
                f"Using: Train-only GPR → Val pts\n"
                f"─────────────────\n"
                f"Within ±1σ: {n_within_1sig_true}/{n_total_true} ({pct_within_1sig_true:.1f}%)\n"
                f"  Expected: {expected_1sig:.1f}%\n"
                f"Outside ±1σ: {n_outside_1sig_true}/{n_total_true} ({pct_outside_1sig_true:.1f}%)\n"
                f"─────────────────\n"
                f"Within ±2σ: {n_within_2sig_true}/{n_total_true} ({pct_within_2sig_true:.1f}%)\n"
                f"  Expected: {expected_2sig:.1f}%\n"
                f"Outside ±2σ: {n_outside_2sig_true}/{n_total_true} ({pct_outside_2sig_true:.1f}%)"
            )
            
            props = dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8)
            ax7.text(0.98, 0.98, stats_text_true, transform=ax7.transAxes, fontsize=7,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props, family='monospace')

            # Console debug for validation residuals (true predictive calibration)
            _debug_calibration(
                label="Val pts (train-only)",
                std_residuals=true_std_residuals,
                pct1=pct_within_1sig_true,
                pct2=pct_within_2sig_true,
            )
        else:
            ax7.text(0.5, 0.5, 'No scalers available', ha='center', va='center', 
                    transform=ax7.transAxes, fontsize=14)
            pct_within_1sig_true, pct_within_2sig_true = np.nan, np.nan
    else:
        missing = []
        if gpr_train is None:
            missing.append("train-only GPR")
        if x_val_data is None or len(x_val_data) == 0:
            missing.append("validation data")
        ax7.text(0.5, 0.5, f'Missing: {", ".join(missing)}', ha='center', va='center', 
                transform=ax7.transAxes, fontsize=12)
        ax7.set_title('TRUE Calibration: Validation Data (N/A)')
        pct_within_1sig_true, pct_within_2sig_true = np.nan, np.nan
    
    # =========================================================================
    # Plot 8: Calibration Comparison Summary
    # Side-by-side bar chart comparing:
    # - Expected Gaussian calibration
    # - Refitted model at training points (σ ≈ obs_noise only)
    # - Training-only GPR at validation points (TRUE predictive calibration)
    # =========================================================================
    ax8 = axes[3, 1]
    
    # Prepare data for comparison
    categories = ['Within ±1σ', 'Within ±2σ']
    expected_vals = [68.27, 95.45]
    
    # Get refitted model values (from plot 5 if available)
    try:
        refitted_vals = [pct_within_1sig, pct_within_2sig]
    except NameError:
        refitted_vals = [np.nan, np.nan]
    
    # Get true predictive values (from plot 7)
    true_vals = [pct_within_1sig_true, pct_within_2sig_true]
    
    # Check if we have valid data
    has_refitted = not (np.isnan(refitted_vals[0]) and np.isnan(refitted_vals[1]))
    has_true = not (np.isnan(true_vals[0]) and np.isnan(true_vals[1]))
    
    if has_refitted or has_true:
        x_pos = np.arange(len(categories))
        width = 0.25
        
        bars1 = ax8.bar(x_pos - width, expected_vals, width, label='Expected (Gaussian)', 
                        color='gray', alpha=0.7, edgecolor='black')
        bars2 = ax8.bar(x_pos, refitted_vals, width, label='Refitted @ Train pts', 
                        color='wheat', alpha=0.9, edgecolor='black')
        bars3 = ax8.bar(x_pos + width, true_vals, width, label='Train-only @ Val pts', 
                        color='lightgreen', alpha=0.9, edgecolor='black')
        
        ax8.set_ylabel('Percentage (%)')
        ax8.set_title('Calibration Comparison')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(categories)
        ax8.legend(loc='upper left', fontsize=8)
        ax8.set_ylim(0, 105)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        def add_bar_labels(bars, vals):
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    height = bar.get_height()
                    ax8.annotate(f'{val:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=7)
        
        add_bar_labels(bars1, expected_vals)
        add_bar_labels(bars2, refitted_vals)
        add_bar_labels(bars3, true_vals)
        
        # Add explanation text
        explanation = (
            "Refitted @ Train pts:\n"
            "  GPR trained on ALL data, predict at train pts\n"
            "  → latent var≈0, σ≈√obs_noise (not meaningful)\n\n"
            "Train-only @ Val pts:\n"
            "  GPR trained on TRAIN only, predict at VAL pts\n"
            "  → TRUE out-of-sample predictive uncertainty"
        )
        ax8.text(0.98, 0.02, explanation, transform=ax8.transAxes, fontsize=6,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                family='monospace')
    else:
        ax8.text(0.5, 0.5, 'No calibration data available', ha='center', va='center', 
                transform=ax8.transAxes, fontsize=14)
        ax8.set_title('Calibration Comparison')

    # Add overall title
    group_str = ', '.join(f"{k}={v}" for k, v in gpr_result.group_flags.items()) if gpr_result.group_flags else 'No group'
    plt.suptitle(f'GPR Fit: {sample_id}\n{group_str}', fontsize=11)
    plt.tight_layout()
    
    # Save plot
    plot_filename = output_dir / f"GPR_Plot_{sample_id}.png"
    plt.savefig(plot_filename, dpi=cfg.dpi)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plot_filename


def export_gpr_result_to_csv(
    gpr_result: Any,
    group_key: str,
    output_directory: Union[Path, str],
    verbose: bool = True,
) -> Path:
    """
    Export individual GPR result to CSV file with comprehensive metadata header.
    
    The CSV file contains metadata lines at the top (prefixed with #) followed
    by the data columns. This allows downstream code to reconstruct all
    necessary information for operating on the GPR predictions.
    
    Parameters
    ----------
    gpr_result : GPRFitResult
        Complete GPR fitting result containing predictions, scaling info,
        validation metrics, and hyperparameters.
    group_key : str
        Grouping key for filename (e.g., "potential=-1.95").
    output_directory : Path or str
        Directory to save CSV.
    verbose : bool
        Whether to print status.
        
    Returns
    -------
    Path
        Path to the saved CSV file.
        
    Notes
    -----
    Metadata Header Format:
        # METADATA_START
        # sample_id: <value>
        # index_id: <value>
        # group_key: <value>
        # group_flags: <json>
        # x_scaling_type: <value>
        # x_scaling_params: <json>
        # y_scaling_type: <value>
        # y_scaling_params: <json>
        # validation_mae: <value>
        # validation_rmse: <value>
        # hyperparams_str: <value>
        # hyperparams: <json>
        # x_pred_units: transformed (log-time)
        # y_pred_units: original (A/cm2)
        # y_std_units: original (A/cm2)
        # METADATA_END
        x_pred,y_pred,y_std,...
        
    To load with metadata:
        Use load_gpr_csv_with_metadata() function.
    """
    import json
    
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract scaling info
    x_scaling_method = ""
    x_scaling_params = {}
    if gpr_result.x_scaling is not None:
        x_scaling_method = gpr_result.x_scaling.method
        x_scaling_params = gpr_result.x_scaling.params
    
    y_scaling_method = ""
    y_scaling_params = {}
    if gpr_result.y_scaling is not None:
        y_scaling_method = gpr_result.y_scaling.method
        y_scaling_params = gpr_result.y_scaling.params
    
    # Convert hyperparams to JSON-serializable format
    hyperparams_serializable = {}
    for k, v in gpr_result.hyperparams.items():
        try:
            json.dumps(v)  # Test if serializable
            hyperparams_serializable[k] = v
        except (TypeError, ValueError):
            hyperparams_serializable[k] = str(v)  # Convert to string
    
    # Build metadata lines - include all info needed for lossless reconstruction
    posterior = gpr_result.posterior
    physical_scale = posterior.physical_scale_factor if posterior else 1.0
    stat_scaler_mean = posterior.statistical_scaler_mean if posterior else 0.0
    stat_scaler_std = posterior.statistical_scaler_std if posterior else 1.0
    
    metadata_lines = [
        "# METADATA_START",
        f"# sample_id: {gpr_result.sample_id}",
        f"# index_id: {gpr_result.index_id}",
        f"# group_key: {group_key}",
        f"# group_flags: {json.dumps(gpr_result.group_flags)}",
        f"# x_scaling_method: {x_scaling_method}",
        f"# x_scaling_params: {json.dumps(x_scaling_params)}",
        f"# y_scaling_method: {y_scaling_method}",
        f"# y_scaling_params: {json.dumps(y_scaling_params)}",
        f"# physical_scale_factor: {physical_scale}",
        f"# statistical_scaler_mean: {stat_scaler_mean}",
        f"# statistical_scaler_std: {stat_scaler_std}",
        f"# validation_mae: {gpr_result.validation_mae}",
        f"# validation_rmse: {gpr_result.validation_rmse}",
        f"# hyperparams_str: {gpr_result.hyperparams_str}",
        f"# hyperparams: {json.dumps(hyperparams_serializable)}",
        f"# num_prediction_points: {len(gpr_result.x_pred_transformed)}",
        "# GRID_TYPE: shared_regulated",
        f"# grid_x_min: {float(gpr_result.x_pred_transformed.min())}",
        f"# grid_x_max: {float(gpr_result.x_pred_transformed.max())}",
        "# COORDINATE_SYSTEM: normalized_user_defined",
        "# x_pred_units: transformed (as per x_scaling_method)",
        "# y_pred_units: normalized (as per y_scaling_method, e.g., I/I_mid)",
        "# y_std_units: normalized (as per y_scaling_method)",
        "# NOTE: To get original units, multiply y by physical_scale_factor",
        "# NOTE: All curves in the same group share the same x_pred grid (J_R)",
        "# METADATA_END",
    ]
    
    # Prepare data in NORMALIZED space (user-defined scale, not StandardScaler)
    # This is lossless - original units can be recovered via physical_scale_factor
    x_pred = gpr_result.x_pred_transformed
    
    # Use normalized space values from posterior (properly scaled)
    if posterior is not None:
        y_pred_normalized = posterior.get_mean_normalized()
        y_std_normalized = posterior.get_std_normalized()
    else:
        # Fallback to transformed values if no posterior
        y_pred_normalized = gpr_result.y_pred_transformed
        y_std_normalized = gpr_result.y_std_transformed
    
    # Also include original-space x if available
    x_pred_original = gpr_result.x_pred
    
    df = pd.DataFrame({
        'x_pred_transformed': x_pred.flatten(),
        'x_pred_original': x_pred_original.flatten() if x_pred_original is not None else np.nan,
        'y_pred_normalized': y_pred_normalized,
        'y_std_normalized': y_std_normalized,
    })
    
    # Use sample_id directly - it already contains the full filename info
    # Group key is stored in metadata header for reference
    filename = output_dir / f"Individual_GPR_{gpr_result.sample_id}.csv"
    
    # Write metadata + data
    with open(filename, 'w', newline='') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)
    
    if verbose:
        print(f"Saved individual GPR results to {filename}")
    
    return filename


def export_diagnostic_data(
    gpr_result: Any,
    original_data_df: pd.DataFrame,
    output_directory: Union[Path, str],
    *,
    x_col: str = 'x_transformed',
    y_col: str = 'y_transformed',
    verbose: bool = True,
) -> Optional[Path]:
    """
    Export training & validation point-level data needed to redraw diagnostic
    plots 5-8 of the 8-panel individual GPR figure.

    Creates ``Diagnostic_Data_<sample_id>.csv`` containing:

    * **Training rows** – predictions of the *refitted* (all-data) GPR at each
      training point, plus standardized residuals  (→ plots 5, 6).
    * **Validation rows** – predictions of the *train-only* GPR at each
      held-out validation point, plus standardized residuals  (→ plot 7).
    * **Metadata header** – calibration percentages for both training-refit
      and true-validation scenarios  (→ plot 8).

    All y-values are in user-defined normalized space (e.g. I/I_mid), not
    StandardScaler space.

    Parameters
    ----------
    gpr_result : GPRFitResult
        GPR result *before* ``release_model_memory()`` has been called.
        Must still contain ``gpr_model``, ``gpr_model_train``, ``scaler_X``,
        ``scaler_y``, ``x_val``, ``y_val``.
    original_data_df : pd.DataFrame
        DataFrame with training data (columns *x_col*, *y_col*).
    output_directory : Path or str
        Directory to save the CSV.
    x_col, y_col : str
        Column names in *original_data_df*.
    verbose : bool
        Print status message.

    Returns
    -------
    Path or None
        Path to the saved CSV, or ``None`` if no data could be exported.
    """
    import json

    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)

    sample_id = gpr_result.sample_id
    scaler_X = gpr_result.scaler_X
    scaler_y = gpr_result.scaler_y

    if scaler_X is None or scaler_y is None:
        if verbose:
            print(f"  [Diagnostic] Skip {sample_id}: no scalers available")
        return None

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Training points  (refitted GPR → all data)
    # ------------------------------------------------------------------
    pct_within_1sig = float('nan')
    pct_within_2sig = float('nan')

    x_train = original_data_df[x_col].values if x_col in original_data_df.columns else np.array([])
    y_train = original_data_df[y_col].values if y_col in original_data_df.columns else np.array([])

    if len(x_train) > 0 and gpr_result.gpr_model is not None:
        X_train_scaled = scaler_X.transform(x_train.reshape(-1, 1))
        y_pred_sc, y_std_sc = gpr_result.gpr_model.predict(X_train_scaled, return_std=True)
        obs_noise = float(getattr(gpr_result.gpr_model, 'alpha', 0.0) or 0.0)

        y_pred_norm = y_pred_sc * scaler_y.scale_[0] + scaler_y.mean_[0]
        y_std_norm = y_std_sc * scaler_y.scale_[0]
        if obs_noise > 0.0:
            y_std_norm = np.sqrt(y_std_norm ** 2 + obs_noise * (scaler_y.scale_[0] ** 2))

        residuals = y_train - y_pred_norm
        std_res = residuals / (y_std_norm + 1e-10)

        n = len(std_res)
        pct_within_1sig = float(np.sum(np.abs(std_res) <= 1) / n * 100)
        pct_within_2sig = float(np.sum(np.abs(std_res) <= 2) / n * 100)

        for i in range(n):
            rows.append({
                'data_type': 'training',
                'x': x_train[i],
                'y_observed': y_train[i],
                'y_pred': y_pred_norm[i],
                'y_std': y_std_norm[i],
                'standardized_residual': std_res[i],
            })

    # ------------------------------------------------------------------
    # Validation points  (train-only GPR → held-out data)
    # ------------------------------------------------------------------
    pct_within_1sig_true = float('nan')
    pct_within_2sig_true = float('nan')

    gpr_train = getattr(gpr_result, 'gpr_model_train', None)
    x_val = getattr(gpr_result, 'x_val', None)
    y_val = getattr(gpr_result, 'y_val', None)

    if (gpr_train is not None and
            x_val is not None and len(x_val) > 0 and
            y_val is not None and len(y_val) > 0):
        X_val_scaled = scaler_X.transform(x_val.reshape(-1, 1))
        y_vp_sc, y_vs_sc = gpr_train.predict(X_val_scaled, return_std=True)
        obs_noise_t = float(getattr(gpr_train, 'alpha', 0.0) or 0.0)

        y_vp_norm = y_vp_sc * scaler_y.scale_[0] + scaler_y.mean_[0]
        y_vs_norm = y_vs_sc * scaler_y.scale_[0]
        if obs_noise_t > 0.0:
            y_vs_norm = np.sqrt(y_vs_norm ** 2 + obs_noise_t * (scaler_y.scale_[0] ** 2))

        true_res = y_val - y_vp_norm
        true_std_res = true_res / (y_vs_norm + 1e-10)

        nv = len(true_std_res)
        pct_within_1sig_true = float(np.sum(np.abs(true_std_res) <= 1) / nv * 100)
        pct_within_2sig_true = float(np.sum(np.abs(true_std_res) <= 2) / nv * 100)

        for i in range(nv):
            rows.append({
                'data_type': 'validation',
                'x': x_val[i],
                'y_observed': y_val[i],
                'y_pred': y_vp_norm[i],
                'y_std': y_vs_norm[i],
                'standardized_residual': true_std_res[i],
            })

    if not rows:
        if verbose:
            print(f"  [Diagnostic] Skip {sample_id}: no training/validation data")
        return None

    # ------------------------------------------------------------------
    # Metadata header
    # ------------------------------------------------------------------
    n_training = sum(1 for r in rows if r['data_type'] == 'training')
    n_validation = sum(1 for r in rows if r['data_type'] == 'validation')

    metadata_lines = [
        "# DIAGNOSTIC_METADATA_START",
        f"# sample_id: {sample_id}",
        f"# n_training: {n_training}",
        f"# n_validation: {n_validation}",
        f"# refitted_pct_within_1sig: {pct_within_1sig}",
        f"# refitted_pct_within_2sig: {pct_within_2sig}",
        f"# true_pct_within_1sig: {pct_within_1sig_true}",
        f"# true_pct_within_2sig: {pct_within_2sig_true}",
        f"# expected_1sig: 68.27",
        f"# expected_2sig: 95.45",
        "# COORDINATE_SYSTEM: normalized_user_defined",
        "# NOTE: data_type='training'  → refitted GPR (all data) predicted at training points  (plots 5,6)",
        "# NOTE: data_type='validation' → train-only GPR predicted at held-out validation points (plot 7)",
        "# NOTE: Calibration percentages in metadata header support plot 8",
        "# DIAGNOSTIC_METADATA_END",
    ]

    df = pd.DataFrame(rows)
    filename = output_dir / f"Diagnostic_Data_{sample_id}.csv"

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        df.to_csv(f, index=False)

    if verbose:
        print(f"  Saved diagnostic data to {filename}")

    return filename


def export_covariance_matrix(
    gpr_result: Any,
    output_directory: Union[Path, str],
    verbose: bool = True,
) -> Path:
    """
    Export full posterior covariance matrix C_g,post to CSV in user-defined normalized space.
    
    The covariance matrix is the GROUND TRUTH / LATENT function posterior
    covariance (C_g,post), converted from StandardScaler space to the user's 
    normalized space (e.g., middle_average, peak, log scaling) before export.
    
    For FGPR aggregation (pip3, Eqn 3.5a):
        C_e,r = C_post,r + σ_btw² I
    where C_post,r is exactly this exported matrix.
    
    With σ_f = 0 (observations treated as ground truth):
        C_f,post = C_g,post (no observation noise to add).
    
    Parameters
    ----------
    gpr_result : GPRFitResult
        Complete GPR fitting result containing posterior with covariance.
    output_directory : Path or str
        Directory to save CSV.
    verbose : bool
        Whether to print status.
        
    Returns
    -------
    Path
        Path to the saved CSV file.
        
    Notes
    -----
    The CSV file contains:
    - Header with metadata (sample_id, units, matrix dimensions)
    - Full covariance matrix with x_pred values as row/column indices
    
    The covariance is in user-defined normalized space (e.g., I/I_mid units squared).
    """
    import json
    
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if covariance exists
    posterior = gpr_result.posterior
    if posterior is None or posterior.covariance is None:
        if verbose:
            print(f"No covariance matrix available for {gpr_result.sample_id}")
        return None
    
    # Get covariance in normalized space (not StandardScaler space)
    cov = posterior.get_covariance_normalized()
    x_pred = gpr_result.x_pred_transformed
    
    # Get scaling info for metadata
    y_scaling_method = ""
    if gpr_result.y_scaling is not None:
        y_scaling_method = gpr_result.y_scaling.method
    
    x_scaling_method = ""
    if gpr_result.x_scaling is not None:
        x_scaling_method = gpr_result.x_scaling.method
    
    # Build metadata header
    metadata_lines = [
        "# COVARIANCE_MATRIX_EXPORT",
        f"# sample_id: {gpr_result.sample_id}",
        f"# matrix_shape: {cov.shape[0]}x{cov.shape[1]}",
        f"# y_scaling_method: {y_scaling_method}",
        f"# x_scaling_method: {x_scaling_method}",
        f"# units: normalized_y_units_squared",
        "# note: Covariance is in user-defined normalized space (post-y_scaling)",
        "# METADATA_END",
    ]
    
    # Create DataFrame with x_pred as index/column labels
    df = pd.DataFrame(cov, index=x_pred.flatten(), columns=x_pred.flatten())
    
    filename = output_dir / f"Covariance_Matrix_{gpr_result.sample_id}.csv"
    
    # Write metadata + data
    with open(filename, 'w', newline='') as f:
        for line in metadata_lines:
            f.write(line + '\n')
        df.to_csv(f)
    
    if verbose:
        print(f"Saved covariance matrix ({cov.shape[0]}x{cov.shape[1]}) to {filename}")
    
    return filename


def load_gpr_csv_with_metadata(filepath: Union[Path, str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load a GPR CSV file and parse its metadata header.
    
    Parameters
    ----------
    filepath : Path or str
        Path to the CSV file with metadata header.
        
    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        - metadata: Dictionary containing all parsed metadata fields
        - df: DataFrame with the prediction data
        
    Example
    -------
    >>> metadata, df = load_gpr_csv_with_metadata("Individual_GPR_sample1_potential_-1.95.csv")
    >>> print(metadata['sample_id'])
    'sample1'
    >>> print(metadata['y_scaling_params'])
    {'factor': 0.025, 'peak_value': -0.025}
    >>> print(df.columns.tolist())
    ['x_pred_transformed', 'x_pred_original', 'y_pred', 'y_std']
    """
    import json
    
    filepath = Path(filepath)
    metadata = {}
    header_lines = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines += 1
                if line.startswith('# METADATA_START') or line.startswith('# METADATA_END'):
                    continue
                # Parse key: value
                if ':' in line:
                    key_val = line[2:].strip()  # Remove '# '
                    colon_idx = key_val.find(':')
                    key = key_val[:colon_idx].strip()
                    val = key_val[colon_idx + 1:].strip()
                    
                    # Try to parse JSON for dict/list values
                    if key in ('group_flags', 'x_scaling_params', 'y_scaling_params', 'hyperparams'):
                        try:
                            val = json.loads(val)
                        except json.JSONDecodeError:
                            pass
                    # Try to parse numeric values
                    elif key in ('index_id',):
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    elif key in ('validation_mae', 'validation_rmse'):
                        try:
                            val = float(val)
                        except ValueError:
                            pass
                    
                    metadata[key] = val
            else:
                break
    
    # Read data portion (skip metadata lines)
    df = pd.read_csv(filepath, skiprows=header_lines)
    
    return metadata, df


def export_validation_results(
    validation_results: List[Dict[str, Any]],
    group_key: str,
    output_directory: Union[Path, str],
    verbose: bool = True,
) -> Path:
    """
    Export validation results for a group to CSV.
    
    Parameters
    ----------
    validation_results : List[Dict]
        List of validation result dictionaries.
    group_key : str
        Grouping key for filename.
    output_directory : Path or str
        Directory to save CSV.
    verbose : bool
        Whether to print status.
        
    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(validation_results)
    safe_group_key = group_key.replace('|', '_').replace('=', '_')
    filename = output_dir / f"Validation_Results_{safe_group_key}.csv"
    df.to_csv(filename, index=False)
    
    if verbose:
        print(f"Saved validation results to {filename}")
    
    return filename


# =============================================================================
# Posterior Covariance Visualization Utilities
# =============================================================================

def plot_posterior_covariance_diagnostics(
    gpr_result: Any,
    output_directory: Union[Path, str],
    *,
    sample_id: str = "",
    max_points_to_show: int = 100,
    show_plot: bool = False,
) -> Optional[Path]:
    """
    Plot diagnostic visualizations of the posterior covariance matrix.
    
    Creates visualizations to verify the covariance structure:
    1. Heatmap of covariance matrix (subset)
    2. Correlation decay with distance
    3. Variance along the curve
    
    Parameters
    ----------
    gpr_result : GPRFitResult
        Complete GPR result with posterior covariance.
    output_directory : Path or str
        Directory to save the plot.
    sample_id : str
        Sample identifier for filename.
    max_points_to_show : int
        Maximum points to show in heatmap (for performance).
    show_plot : bool
        Whether to display the plot interactively.
        
    Returns
    -------
    Optional[Path]
        Path to the saved plot, or None if no covariance available.
    """
    if gpr_result.posterior is None or gpr_result.posterior.covariance is None:
        print(f"No full covariance available for {sample_id}")
        return None
    
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    cov = gpr_result.posterior.covariance
    n = cov.shape[0]
    
    # Subsample if too large
    if n > max_points_to_show:
        indices = np.linspace(0, n-1, max_points_to_show, dtype=int)
        cov_subset = cov[np.ix_(indices, indices)]
    else:
        indices = np.arange(n)
        cov_subset = cov
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Covariance heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(cov_subset, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.abs(cov_subset).max(), vmax=np.abs(cov_subset).max())
    ax1.set_title(f'Posterior Covariance (n={len(indices)})')
    ax1.set_xlabel('Point index')
    ax1.set_ylabel('Point index')
    plt.colorbar(im1, ax=ax1, label='Covariance')
    
    # Plot 2: Correlation decay (first row of correlation matrix)
    ax2 = axes[1]
    std = np.sqrt(np.diag(cov))
    std_outer = np.outer(std, std)
    corr = cov / (std_outer + 1e-10)
    # Take correlation of first point with all others
    ax2.plot(np.arange(n), corr[0, :], 'b-', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Distance from first point')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Correlation Decay from First Point')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance along curve
    ax3 = axes[2]
    variance = np.diag(cov)
    x_pred = gpr_result.x_pred_transformed.flatten() if gpr_result.x_pred_transformed is not None else np.arange(n)
    ax3.plot(x_pred, variance, 'g-', linewidth=2)
    ax3.set_xlabel('x (transformed)')
    ax3.set_ylabel('Variance')
    ax3.set_title('Posterior Variance Along Curve')
    ax3.grid(True, alpha=0.3)
    
    sample_label = sample_id or gpr_result.sample_id
    plt.suptitle(f'Posterior Covariance Diagnostics: {sample_label}', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_filename = output_dir / f"Covariance_Diagnostics_{sample_label}.png"
    plt.savefig(plot_filename, dpi=150)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return plot_filename


def save_posterior_covariance(
    gpr_result: Any,
    output_directory: Union[Path, str],
    *,
    format: str = 'npz',
    sample_id: str = "",
) -> Optional[Path]:
    """
    Save posterior covariance matrix to file for later curve aggregation.
    
    Parameters
    ----------
    gpr_result : GPRFitResult
        Complete GPR result with posterior.
    output_directory : Path or str
        Directory to save the covariance.
    format : str
        Save format: 'npz' (compressed numpy), 'npy', or 'csv'.
    sample_id : str
        Sample identifier for filename.
        
    Returns
    -------
    Optional[Path]
        Path to the saved file, or None if no covariance available.
    """
    output_dir = Path(output_directory)
    os.makedirs(output_dir, exist_ok=True)
    
    sample_label = sample_id or gpr_result.sample_id
    
    if gpr_result.posterior is None:
        print(f"No posterior available for {sample_label}")
        return None
    
    posterior = gpr_result.posterior
    
    # Prepare data to save
    data = {
        'mean': posterior.mean,
        'std': posterior.std,
        'physical_scale_factor': posterior.physical_scale_factor,
        'statistical_scaler_mean': posterior.statistical_scaler_mean,
        'statistical_scaler_std': posterior.statistical_scaler_std,
        'covariance_mode': posterior.covariance_mode,
    }
    
    # Add x_pred for alignment in aggregation
    if gpr_result.x_pred_transformed is not None:
        data['x_pred_transformed'] = gpr_result.x_pred_transformed.flatten()
    
    # Add covariance based on storage mode
    if posterior.covariance is not None:
        data['covariance'] = posterior.covariance
    elif posterior.covariance_cholesky is not None:
        data['covariance_cholesky'] = posterior.covariance_cholesky
    elif posterior.covariance_sparse is not None:
        # Convert sparse to arrays for saving
        data['covariance_sparse_data'] = posterior.covariance_sparse.data
        data['covariance_sparse_indices'] = posterior.covariance_sparse.indices
        data['covariance_sparse_indptr'] = posterior.covariance_sparse.indptr
        data['covariance_sparse_shape'] = np.array(posterior.covariance_sparse.shape)
    
    # Save based on format
    if format == 'npz':
        filename = output_dir / f"Posterior_{sample_label}.npz"
        np.savez_compressed(filename, **data)
    elif format == 'npy':
        # Save each array separately
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(output_dir / f"Posterior_{sample_label}_{key}.npy", value)
        filename = output_dir / f"Posterior_{sample_label}_metadata.npy"
        np.save(filename, {k: v for k, v in data.items() if not isinstance(v, np.ndarray)})
    else:  # csv - only for diagonal mode
        filename = output_dir / f"Posterior_{sample_label}.csv"
        df = pd.DataFrame({
            'mean': posterior.mean,
            'std': posterior.std,
        })
        if gpr_result.x_pred_transformed is not None:
            df['x_pred'] = gpr_result.x_pred_transformed.flatten()
        df.to_csv(filename, index=False)
    
    return filename