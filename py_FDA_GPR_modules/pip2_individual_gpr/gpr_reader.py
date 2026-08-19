# pip2_individual_gpr/gpr_reader.py
"""
GPR Result Reader Module - Reconstruct GPR results from exported CSVs.

This module provides a lossless way to read GPR results from CSV files,
enabling downstream processing without re-running the GPR fitting.

The exported CSVs contain:
1. Individual_GPR_*.csv - Mean and std predictions in normalized space
2. Covariance_Matrix_*.csv - Full posterior covariance in normalized space

Together, these provide all information needed to reconstruct the full
posterior distribution for curve aggregation (pip3).

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class ReconstructedPosterior:
    """
    Reconstructed posterior distribution from exported CSVs.
    
    All values are in user-defined normalized space (e.g., I/I_mid).
    Use accessor methods to convert to original physical units.
    
    Attributes
    ----------
    mean_normalized : np.ndarray
        Posterior mean in normalized space.
    std_normalized : np.ndarray
        Posterior std in normalized space.
    covariance_normalized : Optional[np.ndarray]
        Full posterior covariance in normalized space.
    physical_scale_factor : float
        Factor to convert from normalized to original units.
    """
    mean_normalized: np.ndarray
    std_normalized: np.ndarray
    covariance_normalized: Optional[np.ndarray] = None
    physical_scale_factor: float = 1.0
    
    def get_mean_original(self) -> np.ndarray:
        """Get posterior mean in original physical units (e.g., A/cm²)."""
        return self.mean_normalized * self.physical_scale_factor
    
    def get_std_original(self) -> np.ndarray:
        """Get posterior std in original physical units."""
        return self.std_normalized * abs(self.physical_scale_factor)
    
    def get_covariance_original(self) -> Optional[np.ndarray]:
        """Get covariance in original physical units."""
        if self.covariance_normalized is None:
            return None
        return self.covariance_normalized * (self.physical_scale_factor ** 2)
    
    def get_variance_normalized(self) -> np.ndarray:
        """Get variance (diagonal of covariance) in normalized space."""
        if self.covariance_normalized is not None:
            return np.diag(self.covariance_normalized)
        return self.std_normalized ** 2


@dataclass
class ReconstructedGPRResult:
    """
    Reconstructed GPR result from exported CSV files.
    
    This class provides the same interface as GPRFitResult but is
    constructed from exported CSVs rather than live GPR fitting.
    
    Attributes
    ----------
    sample_id : str
        Sample identifier.
    index_id : int
        Unique index for this curve.
    group_flags : Dict[str, Any]
        Generic grouping flags.
    x_pred_transformed : np.ndarray
        Prediction x points (transformed, e.g., log-time).
    x_pred_original : np.ndarray
        Prediction x points (original scale).
    posterior : ReconstructedPosterior
        Reconstructed posterior distribution.
    metadata : Dict[str, Any]
        All metadata from the CSV header.
    """
    sample_id: str
    index_id: int
    group_flags: Dict[str, Any]
    x_pred_transformed: np.ndarray
    x_pred_original: Optional[np.ndarray]
    posterior: ReconstructedPosterior
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def y_pred_normalized(self) -> np.ndarray:
        """Get predicted y values in normalized space."""
        return self.posterior.mean_normalized
    
    @property
    def y_std_normalized(self) -> np.ndarray:
        """Get prediction uncertainty in normalized space."""
        return self.posterior.std_normalized
    
    @property
    def y_pred_original(self) -> np.ndarray:
        """Get predicted y values in original physical units."""
        return self.posterior.get_mean_original()
    
    @property
    def y_std_original(self) -> np.ndarray:
        """Get prediction uncertainty in original physical units."""
        return self.posterior.get_std_original()
    
    @property
    def covariance_normalized(self) -> Optional[np.ndarray]:
        """Get full posterior covariance in normalized space."""
        return self.posterior.covariance_normalized
    
    @property
    def physical_scale_factor(self) -> float:
        """Get the physical scaling factor (s_r from normalization)."""
        return self.posterior.physical_scale_factor


def load_gpr_result_csv(filepath: Union[Path, str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load a GPR result CSV file and parse its metadata header.
    
    Parameters
    ----------
    filepath : Path or str
        Path to the Individual_GPR_*.csv file.
        
    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame]
        - metadata: Dictionary containing all parsed metadata fields
        - df: DataFrame with the prediction data
    """
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
                    elif key in ('index_id', 'num_prediction_points'):
                        try:
                            val = int(val)
                        except ValueError:
                            pass
                    elif key in ('validation_mae', 'validation_rmse', 'physical_scale_factor',
                                 'statistical_scaler_mean', 'statistical_scaler_std'):
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


def load_covariance_csv(filepath: Union[Path, str]) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Load a covariance matrix CSV file.
    
    Parameters
    ----------
    filepath : Path or str
        Path to the Covariance_Matrix_*.csv file.
        
    Returns
    -------
    Tuple[Dict[str, Any], np.ndarray]
        - metadata: Dictionary containing matrix metadata
        - covariance: Full covariance matrix as numpy array
    """
    filepath = Path(filepath)
    metadata = {}
    header_lines = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines += 1
                if ':' in line:
                    key_val = line[2:].strip()
                    colon_idx = key_val.find(':')
                    if colon_idx > 0:
                        key = key_val[:colon_idx].strip()
                        val = key_val[colon_idx + 1:].strip()
                        metadata[key] = val
            else:
                break
    
    # Read covariance matrix
    df = pd.read_csv(filepath, skiprows=header_lines, index_col=0)
    covariance = df.values
    
    return metadata, covariance


def reconstruct_gpr_result(
    gpr_csv_path: Union[Path, str],
    covariance_csv_path: Optional[Union[Path, str]] = None,
    auto_find_covariance: bool = True,
) -> ReconstructedGPRResult:
    """
    Reconstruct a complete GPR result from exported CSV files.
    
    This provides lossless reconstruction of the GPR posterior distribution
    at the user-defined normalized scale.
    
    Parameters
    ----------
    gpr_csv_path : Path or str
        Path to the Individual_GPR_*.csv file.
    covariance_csv_path : Path or str, optional
        Path to the Covariance_Matrix_*.csv file.
        If None and auto_find_covariance=True, will search for matching file.
    auto_find_covariance : bool
        If True, automatically search for matching covariance file.
        
    Returns
    -------
    ReconstructedGPRResult
        Complete reconstructed GPR result with posterior.
        
    Example
    -------
    >>> result = reconstruct_gpr_result("Individual_GPR_sample1.csv")
    >>> print(result.y_pred_normalized)  # Normalized space
    >>> print(result.y_pred_original)     # Original units
    >>> print(result.covariance_normalized.shape)  # Full covariance
    """
    gpr_csv_path = Path(gpr_csv_path)
    
    # Load main GPR result
    metadata, df = load_gpr_result_csv(gpr_csv_path)
    
    # Extract data
    x_pred_transformed = df['x_pred_transformed'].values
    x_pred_original = df['x_pred_original'].values if 'x_pred_original' in df else None
    
    # Handle both old and new column names
    if 'y_pred_normalized' in df:
        y_pred_normalized = df['y_pred_normalized'].values
        y_std_normalized = df['y_std_normalized'].values
    else:
        # Fallback for older format
        y_pred_normalized = df['y_pred'].values
        y_std_normalized = df['y_std'].values
    
    # Get physical scale factor
    physical_scale_factor = metadata.get('physical_scale_factor', 1.0)
    
    # Try to load covariance matrix
    covariance = None
    if covariance_csv_path is not None:
        _, covariance = load_covariance_csv(covariance_csv_path)
    elif auto_find_covariance:
        # Search for matching covariance file
        sample_id = metadata.get('sample_id', '')
        cov_filename = f"Covariance_Matrix_{sample_id}.csv"
        cov_path = gpr_csv_path.parent / cov_filename
        if cov_path.exists():
            _, covariance = load_covariance_csv(cov_path)
    
    # Construct posterior
    posterior = ReconstructedPosterior(
        mean_normalized=y_pred_normalized,
        std_normalized=y_std_normalized,
        covariance_normalized=covariance,
        physical_scale_factor=physical_scale_factor,
    )
    
    # Construct result
    result = ReconstructedGPRResult(
        sample_id=metadata.get('sample_id', 'unknown'),
        index_id=metadata.get('index_id', 0),
        group_flags=metadata.get('group_flags', {}),
        x_pred_transformed=x_pred_transformed,
        x_pred_original=x_pred_original,
        posterior=posterior,
        metadata=metadata,
    )
    
    return result


def reconstruct_all_gpr_results(
    directory: Union[Path, str],
    pattern: str = "Individual_GPR_*.csv",
) -> List[ReconstructedGPRResult]:
    """
    Reconstruct all GPR results from a directory.
    
    Parameters
    ----------
    directory : Path or str
        Directory containing exported CSV files.
    pattern : str
        Glob pattern for GPR result files.
        
    Returns
    -------
    List[ReconstructedGPRResult]
        List of reconstructed GPR results.
    """
    directory = Path(directory)
    results = []
    
    for csv_file in sorted(directory.glob(pattern)):
        try:
            result = reconstruct_gpr_result(csv_file)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
    
    return results


def plot_reconstructed_gpr(
    result: ReconstructedGPRResult,
    output_path: Optional[Union[Path, str]] = None,
    show_covariance: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 100,
) -> None:
    """
    Plot reconstructed GPR result with full diagnostics.
    
    This recreates the same 6-panel diagnostic plot as the pipeline.
    
    Parameters
    ----------
    result : ReconstructedGPRResult
        Reconstructed GPR result.
    output_path : Path or str, optional
        Path to save the plot. If None, displays interactively.
    show_covariance : bool
        Whether to show full covariance matrix.
    figsize : Tuple[int, int]
        Figure size.
    dpi : int
        DPI for saved figure.
    """
    import matplotlib.pyplot as plt
    
    x = result.x_pred_transformed
    y_mean = result.y_pred_normalized
    y_std = result.y_std_normalized
    cov = result.covariance_normalized
    
    # Get axis labels from metadata
    y_scaling_method = result.metadata.get('y_scaling_method', 'normalized')
    x_scaling_method = result.metadata.get('x_scaling_method', 'transformed')
    
    y_label = f"Normalized ({y_scaling_method})"
    x_label = f"T (Seconds) ({x_scaling_method})"
    
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # =========================================================================
    # Plot 1: GPR Prediction (top-left)
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.plot(x, y_mean, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(x, y_mean - 2*y_std, y_mean + 2*y_std, 
                     alpha=0.3, color='blue', label='±2σ')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(f'GPR Prediction (from CSV)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 2: Full Covariance Matrix (top-right)
    # =========================================================================
    ax2 = axes[0, 1]
    if cov is not None:
        im = ax2.imshow(cov, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax2, label=f'Covariance ({y_label}²)')
        ax2.set_xlabel('Point index')
        ax2.set_ylabel('Point index')
        ax2.set_title(f'Full Covariance Matrix ({cov.shape[0]}×{cov.shape[1]})')
    else:
        ax2.text(0.5, 0.5, 'No covariance data', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Covariance Matrix (not available)')
    
    # =========================================================================
    # Plot 3: Correlation Decay (middle-left)
    # =========================================================================
    ax3 = axes[1, 0]
    if cov is not None:
        # Compute correlation from covariance
        std_outer = np.outer(y_std, y_std)
        std_outer[std_outer == 0] = 1  # Avoid division by zero
        corr = cov / std_outer
        
        mid_idx = len(x) // 2
        correlations = corr[mid_idx, :]
        distances = np.abs(x - x[mid_idx])
        
        ax3.scatter(distances, correlations, alpha=0.5, s=15, c='steelblue')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel(f'Distance from midpoint ({x_label})')
        ax3.set_ylabel('Correlation')
        ax3.set_title('Correlation Decay with Distance')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No covariance data', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=14)
    
    # =========================================================================
    # Plot 4: Variance Along Curve (middle-right)
    # =========================================================================
    ax4 = axes[1, 1]
    variance = y_std ** 2
    ax4.plot(x, variance, 'g-', linewidth=2)
    ax4.set_xlabel(x_label)
    ax4.set_ylabel(f'Variance ({y_label}²)')
    ax4.set_title('Posterior Variance Along Curve')
    ax4.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 5: Verification - Compare variance from std vs covariance diagonal
    # =========================================================================
    ax5 = axes[2, 0]
    if cov is not None:
        cov_diag = np.diag(cov)
        ax5.plot(x, variance, 'g-', linewidth=2, label='From σ²')
        ax5.plot(x, cov_diag, 'r--', linewidth=2, label='From diag(C)')
        ax5.set_xlabel(x_label)
        ax5.set_ylabel(f'Variance ({y_label}²)')
        ax5.set_title('Variance Consistency Check')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)
        
        # Compute error
        max_diff = np.max(np.abs(variance - cov_diag))
        ax5.text(0.02, 0.98, f'Max diff: {max_diff:.2e}', 
                 transform=ax5.transAxes, va='top', fontsize=10)
    else:
        ax5.plot(x, variance, 'g-', linewidth=2)
        ax5.set_xlabel(x_label)
        ax5.set_ylabel(f'Variance ({y_label}²)')
        ax5.set_title('Posterior Variance (no covariance to compare)')
        ax5.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 6: Std comparison (bottom-right)
    # =========================================================================
    ax6 = axes[2, 1]
    ax6.plot(x, y_std, 'purple', linewidth=2, label='Predicted σ')
    ax6.set_xlabel(x_label)
    ax6.set_ylabel(f'Standard Deviation ({y_label})')
    ax6.set_title('Pointwise Uncertainty')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # Add title
    plt.suptitle(f'Reconstructed GPR: {result.sample_id}', fontsize=11)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def verify_reconstruction(
    original_result: Any,  # GPRFitResult from pipeline
    reconstructed_result: ReconstructedGPRResult,
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """
    Verify that reconstructed result matches original pipeline result.
    
    Parameters
    ----------
    original_result : GPRFitResult
        Original result from the pipeline.
    reconstructed_result : ReconstructedGPRResult
        Reconstructed result from CSVs.
    tolerance : float
        Tolerance for numerical comparisons.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with verification results.
    """
    results = {
        'sample_id_match': original_result.sample_id == reconstructed_result.sample_id,
        'x_pred_match': False,
        'y_mean_match': False,
        'y_std_match': False,
        'covariance_match': False,
        'all_passed': False,
    }
    
    # Compare x_pred (flatten both to handle shape differences)
    x_diff = np.max(np.abs(
        original_result.x_pred_transformed.flatten() - reconstructed_result.x_pred_transformed.flatten()
    ))
    results['x_pred_match'] = x_diff < tolerance
    results['x_pred_max_diff'] = x_diff
    
    # Compare y_mean in normalized space
    original_mean = original_result.posterior.get_mean_normalized()
    y_diff = np.max(np.abs(original_mean - reconstructed_result.y_pred_normalized))
    results['y_mean_match'] = y_diff < tolerance
    results['y_mean_max_diff'] = y_diff
    
    # Compare y_std in normalized space
    original_std = original_result.posterior.get_std_normalized()
    std_diff = np.max(np.abs(original_std - reconstructed_result.y_std_normalized))
    results['y_std_match'] = std_diff < tolerance
    results['y_std_max_diff'] = std_diff
    
    # Compare covariance if available
    if (original_result.posterior.covariance is not None and 
        reconstructed_result.covariance_normalized is not None):
        original_cov = original_result.posterior.get_covariance_normalized()
        cov_diff = np.max(np.abs(original_cov - reconstructed_result.covariance_normalized))
        results['covariance_match'] = cov_diff < tolerance
        results['covariance_max_diff'] = cov_diff
    elif (original_result.posterior.covariance is None and 
          reconstructed_result.covariance_normalized is None):
        results['covariance_match'] = True
        results['covariance_max_diff'] = 0.0
    
    # Overall pass/fail
    results['all_passed'] = all([
        results['sample_id_match'],
        results['x_pred_match'],
        results['y_mean_match'],
        results['y_std_match'],
        results['covariance_match'],
    ])
    
    return results
