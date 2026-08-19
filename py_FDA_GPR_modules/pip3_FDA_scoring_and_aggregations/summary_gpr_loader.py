# pip3_FDA_scoring_and_aggregations/summary_gpr_loader.py
"""
Loader for individual GPR CSV files with metadata parsing.

This module handles:
- Loading individual GPR predictions from CSV files
- Parsing metadata headers to extract scaling information
- Reconstructing ScalingInfo objects from metadata
- Grouping GPRs by common keys (e.g., potential)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict

# Import ScalingInfo from pip1
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pip1_datapreprocessing import ScalingInfo


@dataclass
class IndividualGPRData:
    """
    Data container for a single individual GPR prediction with metadata.
    
    Attributes
    ----------
    sample_id : str
        Unique identifier for the sample.
    index_id : int
        Numeric index within the dataset.
    group_key : str
        Grouping key (e.g., "potential=-1.95").
    group_flags : Dict[str, Any]
        Grouping flags (e.g., {"potential": -1.95}).
        
    x_pred_transformed : np.ndarray
        X values in transformed space (e.g., log-time).
    x_pred_original : np.ndarray
        X values in original space (e.g., seconds).
    y_pred : np.ndarray
        Predicted Y values in original scale.
    y_std : np.ndarray
        Standard deviation of predictions.
        
    x_scaling : ScalingInfo
        X-axis scaling information (reconstructed from metadata).
    y_scaling : ScalingInfo
        Y-axis scaling information (reconstructed from metadata).
        
    validation_mae : float
        Mean absolute error from validation.
    validation_rmse : float
        Root mean squared error from validation.
        
    filepath : Path
        Original file path.
    """
    sample_id: str
    index_id: int
    group_key: str
    group_flags: Dict[str, Any]
    
    x_pred_transformed: np.ndarray
    x_pred_original: np.ndarray
    y_pred: np.ndarray
    y_std: np.ndarray
    
    x_scaling: ScalingInfo
    y_scaling: ScalingInfo
    
    validation_mae: float = 0.0
    validation_rmse: float = 0.0
    
    filepath: Optional[Path] = None

    # Optional regulated-grid covariance (normalized units)
    covariance_matrix: Optional[np.ndarray] = None
    covariance_grid: Optional[np.ndarray] = None
    covariance_units: str = "normalized"
    
    @property
    def n_points(self) -> int:
        """Number of prediction points."""
        return len(self.y_pred)
    
    @property
    def y_pred_normalized(self) -> np.ndarray:
        """Normalized Y predictions using y_scaling.transform()."""
        return self.y_scaling.transform(self.y_pred)
    
    @property
    def y_std_normalized(self) -> np.ndarray:
        """Normalized Y standard deviation using y_scaling.transform()."""
        return self.y_scaling.transform(self.y_std)

    @property
    def covariance_real(self) -> Optional[np.ndarray]:
        """Covariance matrix converted to real scale if available."""
        if self.covariance_matrix is None:
            return None
        scale = _get_scale_multiplier(self.y_scaling)
        return (scale ** 2) * self.covariance_matrix
    
    def x_to_original(self, x_transformed: np.ndarray) -> np.ndarray:
        """Convert X from transformed to original space using x_scaling.inverse_transform()."""
        return self.x_scaling.inverse_transform(x_transformed)
    
    def x_to_transformed(self, x_original: np.ndarray) -> np.ndarray:
        """Convert X from original to transformed space using x_scaling.transform()."""
        return self.x_scaling.transform(x_original)
    
    def y_to_original(self, y_normalized: np.ndarray) -> np.ndarray:
        """Convert Y from normalized to original scale using y_scaling.inverse_transform()."""
        return self.y_scaling.inverse_transform(y_normalized)
    
    def y_to_normalized(self, y_original: np.ndarray) -> np.ndarray:
        """Convert Y from original to normalized scale using y_scaling.transform()."""
        return self.y_scaling.transform(y_original)
    
    @staticmethod
    def to_arrays(
        gpr_list: List["IndividualGPRData"],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[ScalingInfo], List[str]]:
        """
        Convert a list of IndividualGPRData to arrays for compute_summary_gpr.
        
        Assumes all curves share the same x_pred grid (no interpolation).
        For curves with different grids, use the orchestrator's interpolation.
        
        Parameters
        ----------
        gpr_list : List[IndividualGPRData]
            List of GPR data objects with same x grid.
            
        Returns
        -------
        Tuple containing:
            x_pred : np.ndarray, shape (n_points,)
                X values in transformed space (from first curve).
            y_array : np.ndarray, shape (n_curves, n_points)
                Y predictions in original scale.
            S_array : np.ndarray, shape (n_curves, n_points)
                Standard deviations in original scale.
            y_scalings : List[ScalingInfo]
                Y-axis scaling info for each curve.
            sample_ids : List[str]
                Sample identifiers.
                
        Raises
        ------
        ValueError
            If gpr_list is empty.
            
        Example
        -------
        >>> gprs = load_all_individual_gprs(directory)
        >>> x, y, S, scalings, ids = IndividualGPRData.to_arrays(gprs)
        >>> result = compute_summary_gpr(y, S, x, scalings, ...)
        """
        if not gpr_list:
            raise ValueError("gpr_list is empty")
        
        x_pred = gpr_list[0].x_pred_transformed
        y_array = np.stack([gpr.y_pred for gpr in gpr_list])
        S_array = np.stack([np.abs(gpr.y_std) for gpr in gpr_list])
        y_scalings = [gpr.y_scaling for gpr in gpr_list]
        sample_ids = [gpr.sample_id for gpr in gpr_list]
        
        return x_pred, y_array, S_array, y_scalings, sample_ids


def _reconstruct_scaling_info(method: str, params: Dict[str, Any]) -> ScalingInfo:
    """
    Reconstruct a ScalingInfo object from metadata.
    
    Parameters
    ----------
    method : str
        Scaling method name (e.g., "log_log10", "peak", "identity").
    params : Dict[str, Any]
        Scaling parameters.
        
    Returns
    -------
    ScalingInfo
        Reconstructed scaling information object.
    """
    # Parse method name - it may include base for log transforms
    # e.g., "log_log10" means log transform with base log10
    if method.startswith("log_"):
        base = method.split("_", 1)[1] if "_" in method else "log10"
        shift = params.get("shift", 1e-9)
        return ScalingInfo.log_transform(shift=shift, base=base)
    
    elif method == "log":
        base = params.get("base", "log10")
        shift = params.get("shift", 1e-9)
        return ScalingInfo.log_transform(shift=shift, base=base)
    
    elif method == "peak":
        # Reconstruct peak normalization using divide_by_factor
        factor = params.get("factor", 1.0)
        return ScalingInfo.divide_by_factor(factor, method_name='peak')
    
    elif method == "middle_average":
        # Reconstruct middle average using divide_by_factor
        factor = params.get("factor", 1.0)
        return ScalingInfo.divide_by_factor(factor, method_name='middle_average')
    
    elif method == "divide":
        factor = params.get("factor", 1.0)
        return ScalingInfo.divide_by_factor(factor)
    
    elif method == "standardize":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        return ScalingInfo.standardize(mean, std)
    
    elif method == "minmax":
        min_val = params.get("min_val", 0.0)
        max_val = params.get("max_val", 1.0)
        feature_range = params.get("feature_range", (0, 1))
        return ScalingInfo.minmax(min_val, max_val, feature_range)
    
    elif method == "identity" or method == "":
        return ScalingInfo.identity()
    
    else:
        # Unknown method - return identity with warning
        print(f"Warning: Unknown scaling method '{method}', using identity")
        return ScalingInfo.identity()


def _get_scale_multiplier(scaling: ScalingInfo) -> float:
    """Return linear scale multiplier to convert normalized → real values."""
    method = scaling.method
    if method in ("peak", "middle_average", "divide", "identity", ""):
        return float(scaling.params.get("factor", 1.0))
    if method == "standardize":
        return float(scaling.params.get("std", 1.0))
    if method == "minmax":
        min_val = float(scaling.params.get("min_val", 0.0))
        max_val = float(scaling.params.get("max_val", 1.0))
        feature_range = scaling.params.get("feature_range", (0, 1))
        scale = (max_val - min_val) / (feature_range[1] - feature_range[0])
        return float(scale)
    # Log transforms and unknown non-linear scalings: fallback to 1.0
    return 1.0


def _find_covariance_path(gpr_path: Path) -> Optional[Path]:
    """Find matching covariance matrix CSV for a given individual GPR file."""
    stem = gpr_path.stem
    suffix = stem[len("Individual_GPR_"):] if stem.startswith("Individual_GPR_") else stem
    candidate = gpr_path.with_name(f"Covariance_Matrix_{suffix}.csv")
    if candidate.exists():
        return candidate
    # Fallback glob search
    matches = list(gpr_path.parent.glob(f"Covariance_Matrix*{suffix}*.csv"))
    return matches[0] if matches else None


def _load_covariance_matrix(cov_path: Path, verbose: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load covariance matrix CSV exported from pip2 regulation workflow."""
    try:
        raw = pd.read_csv(cov_path, comment="#", header=None)
        if raw.empty:
            return None, None
        grid = raw.iloc[0, 1:].to_numpy(dtype=float)
        cov = raw.iloc[1:, 1:].to_numpy(dtype=float)
        if cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Covariance matrix not square: {cov.shape}")
        if cov.shape[0] != grid.size:
            raise ValueError(
                f"Covariance dimension {cov.shape} does not match grid length {grid.size}"
            )
        if verbose:
            print(f"    Loaded covariance: {cov.shape} from {cov_path.name}")
        return cov, grid
    except Exception as exc:
        print(f"Warning: failed to load covariance from {cov_path}: {exc}")
        return None, None


def load_individual_gpr_with_metadata(
    filepath: Union[Path, str],
    verbose: bool = False,
) -> IndividualGPRData:
    """
    Load a single individual GPR CSV file with metadata.
    
    Parameters
    ----------
    filepath : Path or str
        Path to the CSV file.
    verbose : bool
        Whether to print loading status.
        
    Returns
    -------
    IndividualGPRData
        Loaded GPR data with metadata.
        
    Example
    -------
    >>> gpr_data = load_individual_gpr_with_metadata("Individual_GPR_sample1_potential_-1.95.csv")
    >>> print(gpr_data.sample_id, gpr_data.group_key)
    >>> print(gpr_data.y_scaling.method)
    """
    filepath = Path(filepath)
    
    # Parse metadata
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
                    
                    # Parse JSON for dict values
                    if key in ('group_flags', 'x_scaling_params', 'y_scaling_params', 'hyperparams'):
                        try:
                            val = json.loads(val)
                        except json.JSONDecodeError:
                            pass
                    elif key == 'index_id':
                        try:
                            val = int(val)
                        except ValueError:
                            val = 0
                    elif key in ('validation_mae', 'validation_rmse'):
                        try:
                            val = float(val)
                        except ValueError:
                            val = 0.0
                    
                    metadata[key] = val
            else:
                break
    
    # Read data portion
    df = pd.read_csv(filepath, skiprows=header_lines)
    
    # Reconstruct scaling info early (needed for normalized-only data)
    x_method = metadata.get('x_scaling_method', 'identity')
    x_params = metadata.get('x_scaling_params', {})
    x_scaling = _reconstruct_scaling_info(x_method, x_params)

    y_method = metadata.get('y_scaling_method', 'identity')
    y_params = metadata.get('y_scaling_params', {})
    y_scaling = _reconstruct_scaling_info(y_method, y_params)

    # Extract X arrays
    x_pred_transformed = df['x_pred_transformed'].values
    x_pred_original = df['x_pred_original'].values if 'x_pred_original' in df.columns else np.exp(x_pred_transformed)

    # Extract Y arrays (support normalized-only columns)
    y_pred: np.ndarray
    y_std: np.ndarray
    if 'y_pred' in df.columns:
        y_pred = df['y_pred'].values
        if 'y_std' in df.columns:
            y_std = df['y_std'].values
        elif 'y_std_normalized' in df.columns:
            # Some debug outputs store std with normalized name even when y_pred exists
            y_std = _reconstruct_scaling_info(
                metadata.get('y_scaling_method', 'identity'),
                metadata.get('y_scaling_params', {}),
            ).inverse_transform(df['y_std_normalized'].values)
        else:
            y_std = np.zeros_like(y_pred)
    elif 'y_pred_normalized' in df.columns:
        y_pred_norm = df['y_pred_normalized'].values
        if 'y_std_normalized' in df.columns:
            y_std_norm = df['y_std_normalized'].values
        elif 'y_std' in df.columns:
            y_std_norm = df['y_std'].values
        else:
            y_std_norm = np.zeros_like(y_pred_norm)

        # Convert normalized values back to real scale
        y_pred = y_scaling.inverse_transform(y_pred_norm)
        y_std = y_scaling.inverse_transform(y_std_norm)
    else:
        raise ValueError(f"Unsupported GPR CSV schema in {filepath.name}")
    
    # Optional covariance loading (regulated shared grid)
    cov_matrix: Optional[np.ndarray] = None
    cov_grid: Optional[np.ndarray] = None
    cov_path = _find_covariance_path(filepath)
    if cov_path is not None:
        cov_matrix, cov_grid = _load_covariance_matrix(cov_path, verbose=verbose)

    # Build result
    result = IndividualGPRData(
        sample_id=metadata.get('sample_id', filepath.stem),
        index_id=metadata.get('index_id', 0),
        group_key=metadata.get('group_key', ''),
        group_flags=metadata.get('group_flags', {}),
        x_pred_transformed=x_pred_transformed,
        x_pred_original=x_pred_original,
        y_pred=y_pred,
        y_std=np.abs(y_std),  # Ensure positive
        x_scaling=x_scaling,
        y_scaling=y_scaling,
        validation_mae=metadata.get('validation_mae', 0.0),
        validation_rmse=metadata.get('validation_rmse', 0.0),
        filepath=filepath,
        covariance_matrix=cov_matrix,
        covariance_grid=cov_grid,
        covariance_units="normalized",
    )
    
    if verbose:
        print(f"Loaded {filepath.name}: {result.n_points} points, group={result.group_key}")
    
    return result


def load_all_individual_gprs(
    directory: Union[Path, str],
    pattern: str = "Individual_GPR_*.csv",
    verbose: bool = False,
) -> List[IndividualGPRData]:
    """
    Load all individual GPR files from a directory.
    
    Parameters
    ----------
    directory : Path or str
        Directory containing GPR CSV files.
    pattern : str
        Glob pattern for finding files.
    verbose : bool
        Whether to print loading status.
        
    Returns
    -------
    List[IndividualGPRData]
        List of loaded GPR data objects.
    """
    directory = Path(directory)
    files = list(directory.glob(pattern))
    
    if verbose:
        print(f"Found {len(files)} files matching pattern '{pattern}'")
    
    results = []
    for fp in files:
        try:
            gpr_data = load_individual_gpr_with_metadata(fp, verbose=verbose)
            results.append(gpr_data)
        except Exception as e:
            print(f"Error loading {fp.name}: {e}")
    
    return results


def group_gprs_by_key(
    gpr_list: List[IndividualGPRData],
    key_attr: str = "group_key",
) -> Dict[str, List[IndividualGPRData]]:
    """
    Group individual GPRs by a common key.
    
    Parameters
    ----------
    gpr_list : List[IndividualGPRData]
        List of GPR data objects.
    key_attr : str
        Attribute name to group by. Default is "group_key".
        Can also use nested access like "group_flags.potential".
        
    Returns
    -------
    Dict[str, List[IndividualGPRData]]
        GPRs grouped by the key.
        
    Example
    -------
    >>> gprs = load_all_individual_gprs(output_dir)
    >>> grouped = group_gprs_by_key(gprs)
    >>> for key, group in grouped.items():
    ...     print(f"{key}: {len(group)} curves")
    """
    groups: Dict[str, List[IndividualGPRData]] = defaultdict(list)
    
    for gpr in gpr_list:
        # Handle nested access
        if '.' in key_attr:
            parts = key_attr.split('.')
            val = gpr
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part, '')
                else:
                    val = getattr(val, part, '')
            key = str(val)
        else:
            key = str(getattr(gpr, key_attr, ''))
        
        groups[key].append(gpr)
    
    return dict(groups)
