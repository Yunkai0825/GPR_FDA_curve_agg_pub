# pip1_datapreprocessing/data_preprocessor.py
"""
Data Preprocessing Orchestrator for the GPR-FDA Pipeline.

This module provides a unified entry point for all data preprocessing operations,
accepting data from pip0_dataloading and returning preprocessed data ready for GPR fitting.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field

# Import from pip0 for type hints
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .preproc_config import PreprocCfg
from .preprocessing_functions import (
    downsample_data,
    apply_x_filter,
    filter_by_y_threshold,
    ScalingInfo,
)


@dataclass
class PreprocessedCurve:
    """
    Container for a single preprocessed curve ready for GPR fitting.
    
    Contains data at all stages of preprocessing with transformation info.
    
    UNITS GUIDE:
    - "original" = raw units from instrument (e.g., seconds, A/cm²)
    - "transformed" = after log/normalization (e.g., log(seconds), normalized I)
    
    Attributes
    ----------
    sample_id : str
        Sample identifier.
    group_flags : Dict[str, Any]
        Generic grouping flags (e.g., {"potential": -1.95}).
    file_path : Optional[Path]
        Source file path.
    
    Original Data [ORIGINAL UNITS]:
    -------------------------------
    x_original : np.ndarray
        Original x values. Unit: seconds (or whatever x_col contains).
    y_original : np.ndarray
        Original y values. Unit: A/cm² (or whatever y_col contains).
    
    Full Transformed Data [TRANSFORMED UNITS, for validation]:
    ----------------------------------------------------------
    x_transformed : np.ndarray
        Full transformed x values. Unit: log(seconds) if use_log_x=True.
    y_transformed : np.ndarray
        Full normalized y values. Unit: dimensionless (normalized by peak/avg).
    
    Training Data [TRANSFORMED UNITS, downsampled for GPR]:
    -------------------------------------------------------
    x_train_transformed : np.ndarray
        Downsampled x values. Unit: log(seconds) if use_log_x=True.
        Shape: (n_points, 1) for sklearn compatibility.
    y_train_transformed : np.ndarray
        Downsampled y values. Unit: dimensionless (normalized).
        Shape: (n_points,)
    
    Scaling Information [for inverse transforms]:
    ---------------------------------------------
    x_scaling : ScalingInfo
        Transformation info for x (e.g., log transform with shift).
        Use x_scaling.inverse_transform() to convert back to original units.
    y_scaling : ScalingInfo
        Transformation info for y (e.g., peak normalization with factor).
        Use y_scaling.inverse_transform() to convert back to A/cm².
    
    DataFrames [MIXED UNITS]:
    -------------------------
    df_full_original_and_transformed : pd.DataFrame
        Full DataFrame after filtering. Contains columns:
        - Original: x_col, y_col (original units)
        - Transformed: 'x_transformed', 'y_transformed' (transformed units)
    df_downsampled_transformed : pd.DataFrame
        Downsampled DataFrame. Same columns as df_full_original_and_transformed.
    """
    # Identifiers
    sample_id: str
    group_flags: Dict[str, Any]
    file_path: Optional[Path] = None
    
    # Original data [ORIGINAL UNITS: seconds, A/cm²]
    x_original: Optional[np.ndarray] = None
    y_original: Optional[np.ndarray] = None
    
    # Transformed data, full [TRANSFORMED UNITS: log(s), normalized]
    x_transformed: Optional[np.ndarray] = None
    y_transformed: Optional[np.ndarray] = None
    
    # Training data, downsampled [TRANSFORMED UNITS: log(s), normalized]
    x_train_transformed: Optional[np.ndarray] = None
    y_train_transformed: Optional[np.ndarray] = None
    
    # Scaling information (for inverse transforms)
    x_scaling: Optional[ScalingInfo] = None
    y_scaling: Optional[ScalingInfo] = None
    
    # DataFrames [MIXED: contains both original and transformed columns]
    df_full_original_and_transformed: Optional[pd.DataFrame] = None
    df_downsampled_transformed: Optional[pd.DataFrame] = None
    
    # --- Backward compatibility aliases ---
    @property
    def x_train(self) -> Optional[np.ndarray]:
        """Alias for x_train_transformed (backward compat)."""
        return self.x_train_transformed
    
    @property
    def y_train(self) -> Optional[np.ndarray]:
        """Alias for y_train_transformed (backward compat)."""
        return self.y_train_transformed
    
    @property
    def df_downsampled(self) -> Optional[pd.DataFrame]:
        """Alias for df_downsampled_transformed (backward compat)."""
        return self.df_downsampled_transformed
    
    @property
    def df_full(self) -> Optional[pd.DataFrame]:
        """Alias for df_full_original_and_transformed (backward compat)."""
        return self.df_full_original_and_transformed
    
    # --- Utility properties ---
    
    @property
    def num_points(self) -> int:
        """Number of training points (downsampled)."""
        return len(self.x_train) if self.x_train is not None else 0
    
    @property
    def num_points_full(self) -> int:
        """Number of points before downsampling."""
        return len(self.x_transformed) if self.x_transformed is not None else 0
    
    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full transformed data for validation (before downsampling)."""
        x_val = self.x_transformed.reshape(-1, 1) if self.x_transformed is not None else np.array([]).reshape(-1, 1)
        y_val = self.y_transformed if self.y_transformed is not None else np.array([])
        return x_val, y_val
    
    def inverse_transform_y(self, y_normalized: np.ndarray) -> np.ndarray:
        """Transform y values from normalized back to original scale."""
        if self.y_scaling is None:
            return y_normalized
        return self.y_scaling.inverse_transform(y_normalized)
    
    def inverse_transform_x(self, x_transformed: np.ndarray) -> np.ndarray:
        """Transform x values from transformed back to original scale."""
        if self.x_scaling is None:
            return x_transformed
        return self.x_scaling.inverse_transform(x_transformed)


@dataclass
class PreprocessingResult:
    """
    Result container for preprocessing operations.
    
    Attributes
    ----------
    curves : List[PreprocessedCurve]
        Successfully preprocessed curves.
    curves_by_group : Dict[str, List[PreprocessedCurve]]
        Curves grouped by a string representation of group_flags.
    curves_by_primary_key : Dict[Any, List[PreprocessedCurve]]
        Curves grouped by primary grouping key value.
    primary_grouping_key : str
        Name of the primary grouping key used.
    skipped : List[Dict[str, Any]]
        List of skipped curves with reasons.
    config : PreprocCfg
        Configuration used for preprocessing.
    """
    curves: List[PreprocessedCurve] = field(default_factory=list)
    curves_by_group: Dict[str, List[PreprocessedCurve]] = field(default_factory=dict)
    curves_by_primary_key: Dict[Any, List[PreprocessedCurve]] = field(default_factory=dict)
    primary_grouping_key: str = ""
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[PreprocCfg] = None
    
    @property
    def num_curves(self) -> int:
        """Total number of successfully preprocessed curves."""
        return len(self.curves)
    
    @property
    def num_skipped(self) -> int:
        """Number of skipped curves."""
        return len(self.skipped)
    
    @property
    def primary_key_values(self) -> List[Any]:
        """List of unique primary key values."""
        return sorted(self.curves_by_primary_key.keys())
    
    @property
    def groups(self) -> List[str]:
        """List of unique group keys."""
        return sorted(self.curves_by_group.keys())
    
    def get_curves_for_primary_key(self, value: Any) -> List[PreprocessedCurve]:
        """Get all preprocessed curves for a specific primary key value."""
        return self.curves_by_primary_key.get(value, [])
    
    def get_curves_for_group(self, group_key: str) -> List[PreprocessedCurve]:
        """Get all preprocessed curves for a specific group."""
        return self.curves_by_group.get(group_key, [])
    
    def get_curves_by_flags(self, **flags) -> List[PreprocessedCurve]:
        """Get curves matching specific group flag values."""
        result = []
        for curve in self.curves:
            match = all(curve.group_flags.get(k) == v for k, v in flags.items())
            if match:
                result.append(curve)
        return result
    
    def summary(self) -> str:
        """Return a summary string of the preprocessing result."""
        lines = [
            "Preprocessing Summary:",
            f"  Successfully preprocessed: {self.num_curves} curves",
            f"  Skipped: {self.num_skipped} curves",
            f"  Unique groups: {len(self.groups)}",
        ]
        key_name = self.primary_grouping_key or "primary key"
        if self.primary_key_values:
            lines.append(f"  Unique {key_name} values: {len(self.primary_key_values)}")
            for val in self.primary_key_values:
                n = len(self.curves_by_primary_key[val])
                lines.append(f"    {val}: {n} curves")
        return "\n".join(lines)


class DataPreprocessor:
    """
    Orchestrator for preprocessing electrochemical transient data.
    
    This class provides a unified interface for:
    - Accepting data from pip0_dataloading (LoadedCurve or raw dicts)
    - Applying time filtering, log transformation, normalization, downsampling
    - Returning preprocessed data ready for GPR fitting
    
    Example
    -------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
    >>> from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
    >>> 
    >>> # Load data from pip0
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> loaded = loader.load_all()
    >>> 
    >>> # Preprocess for GPR
    >>> preprocessor = DataPreprocessor()
    >>> result = preprocessor.preprocess_all(loaded.curves)
    >>> print(result.summary())
    >>> 
    >>> # Access preprocessed data
    >>> for curve in result.curves:
    ...     print(curve.sample_id, curve.X_raw.shape, curve.scaling_factor)
    
    Parameters
    ----------
    config : PreprocCfg, optional
        Preprocessing configuration. If None, uses defaults.
    verbose : bool, optional
        Whether to print progress messages (default: True).
    """
    
    def __init__(
        self,
        config: Optional[PreprocCfg] = None,
        verbose: bool = True,
    ):
        if config is None:
            raise ValueError(
                "config must be provided explicitly (constructed from JSON settings). "
                "No default PreprocCfg is allowed — filtering/grid parameters must come from JSON."
            )
        self.config = config
        self.verbose = verbose
    
    def _create_scaling_from_config(
        self,
        method: str,
        params: Dict[str, Any],
        data: np.ndarray,
        is_x: bool = True,
    ) -> Optional[ScalingInfo]:
        """
        Create a ScalingInfo object from config method and params.
        
        This mirrors ScalingInfo's factory methods but is driven by config.
        
        Parameters
        ----------
        method : str
            Scaling method name (e.g., "log", "peak", "middle_average", "identity").
        params : Dict[str, Any]
            Parameters for the scaling method.
        data : np.ndarray
            Data to compute scaling from (used for data-dependent methods like peak).
        is_x : bool
            Whether this is x-axis scaling (affects default behavior).
            
        Returns
        -------
        Optional[ScalingInfo]
            ScalingInfo object or None if creation failed.
        """
        method_lower = method.lower()
        
        if method_lower == "identity":
            return ScalingInfo.identity()
        
        elif method_lower == "log":
            base = params.get("base", "log10")
            shift = params.get("shift", 1e-9)
            return ScalingInfo.log_transform(shift=shift, base=base)
        
        elif method_lower == "peak":
            # Divide by peak (min or max absolute value)
            scaling_info, success = ScalingInfo.from_peak_normalization(data)
            return scaling_info if success else None
        
        elif method_lower == "middle_average":
            start_frac = params.get("start_fraction", 0.5)
            end_frac = params.get("end_fraction", 0.9)
            scaling_info, success = ScalingInfo.from_middle_average(
                data, start_fraction=start_frac, end_fraction=end_frac
            )
            return scaling_info if success else None
        
        elif method_lower == "divide":
            # Divide by a fixed factor
            factor = params.get("factor", 1.0)
            return ScalingInfo.divide_by_factor(factor, method_name="divide")
        
        else:
            # Unknown method - return identity
            if self.verbose:
                print(f"Unknown scaling method '{method}', using identity.")
            return ScalingInfo.identity()
    
    def preprocess_single(
        self,
        curve_data: Union[dict, Any],  # LoadedCurve or dict
        group_flags: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[PreprocessedCurve], Optional[str]]:
        """
        Preprocess a single curve.
        
        Parameters
        ----------
        curve_data : dict or LoadedCurve
            Curve data with 'x' and 'y' arrays or columns matching config.
        group_flags : Dict[str, Any], optional
            Grouping flags for this curve (e.g., {"potential": -1.95}).
            
        Returns
        -------
        Tuple[Optional[PreprocessedCurve], Optional[str]]
            (preprocessed curve or None, skip reason or None)
        """
        cfg = self.config
        x_col = cfg.x_col
        y_col = cfg.y_col
        
        # Handle both LoadedCurve objects and raw dicts
        curve_dict: Dict[str, Any]
        if isinstance(curve_data, dict):
            curve_dict = curve_data
        elif hasattr(curve_data, 'to_dict'):
            curve_dict = curve_data.to_dict()  # type: ignore
        else:
            # Generic object with x, y arrays
            curve_dict = {
                'file_path': getattr(curve_data, 'file_path', None),
                'sample_id': getattr(curve_data, 'sample_id', 'unknown'),
                'x': getattr(curve_data, 'x', getattr(curve_data, 'x_raw', None)),
                'y': getattr(curve_data, 'y', getattr(curve_data, 'y_raw', None)),
            }
        
        sample_id = curve_dict.get('sample_id', 'unknown')
        file_path = curve_dict.get('file_path')
        
        # Create DataFrame - handle both 'data_points' list and direct 'x'/'y' arrays
        if 'data_points' in curve_dict and curve_dict['data_points']:
            df_original = pd.DataFrame(curve_dict['data_points'])
        elif 'x' in curve_dict and 'y' in curve_dict:
            df_original = pd.DataFrame({x_col: curve_dict['x'], y_col: curve_dict['y']})
        else:
            return None, "No valid data found in curve"
        
        # Sort by x column
        if x_col in df_original.columns:
            df_original = df_original.sort_values(by=x_col).reset_index(drop=True)
        
        # Check minimum curve range (skip if filtering disabled)
        if cfg.enable_filtering:
            x_range = df_original[x_col].max() - df_original[x_col].min()
            if x_range < cfg.min_curve_range:
                skip_reason = f"X range {x_range:.2f} < min_curve_range ({cfg.min_curve_range})"
                if self.verbose:
                    print(f"{skip_reason} for sample {sample_id}. Skipping.")
                return None, skip_reason
        
        # Apply x filter (skip if filtering disabled)
        if cfg.enable_filtering:
            df_filtered = apply_x_filter(df_original, cfg.min_x_cap, cfg.max_x_cap, x_col)
            if df_filtered.empty:
                skip_reason = "No data after applying x caps"
                if self.verbose:
                    print(f"{skip_reason} for sample {sample_id}. Skipping.")
                return None, skip_reason
        else:
            df_filtered = df_original
        
        # Store original x, y before transformation
        x_original = np.asarray(df_filtered[x_col].values)
        y_original = np.asarray(df_filtered[y_col].values)
        
        # Create x_scaling based on config (mirrors ScalingInfo structure)
        x_scaling = self._create_scaling_from_config(
            cfg.x_scaling_method, cfg.x_scaling_params, x_original, is_x=True
        )
        
        df_transformed = df_filtered.copy()
        df_transformed['x_transformed'] = x_scaling.transform(x_original)
        
        # Create y_scaling based on config (mirrors ScalingInfo structure)
        y_scaling = self._create_scaling_from_config(
            cfg.y_scaling_method, cfg.y_scaling_params, y_original, is_x=False
        )
        if y_scaling is None:
            skip_reason = f"Y scaling failed for sample {sample_id}"
            return None, skip_reason
        
        # Apply y normalization
        df_transformed['y_transformed'] = y_scaling.transform(y_original)
        
        # Filter by y threshold (skip if filtering disabled)
        if cfg.enable_filtering:
            df_normalized = filter_by_y_threshold(df_transformed, cfg.y_threshold, 'y_transformed')
            if df_normalized.empty:
                skip_reason = "All data points filtered out after normalization"
                if self.verbose:
                    print(f"{skip_reason} for sample {sample_id}. Skipping.")
                return None, skip_reason
        else:
            df_normalized = df_transformed
        
        # Extract full transformed data (for validation)
        x_transformed = np.asarray(df_normalized['x_transformed'].values)
        y_transformed = np.asarray(df_normalized['y_transformed'].values)
        
        # Downsample
        df_downsampled, _ = downsample_data(df_normalized, cfg.max_points_set, 'x_transformed')
        
        # Extract training data (downsampled + transformed)
        x_train = np.asarray(df_downsampled['x_transformed'].values).reshape(-1, 1)
        y_train = np.asarray(df_downsampled['y_transformed'].values)
        
        if self.verbose:
            print(f"Preprocessed {sample_id}: {len(df_downsampled)} points (from {len(df_normalized)})")
        
        # Use provided group_flags or empty dict
        final_group_flags = group_flags if group_flags is not None else {}
        
        return PreprocessedCurve(
            sample_id=sample_id,
            group_flags=final_group_flags,
            file_path=Path(file_path) if file_path else None,
            # Original data [ORIGINAL UNITS: seconds, A/cm²]
            x_original=x_original,
            y_original=y_original,
            # Transformed data, full [TRANSFORMED UNITS: log(s), normalized]
            x_transformed=x_transformed,
            y_transformed=y_transformed,
            # Training data, downsampled [TRANSFORMED UNITS: log(s), normalized]
            x_train_transformed=x_train,
            y_train_transformed=y_train,
            # Scaling information
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            # DataFrames [MIXED: original + transformed columns]
            df_full_original_and_transformed=df_normalized,
            df_downsampled_transformed=df_downsampled,
        ), None
    
    def preprocess_all(
        self,
        curves: List[Union[dict, Any]],
        group_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
        filter_groups: Optional[Dict[str, Any]] = None,
    ) -> PreprocessingResult:
        """
        Preprocess all curves.
        
        Parameters
        ----------
        curves : List[dict or object]
            List of curves with x/y data.
        group_extractor : Callable, optional
            Function to extract group_flags from each curve.
            Signature: (curve_data) -> Dict[str, Any]
            If None, no grouping is applied.
        filter_groups : Dict[str, Any], optional
            If provided, only preprocess curves matching these group flags.
            
        Returns
        -------
        PreprocessingResult
            Container with all preprocessed curves and metadata.
        """
        result = PreprocessingResult(config=self.config)
        curves_by_primary_key: Dict[Any, List[PreprocessedCurve]] = {}
        curves_by_group: Dict[str, List[PreprocessedCurve]] = {}
        
        if self.verbose:
            print(f"Preprocessing {len(curves)} curves...")
        
        for curve_data in curves:
            # Extract group flags:
            # 1. Use extractor if provided
            # 2. Otherwise, use group_flags from input curve (LoadedCurve)
            # 3. Default to empty dict
            group_flags: Dict[str, Any] = {}
            if group_extractor is not None:
                group_flags = group_extractor(curve_data)
            elif hasattr(curve_data, 'group_flags') and curve_data.group_flags:
                group_flags = dict(curve_data.group_flags)  # Copy to avoid mutation
            
            # Apply group filter
            if filter_groups is not None:
                match = all(group_flags.get(k) == v for k, v in filter_groups.items())
                if not match:
                    continue
            
            # Preprocess
            preprocessed, skip_reason = self.preprocess_single(curve_data, group_flags)
            
            if preprocessed is not None:
                result.curves.append(preprocessed)
                
                # Group by primary key (first key in group_flags)
                if group_flags:
                    primary_key = next(iter(group_flags.keys()))
                    primary_value = group_flags[primary_key]
                    # Round numeric values
                    if isinstance(primary_value, (int, float)):
                        primary_value = round(primary_value, self.config.group_round_digits)
                    if primary_value not in curves_by_primary_key:
                        curves_by_primary_key[primary_value] = []
                    curves_by_primary_key[primary_value].append(preprocessed)
                    result.primary_grouping_key = primary_key
                
                # Group by generic group_flags
                group_key = self._build_group_key(preprocessed.group_flags)
                if group_key not in curves_by_group:
                    curves_by_group[group_key] = []
                curves_by_group[group_key].append(preprocessed)
            else:
                # Get sample_id for tracking skipped curves
                sample_id: str
                if isinstance(curve_data, dict):
                    sample_id = curve_data.get('sample_id', 'unknown')
                else:
                    sample_id = getattr(curve_data, 'sample_id', 'unknown')
                result.skipped.append({
                    'sample_id': sample_id,
                    'group_flags': group_flags,
                    'reason': skip_reason,
                })
        
        result.curves_by_primary_key = curves_by_primary_key
        result.curves_by_group = curves_by_group
        
        if self.verbose:
            print(result.summary())
        
        return result
    
    @staticmethod
    def _build_group_key(group_flags: Dict[str, Any]) -> str:
        """Build a consistent string key from group flags for dictionary indexing."""
        sorted_items = sorted(group_flags.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def preprocess_from_loader_result(
        self,
        loader_result: Any,  # DataLoadingResult from pip0
        group_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
        filter_groups: Optional[Dict[str, Any]] = None,
    ) -> PreprocessingResult:
        """
        Preprocess directly from a DataLoadingResult.
        
        Parameters
        ----------
        loader_result : DataLoadingResult
            Result from pip0 DataLoader.load_all().
        group_extractor : Callable, optional
            Function to extract group_flags from each curve.
        filter_groups : Dict[str, Any], optional
            If provided, only preprocess curves matching these group flags.
            
        Returns
        -------
        PreprocessingResult
            Container with all preprocessed curves and metadata.
        """
        return self.preprocess_all(loader_result.curves, group_extractor, filter_groups)


# Convenience function for quick preprocessing
def preprocess_curves(
    curves: List[Union[dict, Any]],
    config: Optional[PreprocCfg] = None,
    group_extractor: Optional[Callable[[Any], Dict[str, Any]]] = None,
    filter_groups: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> PreprocessingResult:
    """
    Convenience function to preprocess curves.
    
    Parameters
    ----------
    curves : List[dict or object]
        List of curves with x/y data.
    config : PreprocCfg, optional
        Preprocessing configuration.
    group_extractor : Callable, optional
        Function to extract group_flags from each curve.
    filter_groups : Dict[str, Any], optional
        If provided, only preprocess curves matching these group flags.
    verbose : bool, optional
        Whether to print progress messages.
        
    Returns
    -------
    PreprocessingResult
        Container with all preprocessed curves.
        
    Example
    -------
    >>> from py_FDA_GPR_modules.pip1_datapreprocessing import preprocess_curves
    >>> 
    >>> # Define group extractor for your data
    >>> def extract_groups(curve):
    ...     return {"potential": curve.potential}
    >>> 
    >>> preprocessed = preprocess_curves(curves, group_extractor=extract_groups)
    >>> print(f"Preprocessed {preprocessed.num_curves} curves")
    """
    preprocessor = DataPreprocessor(config=config, verbose=verbose)
    return preprocessor.preprocess_all(curves, group_extractor, filter_groups)
