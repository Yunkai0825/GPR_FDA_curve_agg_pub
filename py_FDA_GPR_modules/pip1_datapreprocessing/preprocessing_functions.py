# pip1_datapreprocessing/preprocessing_functions.py
"""
Low-level data preprocessing functions for electrochemical transient data.

This module provides functions for:
- Normalizing current data (peak or middle-average method)
- Inverse normalization (rescaling back to original units)
- Downsampling data in log-time space
- Time filtering and log transformation

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict, Any, Callable
from dataclasses import dataclass, field


# Type alias for transform functions
TransformFunc = Callable[[np.ndarray], np.ndarray]


def _identity(x: np.ndarray) -> np.ndarray:
    """Identity transform (no-op)."""
    return x


@dataclass
class ScalingInfo:
    """
    Container for scaling/normalization information with callable transforms.
    
    This class stores both the transformation functions and metadata needed
    to convert between original and transformed (normalized) scales.
    
    Attributes
    ----------
    method : str
        Name of the transformation method ('log', 'divide_by_peak', 'standardize', etc.).
    params : Dict[str, Any]
        Parameters used for the transformation (e.g., scaling_factor, mean, std).
    transform_func : Callable
        Function to transform original values to normalized scale.
    inverse_func : Callable
        Function to transform normalized values back to original scale.
        
    Example
    -------
    >>> # Create scaling info for log transformation
    >>> x_scaling = ScalingInfo.log_transform(shift=1e-9)
    >>> x_transformed = x_scaling.transform(x_original)
    >>> x_original_back = x_scaling.inverse_transform(x_transformed)
    >>> 
    >>> # Create scaling info for division by peak
    >>> y_scaling = ScalingInfo.divide_by_factor(peak_value)
    >>> y_normalized = y_scaling.transform(y_original)
    >>> y_original_back = y_scaling.inverse_transform(y_normalized)
    """
    method: str = "identity"
    params: Dict[str, Any] = field(default_factory=dict)
    transform_func: TransformFunc = field(default=_identity)
    inverse_func: TransformFunc = field(default=_identity)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data from original to normalized scale."""
        return self.transform_func(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data from normalized back to original scale."""
        return self.inverse_func(data)
    
    def transform_std(self, std: np.ndarray) -> np.ndarray:
        """Transform standard deviation from original to normalized scale.
        
        For an affine transform  y_norm = a * y + b ,  the standard deviation
        transforms as  std_norm = |a| * std  (the offset b does not affect
        spread).  The scale factor a is extracted numerically as
        ``transform(1) - transform(0)`` which is exact for every affine
        method (divide, standardize, minmax, identity).
        
        .. warning::
            For non-affine transforms (e.g. log) this gives only a
            first-order approximation and should be used with care.
        """
        a = self.transform_func(np.ones(1)) - self.transform_func(np.zeros(1))
        return np.abs(a) * std
    
    def inverse_transform_std(self, std: np.ndarray) -> np.ndarray:
        """Transform standard deviation from normalized back to original scale.
        
        Inverse counterpart of :meth:`transform_std`.
        """
        a = self.inverse_func(np.ones(1)) - self.inverse_func(np.zeros(1))
        return np.abs(a) * std
    
    # --- Factory methods for common transformations ---
    
    @classmethod
    def identity(cls) -> "ScalingInfo":
        """Create identity scaling (no transformation)."""
        return cls(
            method="identity",
            params={},
            transform_func=_identity,
            inverse_func=_identity,
        )
    
    @classmethod
    def log_transform(cls, shift: float = 1e-9, base: str = "natural") -> "ScalingInfo":
        """
        Create log transformation scaling.
        
        Parameters
        ----------
        shift : float
            Small value added before log to avoid log(0).
        base : str
            'natural' for ln, 'log10' for log base 10.
        """
        if base == "log10":
            def forward(x: np.ndarray) -> np.ndarray:
                return np.log10(x + shift)
            def inverse(x: np.ndarray) -> np.ndarray:
                return np.power(10, x) - shift
        else:  # natural log
            def forward(x: np.ndarray) -> np.ndarray:
                return np.log(x + shift)
            def inverse(x: np.ndarray) -> np.ndarray:
                return np.exp(x) - shift
        
        return cls(
            method=f"log_{base}",
            params={"shift": shift, "base": base},
            transform_func=forward,
            inverse_func=inverse,
        )
    
    @classmethod
    def divide_by_factor(cls, factor: Union[float, np.ndarray], method_name: str = "divide") -> "ScalingInfo":
        """
        Create scaling by dividing by a factor.
        
        Parameters
        ----------
        factor : float or np.ndarray
            The factor to divide by (e.g., peak value, average value).
        method_name : str
            Name for this scaling method (e.g., 'peak', 'middle_average').
        """
        def forward(x: np.ndarray) -> np.ndarray:
            return x / factor
        def inverse(x: np.ndarray) -> np.ndarray:
            return x * factor
        
        return cls(
            method=method_name,
            params={"factor": factor},
            transform_func=forward,
            inverse_func=inverse,
        )
    
    @classmethod
    def standardize(cls, mean: float, std: float) -> "ScalingInfo":
        """
        Create standardization scaling (z-score normalization).
        
        Parameters
        ----------
        mean : float
            Mean value to subtract.
        std : float
            Standard deviation to divide by.
        """
        def forward(x: np.ndarray) -> np.ndarray:
            return (x - mean) / std
        def inverse(x: np.ndarray) -> np.ndarray:
            return x * std + mean
        
        return cls(
            method="standardize",
            params={"mean": mean, "std": std},
            transform_func=forward,
            inverse_func=inverse,
        )
    
    @classmethod
    def minmax(cls, min_val: float, max_val: float, feature_range: Tuple[float, float] = (0, 1)) -> "ScalingInfo":
        """
        Create min-max scaling.
        
        Parameters
        ----------
        min_val : float
            Minimum value in original data.
        max_val : float
            Maximum value in original data.
        feature_range : tuple
            Desired range (min, max) for scaled data.
        """
        scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
        min_target = feature_range[0]
        
        def forward(x: np.ndarray) -> np.ndarray:
            return (x - min_val) * scale + min_target
        def inverse(x: np.ndarray) -> np.ndarray:
            return (x - min_target) / scale + min_val
        
        return cls(
            method="minmax",
            params={"min_val": min_val, "max_val": max_val, "feature_range": feature_range},
            transform_func=forward,
            inverse_func=inverse,
        )
    
    @classmethod 
    def from_sklearn_scaler(cls, scaler: Any, method_name: str = "sklearn") -> "ScalingInfo":
        """
        Create ScalingInfo from a fitted sklearn scaler.
        
        Parameters
        ----------
        scaler : sklearn transformer
            A fitted sklearn scaler (StandardScaler, MinMaxScaler, etc.).
        method_name : str
            Name for this scaling method.
        """
        def forward(x: np.ndarray) -> np.ndarray:
            x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x
            return scaler.transform(x_reshaped).flatten()
        def inverse(x: np.ndarray) -> np.ndarray:
            x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x
            return scaler.inverse_transform(x_reshaped).flatten()
        
        # Extract params if available
        params: Dict[str, Any] = {"scaler_type": type(scaler).__name__}
        if hasattr(scaler, 'mean_'):
            params['mean'] = scaler.mean_
        if hasattr(scaler, 'scale_'):
            params['scale'] = scaler.scale_
        
        return cls(
            method=method_name,
            params=params,
            transform_func=forward,
            inverse_func=inverse,
        )
    
    @classmethod
    def from_peak_normalization(cls, data: np.ndarray) -> Tuple[Optional["ScalingInfo"], bool]:
        """
        Create ScalingInfo by normalizing to peak absolute value.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to compute peak from.
            
        Returns
        -------
        Tuple[Optional[ScalingInfo], bool]
            (ScalingInfo object, success flag). Returns (None, False) if peak is invalid.
            
        Example
        -------
        >>> y_scaling, ok = ScalingInfo.from_peak_normalization(current_data)
        >>> if ok:
        ...     y_normalized = y_scaling.transform(current_data)
        """
        peak_value = float(np.abs(data).max())
        if peak_value == 0 or np.isnan(peak_value):
            return None, False
        return cls.divide_by_factor(peak_value, method_name='peak'), True
    
    @classmethod
    def from_middle_average(
        cls, 
        data: np.ndarray, 
        start_fraction: float = 0.4, 
        end_fraction: float = 0.6,
    ) -> Tuple[Optional["ScalingInfo"], bool]:
        """
        Create ScalingInfo by normalizing to average of middle portion.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to compute middle average from.
        start_fraction : float
            Start of middle range as fraction (0.0 to 1.0).
        end_fraction : float
            End of middle range as fraction (0.0 to 1.0).
            
        Returns
        -------
        Tuple[Optional[ScalingInfo], bool]
            (ScalingInfo object, success flag). Returns (None, False) if average is invalid.
            
        Example
        -------
        >>> y_scaling, ok = ScalingInfo.from_middle_average(current_data, 0.4, 0.6)
        >>> if ok:
        ...     y_normalized = y_scaling.transform(current_data)
        """
        total_points = len(data)
        start_index = int(start_fraction * total_points)
        end_index = int(end_fraction * total_points)
        
        if start_index >= end_index:
            return None, False
        
        middle_data = data[start_index:end_index]
        if len(middle_data) == 0:
            return None, False
            
        average_value = float(np.mean(middle_data))
        if average_value == 0 or np.isnan(average_value):
            return None, False
        
        scaling = cls.divide_by_factor(average_value, method_name='middle_average')
        scaling.params['start_fraction'] = start_fraction
        scaling.params['end_fraction'] = end_fraction
        return scaling, True
    
    @classmethod
    def from_normalization(
        cls,
        data: np.ndarray,
        method: str = "peak",
        start_fraction: float = 0.4,
        end_fraction: float = 0.6,
    ) -> Tuple[Optional["ScalingInfo"], bool]:
        """
        Create ScalingInfo using the specified normalization method.
        
        This is a unified factory method that dispatches to the appropriate
        normalization strategy.
        
        Parameters
        ----------
        data : np.ndarray
            Data array to normalize.
        method : str
            Normalization method: 'peak' or 'middle_average'.
        start_fraction : float
            Start fraction for middle_average method.
        end_fraction : float
            End fraction for middle_average method.
            
        Returns
        -------
        Tuple[Optional[ScalingInfo], bool]
            (ScalingInfo object, success flag).
            
        Example
        -------
        >>> y_scaling, ok = ScalingInfo.from_normalization(current_data, method='peak')
        >>> if ok:
        ...     y_normalized = y_scaling.transform(current_data)
        ...     y_original = y_scaling.inverse_transform(y_normalized)
        """
        if method == 'peak':
            return cls.from_peak_normalization(data)
        elif method == 'middle_average':
            return cls.from_middle_average(data, start_fraction, end_fraction)
        else:
            return None, False


# Note: downsample_data and related functions have been moved to preprc_downsampling.py
# Import from there for backward compatibility
from .preprc_downsampling import downsample_data


def apply_x_filter(
    df: pd.DataFrame,
    min_x: float,
    max_x: float,
    x_col: str = "x",
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only data within x bounds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with x column.
    min_x : float
        Minimum x value to include.
    max_x : float
        Maximum x value to include.
    x_col : str
        Name of the x column.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[
        (df[x_col] >= min_x) &
        (df[x_col] <= max_x)
    ].copy()


def filter_by_y_threshold(
    df: pd.DataFrame,
    threshold: float,
    y_col: str = "y_transformed",
) -> pd.DataFrame:
    """
    Filter out data points with absolute y below threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with y column.
    threshold : float
        Minimum absolute y value to keep.
    y_col : str
        Name of the y column.
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    return df[df[y_col].abs() >= threshold].copy()
