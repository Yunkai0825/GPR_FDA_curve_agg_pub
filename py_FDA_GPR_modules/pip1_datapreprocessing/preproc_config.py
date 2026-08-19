# pip1_datapreprocessing/preproc_config.py
"""
Configuration dataclass for data preprocessing.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pip0_dataloading.settings_manager import SettingsManager

# Sentinel object for "not provided" — distinct from None
_MISSING = object()


@dataclass
class PreprocCfg:
    """
    Data pre-processing configuration.
    
    All filtering parameters (max_points_set, min_x_cap, max_x_cap,
    min_curve_range, y_threshold) MUST be supplied from JSON settings.
    No hardcoded defaults are provided — if any key is missing from the
    JSON, the pipeline will raise a clear error at construction time.
    
    Attributes
    ----------
    Column Names:
    -------------
    x_col : str
        Name of the x-axis column in input data (default: "x").
    y_col : str
        Name of the y-axis column in input data (default: "y").
    
    Filtering Parameters (REQUIRED — no defaults):
    -----------------------------------------------
    max_points_set : int
        Maximum number of points after downsampling.
    min_x_cap : float
        Minimum x value to include.
    max_x_cap : float
        Maximum x value to include.
    min_curve_range : float
        Minimum total x range required.
    y_threshold : float
        Minimum absolute normalized y to keep.
    
    X Scaling (mirrors ScalingInfo):
    --------------------------------
    x_scaling_method : str
        Scaling method for x-axis. Options:
        - "log": Log transformation (params: base, shift)
        - "identity": No transformation
        Default: "log"
    x_scaling_params : Dict[str, Any]
        Parameters for x scaling method.
        For "log": {"base": "log10"|"natural", "shift": float}
        Default: {"base": "log10", "shift": 1e-9}
    
    Y Scaling (mirrors ScalingInfo):
    --------------------------------
    y_scaling_method : str
        Scaling method for y-axis. Options:
        - "peak": Divide by peak (min/max) value
        - "middle_average": Divide by average of middle portion
        - "identity": No transformation
        Default: "peak"
    y_scaling_params : Dict[str, Any]
        Parameters for y scaling method.
        For "middle_average": {"start_fraction": float, "end_fraction": float}
        Default: {}
    
    Grouping Parameters:
    --------------------
    group_round_digits : int
        Decimal places for rounding group values (default: 2).
    """
    # Column names
    x_col: str = "x"
    y_col: str = "y"
    
    # Master toggle — when False, x-cap, y-threshold, and min-curve-range
    # filters are skipped (downsampling still applies).
    enable_filtering: bool = True

    # Filtering parameters — NO defaults; must be supplied from JSON settings.
    max_points_set: Any = _MISSING
    min_x_cap: Any = _MISSING
    max_x_cap: Any = _MISSING
    min_curve_range: Any = _MISSING
    y_threshold: Any = _MISSING
    
    # X scaling (mirrors ScalingInfo)
    x_scaling_method: str = "log"  # "log" | "identity"
    x_scaling_params: Dict[str, Any] = field(default_factory=lambda: {"base": "log10", "shift": 1e-9})
    
    # Y scaling (mirrors ScalingInfo)
    y_scaling_method: str = "peak"  # "peak" | "middle_average" | "identity"
    y_scaling_params: Dict[str, Any] = field(default_factory=dict)
    
    # Grouping parameters
    group_round_digits: int = 2
    
    def __post_init__(self):
        """Validate that all required filtering parameters were supplied."""
        _required = {
            "max_points_set": "pip1_datapreprocessing.filtering.max_points_set",
            "min_x_cap": "pip1_datapreprocessing.filtering.min_x_cap",
            "max_x_cap": "pip1_datapreprocessing.filtering.max_x_cap",
            "min_curve_range": "pip1_datapreprocessing.filtering.min_curve_range",
            "y_threshold": "pip1_datapreprocessing.filtering.y_threshold",
        }
        missing = [json_path for attr, json_path in _required.items()
                   if getattr(self, attr) is _MISSING]
        if missing:
            raise ValueError(
                f"The following required settings are missing from JSON "
                f"(no hardcoded defaults allowed):\n"
                + "\n".join(f"  - {p}" for p in missing)
            )

    @classmethod
    def from_settings(cls, manager: "SettingsManager") -> "PreprocCfg":
        """
        Create PreprocCfg from a SettingsManager.
        
        Parameters
        ----------
        manager : SettingsManager
            Settings manager with loaded settings.
            
        Returns
        -------
        PreprocCfg
        """
        return manager.get_preproc_config()