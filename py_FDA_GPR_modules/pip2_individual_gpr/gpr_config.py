# pip2_individual_gpr/gpr_config.py
"""
Configuration dataclasses for Individual GPR fitting and export.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any, Union, TYPE_CHECKING
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

if TYPE_CHECKING:
    from ..pip0_dataloading.settings_manager import SettingsManager

# Sentinel for "not provided" — distinct from None
_MISSING = object()


def _default_kernel():
    """Default GPR kernel: Constant * Matern + White noise."""
    return (
        ConstantKernel(1.0, (1e-2, 1e4))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-12, 1e1))
    )


@dataclass
class SharedGridConfig:
    """
    Configuration for shared/regulated prediction grid across curves.
    
    The shared grid J_R ensures all curves are evaluated on the same x-coordinates,
    enabling proper FDA (Functional Data Analysis) aggregation and comparison.
    
    Workflow:
    ---------
    1. Step A: Fit GPR with hyperparameter optimization on train/val split
    2. Step B: Refit on full data with frozen hyperparameters θ*
    3. Step C: "Regulate" predictions onto shared grid J_R
    
    Attributes
    ----------
    enabled : bool
        Whether to use shared grid regulation.
    method : str
        How to compute the shared grid:
        - "auto": linspace(x_min, x_max) computed from all curves in a group
        - "explicit": Use explicit_grid expression
        - "per_group": Compute grid per-group based on that group's x-range
    explicit_grid : Optional[str]
        Python expression for explicit grid (evaluated with numpy as 'np').
        Example: "np.linspace(-3.0, 7.0, 500)"
    auto_num_points : int
        Number of grid points when method="auto" or "per_group".
    auto_padding_fraction : float
        Fraction of range to pad on each side (e.g., 0.05 = 5% padding).
    refit_on_full_data : bool
        Whether to refit GPR on ALL available data (train + validation)
        with frozen hyperparameters before regulating to shared grid.
        When False, the training-only GPR is used directly for
        predictions, bypassing the refit step (faster).
    """
    # NO defaults for grid parameters — must come from JSON settings.
    enabled: Any = _MISSING
    method: Any = _MISSING
    explicit_grid: Any = _MISSING
    auto_num_points: Any = _MISSING
    auto_padding_fraction: Any = _MISSING
    refit_on_full_data: Any = _MISSING
    
    def __post_init__(self):
        """Validate that all required grid parameters were supplied."""
        _required = {
            "enabled": "pip2_individual_gpr.shared_grid.enabled",
            "method": "pip2_individual_gpr.shared_grid.method",
            "explicit_grid": "pip2_individual_gpr.shared_grid.explicit_grid",
            "auto_num_points": "pip2_individual_gpr.shared_grid.auto_num_points",
            "auto_padding_fraction": "pip2_individual_gpr.shared_grid.auto_padding_fraction",
            "refit_on_full_data": "pip2_individual_gpr.shared_grid.refit_on_full_data",
        }
        missing = [json_path for attr, json_path in _required.items()
                   if getattr(self, attr) is _MISSING]
        if missing:
            raise ValueError(
                f"The following required shared_grid settings are missing from JSON "
                f"(no hardcoded defaults allowed):\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
    
    def evaluate_explicit_grid(self) -> Optional[np.ndarray]:
        """
        Evaluate the explicit_grid expression to get the actual grid array.
        
        Returns
        -------
        np.ndarray or None
            The evaluated grid, or None if not specified.
        """
        if self.explicit_grid is None or self.method != "explicit":
            return None
        try:
            # Evaluate the expression with numpy available
            result = eval(self.explicit_grid, {"np": np, "numpy": np})
            return np.asarray(result).flatten()
        except Exception as e:
            raise ValueError(f"Failed to evaluate explicit_grid expression '{self.explicit_grid}': {e}")
    
    def compute_auto_grid(
        self, 
        x_all: np.ndarray,
        num_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute automatic grid based on data range.
        
        Parameters
        ----------
        x_all : np.ndarray
            All x values from curves (concatenated).
        num_points : int, optional
            Override for auto_num_points.
            
        Returns
        -------
        np.ndarray
            Shared grid points.
        """
        n = num_points or self.auto_num_points
        x_min, x_max = x_all.min(), x_all.max()
        
        # Apply padding
        x_range = x_max - x_min
        pad = self.auto_padding_fraction * x_range
        x_min -= pad
        x_max += pad
        
        return np.linspace(x_min, x_max, n)


@dataclass
class GPRCfg:
    """
    GPR hyper-parameters and prediction grid configuration.
    
    Attributes
    ----------
    kernel : sklearn kernel
        GPR kernel (default: Constant * Matern + WhiteKernel).
    n_restarts_optimizer : int
        Number of optimizer restarts for kernel hyperparameter optimization.
    alpha : float
        Regularization term added to diagonal of kernel matrix.
        In theory: this is σ_f² (measurement noise variance).
    normalize_y : bool
        Whether to normalize target values (sklearn internal normalization).
    num_curves_threshold : int
        If more curves than this, use lower prediction resolution.
    num_X_pred_points_individual_default : int
        Default number of prediction points per curve.
    num_X_pred_points_individual_high : int
        Higher resolution for fewer curves.
    local_var_val_flag : bool
        Whether to compute local uncertainty from validation residuals.
        Note: This is deprecated in the current framework as pure GPR posterior
        already provides calibrated uncertainty.
    
    Shared Grid Configuration:
    --------------------------
    shared_grid : SharedGridConfig
        Configuration for the shared/regulated prediction grid.
        This ensures all curves are evaluated on the same x-coordinates.
    
    Posterior Covariance Options:
    -----------------------------
    store_posterior_covariance : bool
        Whether to compute and store the full posterior covariance matrix.
        Required for proper curve aggregation in pip3.
        WARNING: Memory-intensive for large prediction grids.
    covariance_storage_mode : str
        How to store covariance: 'full', 'diagonal', 'sparse', or 'cholesky'.
        - 'full': Store complete N×N matrix (most memory, exact)
        - 'diagonal': Store only diagonal (σ² per point, loses correlations)
        - 'sparse': Store sparse representation (threshold small values)
        - 'cholesky': Store Cholesky factor L where C = LL^T (saves ~50% memory)
    covariance_sparse_threshold : float
        For 'sparse' mode, values below this are set to zero.
    """
    kernel: object = field(default_factory=_default_kernel)
    n_restarts_optimizer: int = 5
    alpha: float = 0.0
    normalize_y: bool = True
    # NO defaults for grid-related parameters — must come from JSON settings.
    num_curves_threshold: Any = _MISSING
    num_X_pred_points_individual_default: Any = _MISSING
    num_X_pred_points_individual_high: Any = _MISSING
    local_var_val_flag: bool = False  # Deprecated: pure GPR posterior is well-calibrated
    
    # Shared grid configuration — NO default; must be constructed from JSON settings.
    shared_grid: Any = _MISSING
    
    # Posterior covariance options
    store_posterior_covariance: bool = True
    covariance_storage_mode: str = "full"  # 'full', 'diagonal', 'sparse', 'cholesky'
    covariance_sparse_threshold: float = 1e-6
    
    def __post_init__(self):
        """Validate that all required GPR parameters were supplied."""
        _required = {
            "num_curves_threshold": "pip2_individual_gpr.gpr_params.num_curves_threshold",
            "num_X_pred_points_individual_default": "pip2_individual_gpr.gpr_params.num_X_pred_points_individual_default",
            "num_X_pred_points_individual_high": "pip2_individual_gpr.gpr_params.num_X_pred_points_individual_high",
            "shared_grid": "pip2_individual_gpr.shared_grid",
        }
        missing = [json_path for attr, json_path in _required.items()
                   if getattr(self, attr) is _MISSING]
        if missing:
            raise ValueError(
                f"The following required GPR settings are missing from JSON "
                f"(no hardcoded defaults allowed):\n"
                + "\n".join(f"  - {p}" for p in missing)
            )
    
    @classmethod
    def from_settings(cls, manager: "SettingsManager") -> Tuple["GPRCfg", "ExportCfg"]:
        """
        Create GPRCfg and ExportCfg from a SettingsManager.
        
        Parameters
        ----------
        manager : SettingsManager
            Settings manager with loaded settings.
            
        Returns
        -------
        Tuple[GPRCfg, ExportCfg]
        """
        return manager.get_gpr_configs()


@dataclass
class ExportCfg:
    """
    Plotting and CSV export configuration.
    
    Attributes
    ----------
    plot_individual_gpr : bool
        Whether to generate diagnostic plots for individual GPR fits.
    individual_curve_alpha : float
        Transparency for individual curves in plots.
    plot_downsample_points : int
        Number of points to show in plots.
    dpi : int
        DPI for saved figures.
    max_points_to_save : int
        Maximum points to save in CSV files.
    x_col_name : str
        Original x column name from data for axis label.
    y_col_name : str
        Original y column name from data for axis label.
    x_transform_method : str
        X transformation method (e.g., "log") for axis label.
    y_transform_method : str
        Y transformation method (e.g., "middle_average") for axis label.
    """
    plot_individual_gpr: bool = True
    individual_curve_alpha: float = 0.20
    plot_downsample_points: int = 500
    dpi: int = 300
    max_points_to_save: int = 10_000
    x_col_name: str = "X"
    y_col_name: str = "Y"
    x_transform_method: str = ""
    y_transform_method: str = ""
