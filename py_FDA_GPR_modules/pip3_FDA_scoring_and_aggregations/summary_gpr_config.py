# pip3_FDA_scoring_and_aggregations/summary_gpr_config.py
"""
Configuration dataclasses for Summary GPR aggregation.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pip0_dataloading.settings_manager import SettingsManager


@dataclass
class SummaryGPRConfig:
    """
    Configuration for Summary GPR aggregation.
    
    Attributes
    ----------
    input_directory : Path
        Directory containing individual GPR CSV files.
    output_directory : Path, optional
        Directory for output files. Defaults to input_directory.
    file_pattern : str
        Glob pattern for finding individual GPR files.
        
    weight_mode : str
        Weight optimization mode: "equal" or "iterative".
    weight_scope : str
        Weight scope: "curve" (one weight per curve) or "point" (weight per point).
        
    include_within_variance : bool
        Whether to include within-model variance in weight optimization.
    include_between_variance : bool
        Whether to include between-model variance in weight optimization.
    variance_aggregation_scale : str
        Scale for variance aggregation: "real" or "normalised".
        
    normalization_summary : bool
        Whether to normalize predictions before aggregation.
        
    plot_individual_gprs : bool
        Whether to overlay individual GPRs on summary plot.
    individual_curve_alpha : float
        Transparency for individual curves in plot.
        
    min_time_cap : float
        Minimum time for x-axis display.
    max_time_cap : float
        Maximum time for x-axis display.
        
    process_groups : List[str], optional
        Specific group keys to process. If None, process all.
    """
    input_directory: Path
    output_directory: Optional[Path] = None
    file_pattern: str = "Individual_GPR_*.csv"
    
    # Aggregation settings
    weight_mode: str = "iterative"  # "equal" | "iterative"
    weight_scope: str = "curve"     # "curve" | "point"
    
    # Variance settings
    include_within_variance: bool = True
    include_between_variance: bool = True
    variance_aggregation_scale: str = "real"  # "real" | "normalised"
    
    # Normalization
    normalization_summary: bool = True
    
    # Plotting
    plot_individual_gprs: bool = True
    individual_curve_alpha: float = 0.20
    
    # Axis labels (from settings column_names)
    x_axis_label: str = "X_label"
    y_axis_label: str = "Y_label"
    y_transform_method: str = ""  # e.g. "middle_average" for normalized iteration plots
    
    # Display limits
    min_time_cap: float = 1e-4
    max_time_cap: Optional[float] = None  # None → auto-derive from data
    
    # Method toggles
    enable_operator_fusion: bool = False  # skip operator fusion when False
    enable_fgpr: bool = False              # skip FGPR when False
    enable_student_t: bool = False         # skip Student-t when False

    # Group filtering
    process_groups: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        self.input_directory = Path(self.input_directory)
        if self.output_directory is None:
            self.output_directory = self.input_directory
        else:
            self.output_directory = Path(self.output_directory)
        
        # Validate options
        if self.weight_mode not in ("equal", "iterative"):
            raise ValueError(f"weight_mode must be 'equal' or 'iterative', got '{self.weight_mode}'")
        if self.weight_scope not in ("curve", "point"):
            raise ValueError(f"weight_scope must be 'curve' or 'point', got '{self.weight_scope}'")
        if self.variance_aggregation_scale not in ("real", "normalised"):
            raise ValueError(f"variance_aggregation_scale must be 'real' or 'normalised', got '{self.variance_aggregation_scale}'")
    
    @classmethod
    def from_settings(
        cls,
        manager: "SettingsManager",
        input_directory: Path,
        output_directory: Optional[Path] = None,
    ) -> Tuple["SummaryGPRConfig", "SummaryGPRHyperParams"]:
        """
        Create SummaryGPRConfig and SummaryGPRHyperParams from a SettingsManager.
        
        Parameters
        ----------
        manager : SettingsManager
            Settings manager with loaded settings.
        input_directory : Path
            Directory containing individual GPR CSVs.
        output_directory : Path, optional
            Output directory.
            
        Returns
        -------
        Tuple[SummaryGPRConfig, SummaryGPRHyperParams]
        """
        return manager.get_summary_gpr_configs(input_directory, output_directory)


@dataclass
class SummaryGPRHyperParams:
    """
    Hyperparameters for Summary GPR algorithm.
    
    Attributes
    ----------
    max_iterations : int, optional
        Maximum iterations for weight optimization. None for unlimited.
    convergence_tol : float
        Convergence tolerance for weight optimization.
    epsilon : float
        Small constant to prevent division by zero.
    confidence_level : float
        Confidence level for uncertainty bands (0-1).
    num_interp_points : int
        Number of interpolation points for summary curve.
    """
    max_iterations: Optional[int] = None
    convergence_tol: float = 1e-6
    epsilon: float = 1e-12
    confidence_level: float = 0.75
    num_interp_points: int = 500

    # FGPR outlier filtering: exclude curves whose |scale_factor| is below
    # this fraction of the median |scale_factor| in the group.  Curves with
    # near-zero scale factors produce extreme normalised values that corrupt
    # sigma_btw^2 estimation.  Set to 0 to disable filtering.
    fgpr_min_scale_factor_ratio: float = 0.01

    # Structured between-curve covariance (FGPR-v2).
    # When True, replaces scalar σ²_btw I with a multi-component operator:
    #   Ĉ_btw = σ²_w I + σ²_s K_smooth(ℓ_b) + σ²_o |1⟩⟨1|
    #         + σ²_d |t⟩⟨t| + σ²_sc |m_ref⟩⟨m_ref|
    fgpr_structured_btw: bool = False

    # Student-t robust aggregation hyperparameters
    student_t_nu: float = 5.0                # initial effective DOF νN (small → heavy tails, large → Gaussian)
    student_t_optimize_nu: bool = True       # optimise ν via Eqn (30) at each iteration
    student_t_nu_bounds: Tuple[float, float] = (1.0, 500.0)  # search bounds for effective DOF νN
    student_t_nu_lb_adaptive: bool = False    # if True, clamp lower bound to 1/N (ultra-heavy tails)
    student_t_max_iterations: int = 100      # max IRLS iterations
    student_t_convergence_tol: float = 1e-6  # convergence tolerance on normalised weight change (Eqn 32)
