# pip5_batch_setting_grid_testing/batch_config.py
"""
Configuration dataclasses for batch grid testing.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class BatchDirParams:
    """
    Directory parameters for batch testing.
    
    Attributes
    ----------
    input_dir : Path
        Directory containing individual GPR CSVs.
    base_output_dir : Path, optional
        Base directory for batch outputs. If None, uses input_dir.
    """
    input_dir: Path
    base_output_dir: Optional[Path] = None
    
    def __post_init__(self):
        self.input_dir = Path(self.input_dir)
        if self.base_output_dir is None:
            self.base_output_dir = self.input_dir
        else:
            self.base_output_dir = Path(self.base_output_dir)
    
    def get_combo_output_dir(self, tag: str) -> Path:
        """Get output directory for a specific combination tag."""
        assert self.base_output_dir is not None
        return self.base_output_dir / tag


@dataclass
class BatchTestingOptions:
    """
    A single combination of testing options.
    
    Attributes
    ----------
    normalization_summary : bool
        Whether to normalize the summary curve.
    weight_mode : str
        Weight mode: 'equal' or 'iterative'.
    weight_scope : str
        Weight scope: 'curve' or 'point'.
    aggregation_method : str
        Aggregation method: 'iterative', 'fgpr', or 'student_t'.
    """
    normalization_summary: bool
    weight_mode: str
    weight_scope: str
    aggregation_method: str = "iterative"
    
    @property
    def tag(self) -> str:
        """Generate a folder tag for this combination."""
        ns_str = "norm" if self.normalization_summary else "real"
        if self.aggregation_method == "fgpr":
            return f"NS_{ns_str}__AM_fgpr"
        if self.aggregation_method == "student_t":
            return f"NS_{ns_str}__AM_student_t"
        return f"NS_{ns_str}__WM_{self.weight_mode}__WS_{self.weight_scope}"
    
    def __repr__(self) -> str:
        return self.tag


@dataclass
class BatchRunConfig:
    """
    Configuration for what to run in batch testing.
    
    Attributes
    ----------
    run_summary_gpr : bool
        Whether to run Summary GPR for each potential.
    run_efficiency_eval : bool
        Whether to run data efficiency evaluation.
    run_combine_gprs : bool
        Whether to combine individual GPRs into consolidated CSVs.
    run_comparison_plots : bool
        Whether to generate comparison plots after all permutations complete.
    export_summary_csvs : bool
        Whether to export aggregated summary CSVs for regenerating plots.
    copy_artifacts : bool
        Whether to copy artifacts to combo output folders.
    """
    run_summary_gpr: bool = True
    run_efficiency_eval: bool = True
    run_combine_gprs: bool = True
    run_comparison_plots: bool = True
    export_summary_csvs: bool = True
    copy_artifacts: bool = True


# Export patterns for artifacts
ARTIFACT_PATTERNS: List[str] = [
    # Summary GPR outputs
    "Summary_GPR_*.csv",
    "Summary_GPR_*.png",
    "Converged_Weights_*.csv",
    "Weight_History_*.csv",
    "Curve_History_*.csv",
    "Summary_GPR_Iterations_*.png",
    "Weight_Distribution_*.png",
    "Weight_Convergence_*.png",
    # Efficiency outputs
    "LearningCurve_*.csv",
    "Summary_Efficiency_*.png",
    "Iteration_Statistics_*.png",
    # Combined GPR outputs
    "Combined_Individual_GPR_*.csv",
]
