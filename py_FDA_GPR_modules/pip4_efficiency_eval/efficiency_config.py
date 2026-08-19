# pip4_efficiency_eval/efficiency_config.py
"""
Configuration dataclasses for Efficiency Evaluation.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pip0_dataloading.settings_manager import SettingsManager

Metric = Literal["rmse", "mae", "max"]
LogBase = Literal["e", "10"]


@dataclass
class DirParams:
    """I/O and paths configuration."""
    indiv_dir: Path
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.indiv_dir = Path(self.indiv_dir)
        if self.output_dir is None:
            self.output_dir = self.indiv_dir
        else:
            self.output_dir = Path(self.output_dir)

    @property
    def summary_csv(self) -> Path:
        if self.output_dir is None:
            raise ValueError("output_dir is not set")
        return self.output_dir / "Summary_Efficiency_summary.csv"


@dataclass
class GlobalParams:
    """Global experiment knobs."""
    metric: Metric = "rmse"
    base_repeats: int = 1000
    max_enum: int = 1000
    q_low: float = 0.25
    q_high: float = 0.75
    random_seed: int = 42


@dataclass
class ScaleParams:
    """Axis-scale / plotting controls."""
    use_log_error: bool = True
    log_base_error: LogBase = "10"
    eps_error: float = 1e-30
    use_log_cost: bool = True
    log_base_cost: LogBase = "10"
    eps_cost: float = 1e-30
    normalize_w_rbar: bool = False


@dataclass
class PlotParams:
    """Plot configuration."""
    figsize: Tuple[int, int] = (10, 4)
    dpi: int = 120
    xlabel: str = "Number of curves"
    ylabel: str = "Error metric"
    time_label: str = "CPU time (s)"


def get_all_configs_from_settings(
    manager: "SettingsManager",
    indiv_dir: Path,
    output_dir: Optional[Path] = None,
) -> Tuple[DirParams, GlobalParams, ScaleParams, PlotParams]:
    """
    Create all efficiency configs from a SettingsManager.
    
    Parameters
    ----------
    manager : SettingsManager
        Settings manager with loaded settings.
    indiv_dir : Path
        Directory containing individual GPR CSVs.
    output_dir : Path, optional
        Output directory.
        
    Returns
    -------
    Tuple[DirParams, GlobalParams, ScaleParams, PlotParams]
    """
    return manager.get_efficiency_configs(indiv_dir, output_dir)


def export_param_classes():
    """Export parameter classes for external use."""
    return DirParams, GlobalParams, ScaleParams, PlotParams
