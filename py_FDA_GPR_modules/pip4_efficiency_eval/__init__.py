# pip4_efficiency_eval/__init__.py
"""
Efficiency Evaluation Module for GPR-FDA.

Evaluates how many curves are needed to reproduce the full-data summary-GPR.
Implements Monte-Carlo learning curve analysis.

Module Structure:
- mc_sampling.py: Monte Carlo sampling utilities (balanced subsets, combinatorics)
- learning_curve.py: Learning curve computation algorithms
- efficiency_core.py: Re-exports from above for backward compatibility
- efficiency_plotting.py: Plotting utilities (core + CSV wrappers)
- pip4_efficiency_eval_orchestrator.py: High-level orchestrator

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from .efficiency_config import (
    DirParams,
    GlobalParams,
    ScaleParams,
    PlotParams,
)

# Reuse loader from pip3 instead of duplicate implementation
from ..pip3_FDA_scoring_and_aggregations import (
    IndividualGPRData,
    load_all_individual_gprs,
    group_gprs_by_key,
)

# MC Sampling utilities
from .mc_sampling import (
    bounded_comb,
    mc_repeats,
    balanced_subset,
    generate_balanced_subsets,
)

# Learning curve computation
from .learning_curve import (
    SubsetResult,
    LearningCurveResult,
    fast_summary_gpr_core,
    error_metric,
    learning_curve,
    learning_curve_layered,
    summarize_learning_curve,
)

# High-level processing from efficiency_core
from .efficiency_core import (
    process_potential_learning_curve,
)

from .efficiency_plotting import (
    # Core plotting functions (take DataFrames directly)
    plot_learning_curve,
    plot_iteration_statistics,
    # CSV wrapper functions
    plot_learning_curve_from_detailed,
    plot_iteration_statistics_from_detailed,
    # Multi-method comparison
    plot_multimethod_comparison,
    plot_multimethod_bar_summary,
    # sigma_btw and covariance diagnostics
    plot_sigma_btw_comparison,
    plot_covariance_heatmaps,
    plot_covariance_diagonal,
    plot_pointwise_sigma_btw,
    export_sigma_btw_csv,
    # Aggregation utility
    aggregate_detailed_to_summary,
)

from .pip4_efficiency_eval_orchestrator import (
    EfficiencyOrchestrator,
)

__all__ = [
    # Config
    "DirParams",
    "GlobalParams",
    "ScaleParams",
    "PlotParams",
    # Loader (from pip3)
    "IndividualGPRData",
    "load_all_individual_gprs",
    "group_gprs_by_key",
    # MC Sampling
    "bounded_comb",
    "mc_repeats",
    "balanced_subset",
    "generate_balanced_subsets",
    # Learning Curve
    "SubsetResult",
    "LearningCurveResult",
    "fast_summary_gpr_core",
    "error_metric",
    "learning_curve",
    "learning_curve_layered",
    "summarize_learning_curve",
    # High-level processing
    "process_potential_learning_curve",
    # Plotting - Core functions
    "plot_learning_curve",
    "plot_iteration_statistics",
    # Plotting - CSV wrappers
    "plot_learning_curve_from_detailed",
    "plot_iteration_statistics_from_detailed",
    # Plotting - Multi-method comparison
    "plot_multimethod_comparison",
    "plot_multimethod_bar_summary",
    # Plotting - Diagnostics
    "plot_sigma_btw_comparison",
    "plot_covariance_heatmaps",
    "plot_covariance_diagonal",
    "plot_pointwise_sigma_btw",
    "export_sigma_btw_csv",
    # Plotting - Aggregation utility
    "aggregate_detailed_to_summary",
    # Orchestrator
    "EfficiencyOrchestrator",
]
