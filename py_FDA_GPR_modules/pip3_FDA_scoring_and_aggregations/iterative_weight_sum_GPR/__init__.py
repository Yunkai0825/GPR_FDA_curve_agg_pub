# iterative_weight_sum_GPR/__init__.py
"""
Iterative Weighted-Sum GPR Aggregation Method.

This subpackage implements the iterative variance-based weight optimization
for curve aggregation. It provides both curve-level and pointwise weight
optimization, along with within-model and between-model variance computation.

Submodules:
- iterative_gpr_core.py: Core aggregation function (compute_summary_gpr)
- iterative_export_helpers.py: CSV export for weights/history
- iterative_plot_helpers.py: Plotting (summary, weight dist/convergence, iterations)
- variance_calculation_helpers/: Within and between variance computation
- weight_calculation_helpers/: Curvewise and pointwise weight optimization

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Core algorithm
from .iterative_gpr_core import (
    SummaryGPRCoreResult,
    compute_summary_gpr,
)

# Variance helpers
from .variance_calculation_helpers.within_variance import compute_within_model_variances
from .variance_calculation_helpers.between_variance import (
    compute_between_model_variances,
    compute_total_model_variance,
    compute_pointwise_variance,
    compute_final_variance,
    compute_weighted_mean,
)

# Weight helpers
from .weight_calculation_helpers.curvewise_weight_helpers import (
    CurvewiseWeightResult,
    iterative_weight_optimization,
    compute_weights_equal,
)
from .weight_calculation_helpers.pointwise_weight_helpers import (
    PointwiseWeightResult,
    pointwise_weight_optimization,
)

# Export helpers
from .iterative_export_helpers import (
    export_iterative_weights_csv,
    export_iterative_history_csv,
)

# Plot helpers
from .iterative_plot_helpers import (
    plot_summary_gpr_from_csv,
    plot_weight_convergence_from_csv,
    plot_summary_curve_iterations_from_csv,
    plot_weight_distribution_from_csv,
)

# Orchestrator
from .iterative_orchestrator import IterativeGPROrchestrator

__all__ = [
    # Core
    'SummaryGPRCoreResult',
    'compute_summary_gpr',
    # Variance
    'compute_within_model_variances',
    'compute_between_model_variances',
    'compute_total_model_variance',
    'compute_pointwise_variance',
    'compute_final_variance',
    'compute_weighted_mean',
    # Weights
    'CurvewiseWeightResult',
    'PointwiseWeightResult',
    'iterative_weight_optimization',
    'pointwise_weight_optimization',
    'compute_weights_equal',
    # Export
    'export_iterative_weights_csv',
    'export_iterative_history_csv',
    # Plot
    'plot_summary_gpr_from_csv',
    'plot_weight_convergence_from_csv',
    'plot_summary_curve_iterations_from_csv',
    'plot_weight_distribution_from_csv',
    # Orchestrator
    'IterativeGPROrchestrator',
]
