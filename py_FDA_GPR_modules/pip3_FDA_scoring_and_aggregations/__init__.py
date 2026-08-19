# pip3_FDA_scoring_and_aggregations/__init__.py
"""
=============================================================================
Pipeline 3: FDA Aggregations (Summary GPR)
=============================================================================

This module aggregates individual Gaussian Process Regression (GPR) models
to create summary GPR models. It loads metadata from individual GPR CSVs,
regenerates scaling information, and performs weighted aggregation.

Three aggregation methods are available, each self-contained in its own
subpackage with core algorithm, export helpers, and plot helpers:

1. iterative_weight_sum_GPR/ — Iterative variance-based weighted sum
   - iterative_gpr_core.py: Core aggregation (compute_summary_gpr)
   - iterative_export_helpers.py: Weight/history CSV export
   - iterative_plot_helpers.py: Summary, weight, convergence, iteration plots
   - variance_calculation_helpers/: Within and between variance
   - weight_calculation_helpers/: Curvewise and pointwise optimization

2. operator_fusion_noweight/ — Precision-space operator fusion
   - operator_fusion_weight_helpers.py: Core fusion algorithm
   - operator_fusion_export_helpers.py: Weight/diagnostics export
   - operator_fusion_plot_helpers.py: Weight convergence plot

3. functional_GPR/ — Functional GPR (FGPR)
   - fgpr_helpers.py: Core FGPR algorithm
   - fgpr_export_helpers.py: Weight/diagnostics export
   - fgpr_plot_helpers.py: Weight distribution plot

4. student_t_agg_iterative/ — Student-t robust curve aggregation
   - student_t_core.py: Core EM/IRLS algorithm (compute_student_t_aggregation)
   - student_t_export_helpers.py: Weight/diagnostics/covariance/iteration export
   - student_t_plot_helpers.py: Curve, weight, convergence, covariance plots

Each method folder also contains an orchestrator that owns
run / export / plot for that method:
- iterative_orchestrator.py
- operator_fusion_orchestrator.py
- fgpr_orchestrator.py

Shared infrastructure (parent level):
- summary_gpr_config.py: Configuration dataclasses
- summary_gpr_loader.py: Load GPRs with metadata
- summary_gpr_plotting.py: Generic base plots + comparison plots
- comparison_orchestrator.py: Cross-method comparison + sigma calibration
- pip3_summary_gpr_orchestrator.py: High-level I/O orchestrator (delegates to method orchestrators)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
=============================================================================
"""

from .summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams

# Iterative Weighted-Sum GPR (Method 1)
from .iterative_weight_sum_GPR import (
    # Core
    SummaryGPRCoreResult,
    compute_summary_gpr,
    # Variance
    compute_within_model_variances,
    compute_between_model_variances,
    compute_total_model_variance,
    compute_pointwise_variance,
    compute_final_variance,
    compute_weighted_mean,
    # Weights
    CurvewiseWeightResult,
    PointwiseWeightResult,
    iterative_weight_optimization,
    pointwise_weight_optimization,
    compute_weights_equal,
    # Export
    export_iterative_weights_csv,
    export_iterative_history_csv,
    # Plot
    plot_summary_gpr_from_csv,
    plot_weight_convergence_from_csv,
    plot_summary_curve_iterations_from_csv,
    plot_weight_distribution_from_csv,
    IterativeGPROrchestrator,
)

# Operator Fusion (Method 2)
from .operator_fusion_noweight import (
    OperatorFusionResult,
    compute_operator_fusion,
    export_operator_weights_csv,
    export_operator_history_csv,
    plot_operator_weight_convergence,
    OperatorFusionOrchestrator,
)

# Functional GPR (Method 3)
from .functional_GPR import (
    FGPRResult,
    compute_fgpr,
    fit_sigma_btw,
    export_fgpr_weights_csv,
    export_fgpr_diagnostics,
    plot_fgpr_weight_distribution,
    FGPROrchestrator,
)

# Student-t Robust Aggregation (Method 4)
from .student_t_agg_iterative import (
    StudentTResult,
    compute_student_t_aggregation,
    export_student_t_weights_csv,
    export_student_t_diagnostics,
    export_student_t_curve_csv,
    export_student_t_covariance_csv,
    export_student_t_iteration_history_csv,
    export_student_t_sigma_calibration_csv,
    plot_student_t_curve,
    plot_student_t_weight_distribution,
    plot_student_t_covariance_heatmap,
    plot_student_t_diagonal_std,
    plot_student_t_sigma_calibration,
    plot_student_t_weight_convergence,
    plot_student_t_curve_iterations,
    plot_student_t_energy_convergence,
    StudentTOrchestrator,
)

# Loader
from .summary_gpr_loader import (
    IndividualGPRData,
    load_individual_gpr_with_metadata,
    load_all_individual_gprs,
    group_gprs_by_key,
)

# Comparison orchestrator
from .comparison_orchestrator import ComparisonOrchestrator

# Main orchestrator
from .pip3_summary_gpr_orchestrator import (
    SummaryGPRResult,
    SummaryGPROrchestrator,
)

# Plotting utilities — generic base functions + comparison plots
from .summary_gpr_plotting import (
    # CSV loaders
    load_summary_gpr_csv,
    load_weight_history_csv,
    load_curve_history_csv,
    load_converged_weights_csv,
    # Generic base plot functions
    plot_summary_gpr,
    plot_weight_convergence,
    plot_summary_curve_iterations,
    plot_weight_distribution,
    # Comparison plots
    plot_summary_comparison,
    plot_summary_comparison_normalized,
)

__all__ = [
    # Config
    'SummaryGPRConfig',
    'SummaryGPRHyperParams',
    
    # Iterative Weighted-Sum GPR (Method 1)
    'SummaryGPRCoreResult',
    'compute_summary_gpr',
    'compute_within_model_variances',
    'compute_between_model_variances',
    'compute_total_model_variance',
    'compute_pointwise_variance',
    'compute_final_variance',
    'compute_weighted_mean',
    'CurvewiseWeightResult',
    'PointwiseWeightResult',
    'iterative_weight_optimization',
    'pointwise_weight_optimization',
    'compute_weights_equal',
    'export_iterative_weights_csv',
    'export_iterative_history_csv',
    'plot_summary_gpr_from_csv',
    'plot_weight_convergence_from_csv',
    'plot_summary_curve_iterations_from_csv',
    'plot_weight_distribution_from_csv',
    
    # Operator Fusion (Method 2)
    'OperatorFusionResult',
    'compute_operator_fusion',
    'export_operator_weights_csv',
    'export_operator_history_csv',
    'plot_operator_weight_convergence',
    
    # Functional GPR (Method 3)
    'FGPRResult',
    'compute_fgpr',
    'fit_sigma_btw',
    'export_fgpr_weights_csv',
    'export_fgpr_diagnostics',
    'plot_fgpr_weight_distribution',
    
    # Student-t Robust Aggregation (Method 4)
    'StudentTResult',
    'compute_student_t_aggregation',
    'export_student_t_weights_csv',
    'export_student_t_diagnostics',
    'export_student_t_curve_csv',
    'export_student_t_covariance_csv',
    'export_student_t_iteration_history_csv',
    'export_student_t_sigma_calibration_csv',
    'plot_student_t_curve',
    'plot_student_t_weight_distribution',
    'plot_student_t_covariance_heatmap',
    'plot_student_t_diagonal_std',
    'plot_student_t_sigma_calibration',
    'plot_student_t_weight_convergence',
    'plot_student_t_curve_iterations',
    'plot_student_t_energy_convergence',
    'StudentTOrchestrator',
    
    # Loader
    'IndividualGPRData',
    'load_individual_gpr_with_metadata',
    'load_all_individual_gprs',
    'group_gprs_by_key',
    
    # Method orchestrators
    'IterativeGPROrchestrator',
    'OperatorFusionOrchestrator',
    'FGPROrchestrator',
    'ComparisonOrchestrator',
    # Main orchestrator
    'SummaryGPRResult',
    'SummaryGPROrchestrator',
    
    # Plotting — CSV loaders
    'load_summary_gpr_csv',
    'load_weight_history_csv',
    'load_curve_history_csv',
    'load_converged_weights_csv',
    # Plotting — Generic base functions
    'plot_summary_gpr',
    'plot_weight_convergence',
    'plot_summary_curve_iterations',
    'plot_weight_distribution',
    # Plotting — Comparison
    'plot_summary_comparison',
    'plot_summary_comparison_normalized',
]
