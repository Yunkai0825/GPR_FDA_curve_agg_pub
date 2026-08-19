# student_t_agg_iterative/__init__.py
"""
Student-t Robust Curve Aggregation Method.

This subpackage implements the Student-t robust aggregation method
from the derivation in _ref/GPR_derivation_Dirac_notation(R).md, Section 1.x.

Each curve is modelled as:
    |m_r⟩ | |m⟩, λ_r  ~  N(|m⟩, (1/λ_r) Ĉ_{e,r})
    λ_r  ~  Gamma(ν/2, ν/2)

This yields curvewise weights  w_r = (ν + N) / (ν + d_r)  via EM/IRLS
iterations, where d_r = ⟨ε_r | Ĉ_{e,r}⁻¹ | ε_r⟩ is the Mahalanobis
energy of curve r.

Submodules:
- student_t_core.py: Core algorithm (compute_student_t_aggregation, StudentTResult)
- student_t_orchestrator.py: Orchestrator with run / export / plot
- student_t_export_helpers.py: CSV/text export for weights, diagnostics, covariance
- student_t_plot_helpers.py: Plot functions (curve, weights, convergence, covariance)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Core algorithm
from .student_t_core import (
    StudentTResult,
    compute_student_t_aggregation,
)

# Export helpers
from .student_t_export_helpers import (
    export_student_t_weights_csv,
    export_student_t_diagnostics,
    export_student_t_curve_csv,
    export_student_t_covariance_csv,
    export_student_t_iteration_history_csv,
    export_student_t_sigma_calibration_csv,
)

# Plot helpers
from .student_t_plot_helpers import (
    plot_student_t_curve,
    plot_student_t_weight_distribution,
    plot_student_t_covariance_heatmap,
    plot_student_t_diagonal_std,
    plot_student_t_sigma_calibration,
    plot_student_t_weight_convergence,
    plot_student_t_curve_iterations,
    plot_student_t_energy_convergence,
)

# Orchestrator
from .student_t_orchestrator import StudentTOrchestrator

__all__ = [
    # Core
    'StudentTResult',
    'compute_student_t_aggregation',
    # Export
    'export_student_t_weights_csv',
    'export_student_t_diagnostics',
    'export_student_t_curve_csv',
    'export_student_t_covariance_csv',
    'export_student_t_iteration_history_csv',
    'export_student_t_sigma_calibration_csv',
    # Plot
    'plot_student_t_curve',
    'plot_student_t_weight_distribution',
    'plot_student_t_covariance_heatmap',
    'plot_student_t_diagonal_std',
    'plot_student_t_sigma_calibration',
    'plot_student_t_weight_convergence',
    'plot_student_t_curve_iterations',
    'plot_student_t_energy_convergence',
    # Orchestrator
    'StudentTOrchestrator',
]
