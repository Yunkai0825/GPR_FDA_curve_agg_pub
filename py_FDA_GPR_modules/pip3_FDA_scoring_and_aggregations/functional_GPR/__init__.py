# functional_GPR/__init__.py
"""
Functional Gaussian Process Regression (FGPR) Aggregation Method.

This subpackage implements the FGPR aggregation method from the derivation
in _ref/GPR_derivation_Dirac_notation(R).md, Section 1.2.

Uses profile negative log-likelihood for optimal σ_btw² estimation
and full covariance-based precision fusion.

Submodules:
- fgpr_helpers.py: Core FGPR algorithm (compute_fgpr, fit_sigma_btw)
- fgpr_export_helpers.py: CSV/text export for weights/diagnostics
- fgpr_plot_helpers.py: Weight distribution plotting

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Core algorithm
from .fgpr_helpers import (
    FGPRResult,
    FGPRCurveOperators,
    compute_fgpr,
    compute_fgpr_structured,
    fit_sigma_btw,
    build_per_curve_operators,
    compute_scalar_weights,
    project_R_mean,
    project_R_cov_agg,       # deprecated – kept for back-compat
    project_R_cov_meas,       # deprecated – kept for back-compat
    rescale_fgpr_to_observation,
)

# Structured between-curve covariance
from .fgpr_structured_btw import (
    StructuredBtwParams,
    StructuredBtwConfig,
    fit_structured_btw,
    effective_sigma2_btw,
)

# Export helpers
from .fgpr_export_helpers import (
    export_fgpr_weights_csv,
    export_fgpr_diagnostics,
    export_fgpr_iteration_history_csv,
)

# Plot helpers
from .fgpr_plot_helpers import (
    plot_fgpr_weight_distribution,
    plot_fgpr_weight_convergence,
    plot_fgpr_curve_iterations,
)

# Orchestrator
from .fgpr_orchestrator import FGPROrchestrator

__all__ = [
    # Core
    'FGPRResult',
    'compute_fgpr',
    'compute_fgpr_structured',
    'fit_sigma_btw',
    # Structured btw
    'StructuredBtwParams',
    'StructuredBtwConfig',
    'fit_structured_btw',
    'effective_sigma2_btw',
    # Export
    'export_fgpr_weights_csv',
    'export_fgpr_diagnostics',
    'export_fgpr_iteration_history_csv',
    # Plot
    'plot_fgpr_weight_distribution',
    'plot_fgpr_weight_convergence',
    'plot_fgpr_curve_iterations',
    # Orchestrator
    'FGPROrchestrator',
]
