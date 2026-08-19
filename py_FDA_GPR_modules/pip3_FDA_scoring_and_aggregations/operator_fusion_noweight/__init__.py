# operator_fusion_noweight/__init__.py
"""
Operator Fusion (No-Weight) Aggregation Method.

This subpackage implements precision-space operator fusion with
Empirical Bayes between-variance estimation for curve aggregation.
Uses full covariance matrices from individual GPR posteriors.

Submodules:
- operator_fusion_weight_helpers.py: Core fusion algorithm
- operator_fusion_export_helpers.py: CSV/text export for weights/diagnostics
- operator_fusion_plot_helpers.py: Weight convergence plotting

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Core algorithm
from .operator_fusion_weight_helpers import (
    OperatorFusionResult,
    compute_operator_fusion,
)

# Export helpers
from .operator_fusion_export_helpers import (
    export_operator_weights_csv,
    export_operator_history_csv,
)

# Plot helpers
from .operator_fusion_plot_helpers import (
    plot_operator_weight_convergence,
)

# Orchestrator
from .operator_fusion_orchestrator import OperatorFusionOrchestrator

__all__ = [
    # Core
    'OperatorFusionResult',
    'compute_operator_fusion',
    # Export
    'export_operator_weights_csv',
    'export_operator_history_csv',
    # Plot
    'plot_operator_weight_convergence',
    # Orchestrator
    'OperatorFusionOrchestrator',
]
