# weight_calculation_helpers/__init__.py
"""
Weight calculation helpers for iterative weighted-sum GPR.

- curvewise_weight_helpers: Curve-level weight optimization
- pointwise_weight_helpers: Pointwise weight optimization
"""

from .curvewise_weight_helpers import (
    CurvewiseWeightResult,
    iterative_weight_optimization,
    compute_weights_equal,
)
from .pointwise_weight_helpers import (
    PointwiseWeightResult,
    pointwise_weight_optimization,
)
