"""
Within-variance computation helpers for Summary GPR.
"""

from __future__ import annotations

import numpy as np


def compute_within_model_variances(S_array: np.ndarray) -> np.ndarray:
    """Average squared std across points for each model (σ_within^2)."""
    n_points = S_array.shape[1]
    sigma_within_i_squared = np.sum(S_array ** 2, axis=1) / n_points
    return sigma_within_i_squared
