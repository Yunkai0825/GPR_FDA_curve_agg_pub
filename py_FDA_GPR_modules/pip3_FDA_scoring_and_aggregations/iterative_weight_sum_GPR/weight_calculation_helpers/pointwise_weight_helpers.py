"""
Pointwise weight optimization helpers for Summary GPR.
"""

from __future__ import annotations

import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class PointwiseWeightResult:
    weights: np.ndarray
    weight_history: List[np.ndarray]
    curve_history: List[np.ndarray]
    n_iterations: int
    converged: bool
    final_delta: float


def pointwise_weight_optimization(
    y_array: np.ndarray,
    S_array: np.ndarray,
    *,
    epsilon: float = 1e-12,
    convergence_tol: float = 1e-6,
    max_iterations: int = 10000,
    return_history: bool = False,
    verbose: bool = True,
) -> PointwiseWeightResult:
    """Optimize pointwise weights via fixed-point iteration."""
    m, n = y_array.shape
    W = np.full((m, n), 1.0 / m, dtype=float)

    weight_history: List[np.ndarray] = [W.copy()] if return_history else []
    curve_history: List[np.ndarray] = []

    converged = False
    final_delta = 0.0
    iteration = 0

    for it in range(max_iterations):
        iteration = it + 1
        W_prev = W.copy()

        I_bar = (W * y_array).sum(axis=0, keepdims=True)

        if return_history:
            curve_history.append(I_bar.flatten().copy())

        V = (y_array - I_bar) ** 2 + S_array ** 2 + epsilon

        W = 1.0 / V
        W /= W.sum(axis=0, keepdims=True)

        final_delta = np.linalg.norm(W - W_prev) / np.sqrt(W.size)

        if return_history:
            weight_history.append(W.copy())

        if final_delta < convergence_tol:
            converged = True
            if verbose:
                print(f"  Pointwise converged after {iteration} iterations (delta_rms={final_delta:.2e})")
            break

    return PointwiseWeightResult(
        weights=W,
        weight_history=[w.mean(axis=1) for w in weight_history] if weight_history else [],
        curve_history=curve_history,
        n_iterations=iteration,
        converged=converged,
        final_delta=final_delta,
    )
