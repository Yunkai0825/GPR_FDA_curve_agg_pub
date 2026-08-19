"""
Operator-fusion weight helpers (precision-space fusion + EB between-variance).

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import sys
import time
from pathlib import Path

# Import ScalingInfo from pip1
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


@dataclass
class OperatorFusionResult:
    y_mean_real: np.ndarray
    y_mean_norm: np.ndarray
    y_std_real: np.ndarray
    y_cov_real: np.ndarray
    y_cov_norm: np.ndarray
    weights: np.ndarray
    weight_history: List[np.ndarray]
    n_models: int
    n_points: int
    iterations: int
    between_variance_used: float


def _scale_multiplier(scaling: ScalingInfo) -> float:
    method = scaling.method
    if method in ("peak", "middle_average", "divide", "identity", ""):
        return float(scaling.params.get("factor", 1.0))
    if method == "standardize":
        return float(scaling.params.get("std", 1.0))
    if method == "minmax":
        min_val = float(scaling.params.get("min_val", 0.0))
        max_val = float(scaling.params.get("max_val", 1.0))
        feature_range = scaling.params.get("feature_range", (0, 1))
        return float((max_val - min_val) / (feature_range[1] - feature_range[0]))
    return 1.0


def _compute_aggregated_scaling(y_scalings: List[ScalingInfo], weights: np.ndarray) -> ScalingInfo:
    if len(y_scalings) == 0:
        return ScalingInfo.identity()
    if len(y_scalings) == 1:
        return y_scalings[0]

    w = weights / weights.sum()
    method = y_scalings[0].method
    all_same = all(s.method == method for s in y_scalings)

    if method in ("peak", "middle_average", "divide") or not all_same:
        factors = [s.params.get("factor", 1.0) for s in y_scalings]
        agg_factor = float(np.dot(w, factors))
        return ScalingInfo.divide_by_factor(agg_factor, method_name=method)

    if method == "standardize":
        means = np.array([s.params.get("mean", 0.0) for s in y_scalings])
        stds = np.array([s.params.get("std", 1.0) for s in y_scalings])
        agg_mean = float(np.dot(w, means))
        agg_std = float(np.dot(w, stds))
        return ScalingInfo.standardize(agg_mean, agg_std)

    if method == "minmax":
        mins = np.array([s.params.get("min_val", 0.0) for s in y_scalings])
        maxs = np.array([s.params.get("max_val", 1.0) for s in y_scalings])
        agg_min = float(np.dot(w, mins))
        agg_max = float(np.dot(w, maxs))
        feature_range = y_scalings[0].params.get("feature_range", (0, 1))
        return ScalingInfo.minmax(agg_min, agg_max, feature_range)

    if method.startswith("log"):
        shift = y_scalings[0].params.get("shift", 1e-9)
        base = y_scalings[0].params.get("base", "log10")
        return ScalingInfo.log_transform(shift=shift, base=base)

    if method == "identity":
        return ScalingInfo.identity()

    factors = [s.params.get("factor", 1.0) for s in y_scalings]
    agg_factor = float(np.dot(w, factors))
    return ScalingInfo.divide_by_factor(agg_factor, method_name=method)


def _estimate_between_variance_eb(
    y_list: List[np.ndarray],
    cov_list: List[np.ndarray],
    *,
    epsilon: float,
    grid_factor: float = 10.0,
    grid_size: int = 40,
) -> float:
    """Empirical Bayes grid search for scalar between-curve variance."""

    t_eb_start = time.perf_counter()
    n_models = len(y_list)
    n_points = y_list[0].shape[0]
    print(f"[Operator EB] starting grid search: {n_models} models, {n_points} points")

    y_stack = np.vstack(y_list)
    between_var = float(np.mean(np.var(y_stack, axis=0)))

    diag_means = [float(np.mean(np.diag(c))) for c in cov_list]
    base = max(1e-12, float(np.median(diag_means)))
    sigma2_min = max(base / grid_factor, between_var / grid_factor, epsilon)
    sigma2_max = max(base * grid_factor, between_var * grid_factor)

    sigmas = [0.0]
    sigmas += list(np.logspace(np.log10(sigma2_min), np.log10(sigma2_max), grid_size))
    n_sigmas = len(sigmas)
    print(f"[Operator EB] evaluating {n_sigmas} sigma candidates "
          f"(each needs {2*n_models+1} matrix inv of {n_points}x{n_points})")

    best_sigma2 = 0.0
    best_ll = -np.inf
    candidates: List[Tuple[float, float]] = []

    t_loop_start = time.perf_counter()
    for s_idx, sigma2 in enumerate(sigmas):
        t_step = time.perf_counter()
        precision_sum = np.zeros((n_points, n_points), dtype=float)
        precision_mean = np.zeros((n_points,), dtype=float)
        ll_terms = 0.0

        for y_i, cov_i in zip(y_list, cov_list):
            K = cov_i + np.eye(n_points) * (sigma2 + epsilon)
            try:
                inv_K = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                inv_K = np.linalg.pinv(K)
            sign, logdet = np.linalg.slogdet(K)
            if sign <= 0:
                ll_terms = -np.inf
                break
            precision_sum += inv_K
            precision_mean += inv_K @ y_i
            ll_terms += logdet

        if not np.isfinite(ll_terms):
            continue

        try:
            cov_post = np.linalg.inv(precision_sum)
        except np.linalg.LinAlgError:
            cov_post = np.linalg.pinv(precision_sum)
        g_sigma = cov_post @ precision_mean

        quad = 0.0
        for y_i, cov_i in zip(y_list, cov_list):
            K = cov_i + np.eye(n_points) * (sigma2 + epsilon)
            try:
                inv_K = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                inv_K = np.linalg.pinv(K)
            delta = y_i - g_sigma
            quad += float(delta.T @ inv_K @ delta)

        ll = -0.5 * (quad + ll_terms)
        candidates.append((sigma2, ll))
        if ll > best_ll:
            best_ll = ll
            best_sigma2 = sigma2

        # Progress every 10 steps or on first/last
        if s_idx == 0 or (s_idx + 1) % 10 == 0 or s_idx == n_sigmas - 1:
            elapsed_step = time.perf_counter() - t_step
            elapsed_total = time.perf_counter() - t_loop_start
            print(f"[Operator EB] grid step {s_idx+1}/{n_sigmas}: "
                  f"{elapsed_step:.2f}s this step, {elapsed_total:.1f}s total")

    t_eb_elapsed = time.perf_counter() - t_eb_start
    if candidates:
        print("[Operator EB] grid info: base={:.6g}, between_var={:.6g}, sigma2_min={:.6g}, sigma2_max={:.6g}".format(
            base, between_var, sigma2_min, sigma2_max
        ))
        top = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
        print("[Operator EB] top sigma_btw^2 candidates (norm units):")
        for rank, (s, llv) in enumerate(top, 1):
            print(f"  {rank}: sigma2={s:.6g}, loglike={llv:.6g}")
        print(f"[Operator EB] chosen sigma_btw^2={best_sigma2:.6g}")
    print(f"[Operator EB] grid search completed in {t_eb_elapsed:.2f}s")

    return best_sigma2


def compute_operator_fusion(
    y_norm_list: List[np.ndarray],
    cov_norm_list: List[np.ndarray],
    y_scalings: List[ScalingInfo],
    *,
    epsilon: float = 1e-8,
    weight_stabilizer: float = 1e-3,
    max_iterations: int = 50,
    convergence_tol: float = 1e-4,
    return_history: bool = False,
) -> OperatorFusionResult:
    """Iterative operator fusion with Mahalanobis-based weights."""
    t_fusion_start = time.perf_counter()
    if len(y_norm_list) == 0:
        raise ValueError("No curves provided for operator fusion")

    n_models = len(y_norm_list)
    n_points = y_norm_list[0].shape[0]
    print(f"[Operator fusion] starting: {n_models} models, {n_points} points, max_iter={max_iterations}")

    t_eb = time.perf_counter()
    between_variance = float(
        _estimate_between_variance_eb(
            y_norm_list,
            cov_norm_list,
            epsilon=epsilon,
        )
    )
    print(f"[Operator fusion] EB between-variance phase took {time.perf_counter()-t_eb:.2f}s")

    t_inv = time.perf_counter()
    cov_invs: List[np.ndarray] = []
    for idx, cov_i in enumerate(cov_norm_list):
        if cov_i.shape[0] != cov_i.shape[1]:
            raise ValueError(f"Covariance for model {idx} not square: {cov_i.shape}")
        if cov_i.shape[0] != n_points:
            raise ValueError(
                f"Covariance dimension {cov_i.shape} does not match n_points={n_points}"
            )
        cov_reg = cov_i + np.eye(n_points) * (epsilon + between_variance)
        try:
            inv_cov = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_reg)
        cov_invs.append(inv_cov)
    print(f"[Operator fusion] covariance inversions took {time.perf_counter()-t_inv:.2f}s")

    weights = np.ones(n_models, dtype=float)
    iterations = 0

    mean_post_norm = np.zeros(n_points, dtype=float)
    cov_post_norm = np.eye(n_points, dtype=float)
    weight_history: List[np.ndarray] = []

    t_iter_start = time.perf_counter()
    for it in range(max_iterations):
        t_it = time.perf_counter()
        iterations = it + 1
        precision_sum = np.zeros((n_points, n_points), dtype=float)
        precision_mean = np.zeros((n_points,), dtype=float)

        for w_i, inv_cov, mean_i in zip(weights, cov_invs, y_norm_list):
            precision_sum += w_i * inv_cov
            precision_mean += w_i * (inv_cov @ mean_i)

        cov_post_norm = np.linalg.inv(precision_sum)
        mean_post_norm = cov_post_norm @ precision_mean

        if return_history:
            weight_history.append((weights / weights.sum()).copy())

        new_weights = np.empty_like(weights)
        for idx, (mean_i, inv_cov) in enumerate(zip(y_norm_list, cov_invs)):
            delta = mean_i - mean_post_norm
            d_i = float(delta.T @ inv_cov @ delta)
            new_weights[idx] = n_points / (d_i + weight_stabilizer)

        w_norm_prev = weights / weights.sum()
        w_norm_new = new_weights / new_weights.sum()
        delta_w = np.linalg.norm(w_norm_new - w_norm_prev) / max(1.0, np.linalg.norm(w_norm_prev))

        weights = new_weights
        it_elapsed = time.perf_counter() - t_it
        if it == 0 or (it + 1) % 10 == 0 or delta_w < convergence_tol:
            print(f"[Operator fusion] iter {it+1}: delta_w={delta_w:.6e}, {it_elapsed:.3f}s")
        if delta_w < convergence_tol:
            break

    std_post_norm = np.sqrt(np.clip(np.diag(cov_post_norm), 0.0, None))

    weights_norm = weights / weights.sum()
    agg_scaling = _compute_aggregated_scaling(y_scalings, weights_norm)
    scale = _scale_multiplier(agg_scaling)

    cov_post_real = (scale ** 2) * cov_post_norm
    mean_post_real = agg_scaling.inverse_transform(mean_post_norm)
    std_post_real = np.sqrt(np.clip(np.diag(cov_post_real), 0.0, None))

    t_iter_elapsed = time.perf_counter() - t_iter_start
    print(f"[Operator fusion] iteration loop: {iterations} iters in {t_iter_elapsed:.2f}s")

    w_min = float(weights_norm.min())
    w_median = float(np.median(weights_norm))
    w_max = float(weights_norm.max())
    t_fusion_total = time.perf_counter() - t_fusion_start
    print(f"[Operator fusion] weight stats (normalized): min={w_min:.6g}, median={w_median:.6g}, max={w_max:.6g}")
    print(f"[Operator fusion] total elapsed: {t_fusion_total:.2f}s")

    return OperatorFusionResult(
        y_mean_real=mean_post_real,
        y_mean_norm=mean_post_norm,
        y_std_real=std_post_real,
        y_cov_real=cov_post_real,
        y_cov_norm=cov_post_norm,
        weights=weights_norm,
        weight_history=weight_history,
        n_models=n_models,
        n_points=n_points,
        iterations=iterations,
        between_variance_used=between_variance,
    )
