# student_t_agg_iterative/student_t_core.py
"""
Student-t robust curve aggregation in Ĥ_J.

Implements the EM/IRLS aggregation from Section 2.2.5c of the derivation:

    |m_r⟩ | |m⟩, λ_r  ~  N(|m⟩, (1/λ_r) Ĉ_{e,r})
    λ_r ~ Gamma(νN/2, νN/2)

    ⟹  w_r = E[λ_r | m_r, m] = (νN + N) / (νN + d_r)     [Eqn 31]
        d_r = ⟨ε_r | Ĉ_{e,r}⁻¹ | ε_r⟩                    [Eqn 29]

    |m⟩ = (Σ_r w_r Ĉ_{e,r}⁻¹)⁻¹ (Σ_r w_r Ĉ_{e,r}⁻¹ |m_r⟩)  [Eqn 28c]
    Ĉ_agg = (Σ_r w_r Ĉ_{e,r}⁻¹)⁻¹                            [Eqn 28a]

Iteration order per step n:
    1. Optimise σ²_btw^(n) with v_r^(n-1)   (diagonal NLL, Eqn 27)
    2. Compute |m_agg^(n)⟩, Ĉ_agg^(n)       (Eqn 28)
    3. Compute d_r^(n)                       (Eqn 29)
    4. Optimise ν^(n)                        (Student-t MLE, Eqn 30)
    5. Update v_r^(n) = (ν^(n)N+N)/(ν^(n)N+d_r^(n))   (Eqn 31)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln
from scipy.linalg import cho_factor, cho_solve

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StudentTResult:
    """
    Result from Student-t robust curve aggregation.

    Attributes
    ----------
    y_mean_real : np.ndarray
        Aggregated posterior mean in real (observation) scale.
    y_mean_norm : np.ndarray
        Aggregated posterior mean in normalised scale.
    y_std_real : np.ndarray
        Posterior std (sqrt(diag(C_agg))) in real scale.
    y_std_norm : np.ndarray
        Posterior std in normalised scale.
    y_cov_real : np.ndarray
        Full posterior covariance in real scale, shape (N, N).
    y_cov_norm : np.ndarray
        Full posterior covariance in normalised scale, shape (N, N).
    y_std_predictive_real : np.ndarray
        Predictive std in real scale: sqrt(diag(s_agg² (C̃_agg + σ²_btw I))).
    y_std_predictive_norm : np.ndarray
        Predictive std in normalised scale.
    y_cov_predictive_real : np.ndarray
        Full predictive covariance in real scale, shape (N, N).
    y_cov_predictive_norm : np.ndarray
        Full predictive covariance in normalised scale, shape (N, N).
    sigma_btw_squared : float
        Optimised between-curve variance σ²_btw in normalised units.
    nu : float
        Student-t degrees-of-freedom parameter used.
    weights : np.ndarray
        Final curvewise Student-t weights (normalised to sum to 1).
    weights_raw : np.ndarray
        Raw (un-normalised) curvewise weights w_r = (νN+N)/(νN+d_r).
    energies : np.ndarray
        Final whole-curve Mahalanobis energies d_r.
    n_models : int
        Number of curves aggregated.
    n_points : int
        Number of grid points N.
    iterations : int
        Number of IRLS iterations performed.
    converged : bool
        Whether the IRLS loop converged.
    max_weight_delta : float
        Maximum normalised weight change at last iteration.
    weight_history : List[np.ndarray]
        Weight vector at each iteration (normalised).
    curve_history : List[np.ndarray]
        Aggregated mean curve (normalised) at each iteration.
    energy_history : List[np.ndarray]
        Per-curve energies at each iteration.
    s_agg : float
        Weighted-average scale factor for norm→real conversion.
    """
    y_mean_real: np.ndarray
    y_mean_norm: np.ndarray
    y_std_real: np.ndarray
    y_std_norm: np.ndarray
    y_cov_real: np.ndarray
    y_cov_norm: np.ndarray
    # Predictive covariance:  C_pred = C_agg + σ²_btw I
    y_std_predictive_real: np.ndarray
    y_std_predictive_norm: np.ndarray
    y_cov_predictive_real: np.ndarray
    y_cov_predictive_norm: np.ndarray
    sigma_btw_squared: float
    nu: float
    weights: np.ndarray            # normalised
    weights_raw: np.ndarray        # un-normalised
    energies: np.ndarray           # final d_r per curve
    n_models: int
    n_points: int
    iterations: int
    converged: bool
    max_weight_delta: float
    weight_history: List[np.ndarray] = field(default_factory=list)
    curve_history: List[np.ndarray] = field(default_factory=list)
    energy_history: List[np.ndarray] = field(default_factory=list)
    sigma_btw_history: List[float] = field(default_factory=list)
    nu_history: List[float] = field(default_factory=list)
    s_agg: float = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_multiplier(scaling: ScalingInfo) -> float:
    """Return multiplicative scale factor from a ScalingInfo object."""
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


def _compute_aggregated_scaling(
    y_scalings: List[ScalingInfo], weights: np.ndarray
) -> ScalingInfo:
    """Weighted aggregation of ScalingInfo objects."""
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
        return ScalingInfo.standardize(float(np.dot(w, means)),
                                       float(np.dot(w, stds)))

    if method == "minmax":
        mins = np.array([s.params.get("min_val", 0.0) for s in y_scalings])
        maxs = np.array([s.params.get("max_val", 1.0) for s in y_scalings])
        feature_range = y_scalings[0].params.get("feature_range", (0, 1))
        return ScalingInfo.minmax(float(np.dot(w, mins)),
                                  float(np.dot(w, maxs)), feature_range)

    if method.startswith("log"):
        shift = y_scalings[0].params.get("shift", 1e-9)
        base = y_scalings[0].params.get("base", "log10")
        return ScalingInfo.log_transform(shift=shift, base=base)

    if method == "identity":
        return ScalingInfo.identity()

    factors = [s.params.get("factor", 1.0) for s in y_scalings]
    agg_factor = float(np.dot(w, factors))
    return ScalingInfo.divide_by_factor(agg_factor, method_name=method)


def _fit_sigma_btw_diagonal(
    m_list: List[np.ndarray],
    Cpost_list: List[np.ndarray],
    *,
    weights: Optional[np.ndarray] = None,
    bounds: Tuple[float, Optional[float]] = (0.0, None),
    jitter: float = 1e-12,
    verbose: bool = True,
) -> Tuple[float, object]:
    """
    Diagonal-NLL 1-D optimisation of sigma_btw^2.

    When *weights* is supplied the per-curve contribution to the NLL
    is scaled by normalised w_r, so outlier curves identified by the
    Student-t E-step contribute less.

    Returns
    -------
    sigma2_star : float
    opt_res : scipy OptimizeResult
    """
    R = len(m_list)
    N = m_list[0].size
    m_stack = np.vstack(m_list)          # (R, N)
    V = np.vstack([np.diag(C) for C in Cpost_list])  # (R, N)

    # Normalise weights (uniform when not provided)
    if weights is not None:
        w = np.asarray(weights, float)
        w_sum = w.sum()
        w = w / w_sum if w_sum > 0 else np.ones(R) / R
    else:
        w = np.ones(R) / R

    lo = bounds[0]
    hi = bounds[1]
    if hi is None:
        spread = float(np.median(np.var(m_stack, axis=0)))
        within = float(np.median([np.median(np.diag(C)) for C in Cpost_list]))
        hi = max(1e-12, 50.0 * (spread + within))

    def nll_diag(sigma2: float) -> float:
        sigma2 = max(sigma2, 0.0)
        S = V + sigma2 + jitter                          # (R, N)
        prec = 1.0 / S                                   # (R, N)
        # Weighted profiled mean per grid point
        w_prec = w[:, None] * prec                        # (R, N)
        mu_j = (w[:, None] * prec * m_stack).sum(axis=0) / w_prec.sum(axis=0)
        resid = m_stack - mu_j[None, :]
        # Weighted NLL: scale each curve's contribution by w_r
        return float(0.5 * np.sum(w[:, None] * (resid**2 / S + np.log(S))))

    res = minimize_scalar(nll_diag, bounds=(lo, hi), method="bounded")
    sigma2_star = float(res.x)

    if verbose:
        print(f"[Student-t] sigma_btw^2 = {sigma2_star:.6g}  "
              f"(weighted diag-NLL = {res.fun:.6g})")
        print(f"[Student-t] search bounds = ({lo:.6g}, {hi:.6g})")

    return sigma2_star, res


def _cholesky_solve(L_r: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve L L^T x = b using Cholesky factors."""
    return cho_solve((L_r, True), b, check_finite=False)


def _optimize_nu(
    energies: np.ndarray,
    N: int,
    log_det_Ce: Optional[np.ndarray] = None,
    nu_bounds: Tuple[float, float] = (1.0, 500.0),
    verbose: bool = False,
) -> float:
    """
    Optimize Student-t degrees-of-freedom ς via Eqn (30).

    log P = Σ_r [ log Γ((ςN+N)/2) - log Γ(ςN/2)
                  - (N/2) log(ςN π) - Σ_i log(λ_{e,r,i})
                  - ((ςN+N)/2) log(1 + d_r/(ςN)) ]

    Since the log-det term Σ_i log(λ_{e,r,i}) does not depend on ς
    (our between-model variance is white noise, so the quality score
    does not impact the covariance structure), it drops out of the
    optimisation as noted in the derivation.

    Parameters
    ----------
    energies : (R,) array
        Mahalanobis distances d_r for each curve.
    N : int
        Number of grid points.
    log_det_Ce : (R,) array, optional
        log-determinant of C_{e,r} for each curve. If None, the constant
        term is ignored (it does not affect the argmax in ς).
    nu_bounds : (lo, hi)
        Search interval for ς.
    verbose : bool
        Print optimisation result.

    Returns
    -------
    nu_star : float
        Optimised degrees-of-freedom.
    """
    R = len(energies)

    def neg_log_lik(nu_val: float) -> float:
        nuN = nu_val * N
        half_nuN = nuN / 2.0
        half_nuN_N = (nuN + N) / 2.0   # = N*(nu_val+1)/2
        ll = R * (gammaln(half_nuN_N) - gammaln(half_nuN)
                  - (N / 2.0) * np.log(nuN * np.pi))
        ll -= half_nuN_N * np.sum(np.log(1.0 + energies / nuN))
        if log_det_Ce is not None:
            ll -= 0.5 * np.sum(log_det_Ce)
        return -ll

    res = minimize_scalar(neg_log_lik,
                          bounds=nu_bounds,
                          method="bounded")
    nu_star = float(res.x)

    # --- Boundary-hit diagnostic ---
    lo, hi = nu_bounds
    at_lower = (nu_star - lo) < 1e-3 * (hi - lo)
    at_upper = (hi - nu_star) < 1e-3 * (hi - lo)

    if verbose:
        bound_tag = ""
        if at_lower:
            # Probe slightly below the lower bound to test gradient direction
            probe = max(lo * 0.5, 0.1)
            nll_at_lo = neg_log_lik(lo)
            nll_at_probe = neg_log_lik(probe)
            if nll_at_probe < nll_at_lo:
                bound_tag = (f"  ** HIT LOWER BOUND (nu={lo}); "
                             f"NLL({probe:.2g})={nll_at_probe:.4g} < "
                             f"NLL({lo})={nll_at_lo:.4g} => "
                             f"true optimum lies below {lo}")
            else:
                bound_tag = (f"  ** AT LOWER BOUND (nu={lo}); "
                             f"NLL({probe:.2g})={nll_at_probe:.4g} >= "
                             f"NLL({lo})={nll_at_lo:.4g} => "
                             f"true optimum is near {lo}")
        elif at_upper:
            bound_tag = f"  ** HIT UPPER BOUND (nu={hi})"

        print(f"[Student-t] nu optimised: {nu_star:.4f}  "
              f"(nuN = {nu_star * N:.1f})  "
              f"(neg-loglik = {res.fun:.6g})")
        if bound_tag:
            print(f"[Student-t] {bound_tag}")

    return nu_star


def _build_effective_covariances(
    cov_norm_list: List[np.ndarray],
    sigma2_btw: float,
    epsilon: float,
    R: int,
    N: int,
) -> Tuple[List[np.ndarray], List, List[np.ndarray]]:
    """Build C_{e,r} = C_{post,r} + sigma_btw^2 I + eps I and Cholesky factors."""
    I_N = np.eye(N)
    Ce_list: List[np.ndarray] = []
    L_list: List = []
    Ce_inv_list: List[np.ndarray] = []
    for r_idx in range(R):
        Ce_r = cov_norm_list[r_idx] + (sigma2_btw + epsilon) * I_N
        Ce_r = 0.5 * (Ce_r + Ce_r.T)  # enforce symmetry
        Ce_list.append(Ce_r)
        try:
            L_r = cho_factor(Ce_r, lower=True, check_finite=False)
            L_list.append(L_r)
            Ce_inv = cho_solve(L_r, I_N, check_finite=False)
        except np.linalg.LinAlgError:
            Ce_inv = np.linalg.pinv(Ce_r)
            L_list.append(None)
        Ce_inv_list.append(Ce_inv)
    return Ce_list, L_list, Ce_inv_list


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def compute_student_t_aggregation(
    y_norm_list: List[np.ndarray],
    cov_norm_list: List[np.ndarray],
    y_scalings: List[ScalingInfo],
    *,
    nu: float = 5.0,
    optimize_nu: bool = True,
    nu_bounds: Tuple[float, float] = (1.0, 500.0),
    nu_lb_adaptive: bool = False,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    epsilon: float = 1e-12,
    outlier_energy_threshold: Optional[float] = None,
    verbose: bool = True,
) -> StudentTResult:
    """
    Student-t robust aggregation of replicated curves (Section 1.x).

    Uses a Gamma(νN/2, νN/2) prior on the per-curve precision λ_r,
    where νN = ν × N is the *effective* degrees-of-freedom.  The
    user-facing parameters ``nu`` and ``nu_bounds`` are specified in
    terms of the effective DOF (νN), and are internally divided by N
    to obtain the raw ν used in the weight formula.

    Parameters
    ----------
    y_norm_list : list of (N,) arrays
        Per-curve posterior means in normalised units.
    cov_norm_list : list of (N, N) arrays
        Per-curve posterior covariances in normalised units.
    y_scalings : list of ScalingInfo
        Per-curve Y-axis scaling info.
    nu : float
        Initial *effective* degrees-of-freedom νN (= ν × N).
        Small → heavy tails (robust); large → Gaussian limit.
        Internally converted to raw ν = nu / N.
    optimize_nu : bool
        If True, ν is optimised at each iteration via Eqn (30).
        If False, ν is kept fixed at the supplied value.
    nu_bounds : (float, float)
        Search bounds for the *effective* DOF νN (= ν × N).
        Internally converted to raw ν bounds by dividing by N.
    nu_lb_adaptive : bool
        If True, clamp the lower bound to 1/N (effective DOF) so
        Gamma(νN/2, νN/2) can reach ultra-heavy tails.
        Default False → lower bound stays at user-specified value.
    max_iterations : int
        Maximum IRLS iterations.
    convergence_tol : float
        Convergence tolerance on normalised weight change (Eqn 32).
    epsilon : float
        Numerical jitter for covariance regularisation.
    outlier_energy_threshold : float, optional
        If set, curves with d_r > threshold get w_r = 0.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    StudentTResult
    """
    t_start = time.perf_counter()

    if len(y_norm_list) == 0:
        raise ValueError("No curves provided for Student-t aggregation")

    R = len(y_norm_list)
    N = y_norm_list[0].shape[0]
    I_N = np.eye(N)

    # Validate
    for idx, (y, c) in enumerate(zip(y_norm_list, cov_norm_list)):
        if y.shape != (N,):
            raise ValueError(f"Curve {idx} mean shape {y.shape} != ({N},)")
        if c.shape != (N, N):
            raise ValueError(f"Curve {idx} cov shape {c.shape} != ({N},{N})")

    # ---- Convert user-facing effective-DOF (νN) to raw ν = νN / N ----
    # The user specifies nu and nu_bounds in terms of effective DOF νN.
    # Internally we work with raw ν; the weight formula uses νN = ν*N.
    nu = nu / N
    nu_lb_eff = nu_bounds[0]
    if nu_lb_adaptive:
        nu_lb_eff = max(nu_lb_eff, 1.0 / N)   # clamp to 1/N for ultra-heavy tails
    nu_bounds = (nu_lb_eff / N, nu_bounds[1] / N)

    if verbose:
        print(f"[Student-t] Starting: {R} curves, {N} grid points, "
              f"nu={nu:.6g} (nuN={nu*N:.4g}) "
              f"({'optimise' if optimize_nu else 'fixed'})")

    # ------------------------------------------------------------------
    # Step 0: Initial sigma_btw^2 with uniform weights
    # ------------------------------------------------------------------
    sigma2_btw, opt_res = _fit_sigma_btw_diagonal(
        y_norm_list, cov_norm_list,
        weights=None,          # uniform for initialisation
        jitter=epsilon, verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Step 1: Build effective covariance and Cholesky factors
    #   C_{e,r} = C_{post,r} + sigma_btw^2 I + eps I
    # ------------------------------------------------------------------
    Ce_list, L_list, Ce_inv_list = _build_effective_covariances(
        cov_norm_list, sigma2_btw, epsilon, R, N,
    )

    # ------------------------------------------------------------------
    # Step 2: Initialise m^(0) via unweighted precision average
    # ------------------------------------------------------------------
    A_init = np.zeros((N, N))
    b_init = np.zeros(N)
    for r_idx in range(R):
        A_init += Ce_inv_list[r_idx]
        b_init += Ce_inv_list[r_idx] @ y_norm_list[r_idx]
    mu_agg = np.linalg.solve(A_init, b_init)

    if verbose:
        print(f"[Student-t] Initial mean computed (unweighted precision avg)")

    # ------------------------------------------------------------------
    # Step 3: IRLS iterations with joint sigma_btw^2 update
    # ------------------------------------------------------------------
    weight_history: List[np.ndarray] = []
    curve_history: List[np.ndarray] = []
    energy_history: List[np.ndarray] = []
    sigma_btw_history: List[float] = [sigma2_btw]
    nu_history: List[float] = [nu]

    weights_raw = np.ones(R)
    converged = False
    max_weight_delta = np.inf
    iterations = 0

    t_iter_start = time.perf_counter()

    for it in range(max_iterations):
        iterations = it + 1

        # ==============================================================
        # Derivation step order (per iteration n):
        #   1. Optimise σ_btw^(n) with v_r^(n-1)          [Eqn 27]
        #   2. Compute m_agg^(n), C_agg^(n)                [Eqn 28]
        #   3. Compute d_r^(n)                             [Eqn 29]
        #   4. Optimise ς^(n)                              [Eqn 30]
        #   5. Update  v_r^(n) = (ς^(n)N+N)/(ς^(n)N+d_r^(n))[Eqn 31]
        # ==============================================================

        # -- Step 1 (Eqn 27): re-fit sigma_btw^2 with previous weights -
        sigma2_btw_new, _ = _fit_sigma_btw_diagonal(
            y_norm_list, cov_norm_list,
            weights=weights_raw,
            jitter=epsilon, verbose=False,
        )
        sigma_btw_history.append(sigma2_btw_new)

        # Rebuild C_{e,r} if sigma_btw^2 changed appreciably
        sigma_rel_change = (abs(sigma2_btw_new - sigma2_btw)
                            / max(abs(sigma2_btw), 1e-15))
        if sigma_rel_change > 1e-10:
            sigma2_btw = sigma2_btw_new
            Ce_list, L_list, Ce_inv_list = _build_effective_covariances(
                cov_norm_list, sigma2_btw, epsilon, R, N,
            )

        # -- Step 2 (Eqn 28): update mean and covariance ----------------
        A = np.zeros((N, N))
        b = np.zeros(N)
        for r_idx in range(R):
            A += weights_raw[r_idx] * Ce_inv_list[r_idx]
            b += weights_raw[r_idx] * (Ce_inv_list[r_idx] @ y_norm_list[r_idx])

        mu_agg_new = np.linalg.solve(A, b)
        curve_history.append(mu_agg_new.copy())
        mu_agg = mu_agg_new

        # -- Step 3 (Eqn 29): Mahalanobis distances with updated mean ---
        residuals = [y_norm_list[r] - mu_agg for r in range(R)]

        energies = np.empty(R)
        for r_idx in range(R):
            e_r = residuals[r_idx]
            if L_list[r_idx] is not None:
                v = cho_solve(L_list[r_idx], e_r, check_finite=False)
                energies[r_idx] = float(e_r @ v)
            else:
                energies[r_idx] = float(e_r @ Ce_inv_list[r_idx] @ e_r)
            energies[r_idx] = max(energies[r_idx], 0.0)  # numerical safety

        energy_history.append(energies.copy())

        # -- Step 4 (Eqn 30): optimise ν (degrees of freedom) ----------
        if optimize_nu:
            nu = _optimize_nu(
                energies, N,
                nu_bounds=nu_bounds,
                verbose=(verbose and it == 0),
            )
        nu_history.append(nu)

        # -- Step 5 (Eqn 31): update precisions v_r --------------------
        new_weights_raw = (nu * N + N) / (nu * N + energies)

        # Optional outlier gating
        if outlier_energy_threshold is not None:
            new_weights_raw[energies > outlier_energy_threshold] = 0.0

        # Normalise for comparison
        w_sum = new_weights_raw.sum()
        if w_sum > 0:
            new_weights_norm = new_weights_raw / w_sum
        else:
            new_weights_norm = np.ones(R) / R
            new_weights_raw = np.ones(R)

        # Store history
        weight_history.append(new_weights_norm.copy())

        # -- Convergence check (Eqn 32: normalised weight change) ------
        if weights_raw.sum() > 0:
            prev_norm = weights_raw / weights_raw.sum()
        else:
            prev_norm = np.ones(R) / R
        max_weight_delta = float(np.max(np.abs(new_weights_norm - prev_norm)))

        weights_raw = new_weights_raw

        if verbose and (it == 0 or (it + 1) % 10 == 0
                        or max_weight_delta < convergence_tol):
            print(f"[Student-t] iter {it+1}: max_dw={max_weight_delta:.6e}, "
                  f"d_min={energies.min():.4g}, d_max={energies.max():.4g}, "
                  f"sigma_btw^2={sigma2_btw:.6g}, "
                  f"nu={nu:.6g} (nuN={nu*N:.4g})")

        if max_weight_delta < convergence_tol:
            converged = True
            break

    t_iter_elapsed = time.perf_counter() - t_iter_start

    if verbose:
        if converged:
            print(f"[Student-t] Converged in {iterations} iterations "
                  f"({t_iter_elapsed:.2f}s)")
        else:
            print(f"[Student-t] Did NOT converge after {iterations} iterations "
                  f"({t_iter_elapsed:.2f}s), max_dw={max_weight_delta:.2e}")
        print(f"[Student-t] Final sigma_btw^2 = {sigma2_btw:.6g}  "
              f"(initial = {sigma_btw_history[0]:.6g})")
        print(f"[Student-t] Final nu = {nu:.4f}  "
              f"(nuN = {nu * N:.1f})  "
              f"(initial = {nu_history[0]:.4f})")

    # ------------------------------------------------------------------
    # Step 4: Final covariance  Ĉ_agg = (Σ w_r C_{e,r}^{-1})^{-1}
    # ------------------------------------------------------------------
    A_final = np.zeros((N, N))
    for r_idx in range(R):
        A_final += weights_raw[r_idx] * Ce_inv_list[r_idx]
    Cagg_norm = np.linalg.solve(A_final, I_N)
    Cagg_norm = 0.5 * (Cagg_norm + Cagg_norm.T)   # enforce symmetry
    std_norm = np.sqrt(np.clip(np.diag(Cagg_norm), 0.0, None))

    # Normalised weights
    weights_norm = weights_raw / weights_raw.sum() if weights_raw.sum() > 0 else np.ones(R) / R

    # Predictive covariance:  C_pred = C_agg + σ²_btw I
    Cpred_norm = Cagg_norm + sigma2_btw * I_N
    Cpred_norm = 0.5 * (Cpred_norm + Cpred_norm.T)
    std_pred_norm = np.sqrt(np.clip(np.diag(Cpred_norm), 0.0, None))

    # ------------------------------------------------------------------
    # Step 5: Rescale normalised → observation scale
    # ------------------------------------------------------------------
    factors = np.array(
        [s.params.get("factor", 1.0) for s in y_scalings], dtype=float,
    )
    s_agg = float(np.dot(weights_norm, factors))
    s_agg2 = s_agg ** 2

    mu_real = s_agg * mu_agg
    Cagg_real = s_agg2 * Cagg_norm
    Cagg_real = 0.5 * (Cagg_real + Cagg_real.T)
    std_real = np.sqrt(np.clip(np.diag(Cagg_real), 0.0, None))

    Cpred_real = s_agg2 * Cpred_norm
    Cpred_real = 0.5 * (Cpred_real + Cpred_real.T)
    std_pred_real = np.sqrt(np.clip(np.diag(Cpred_real), 0.0, None))

    t_total = time.perf_counter() - t_start

    if verbose:
        print(f"[Student-t] Aggregated {R} curves on {N}-point grid")
        print(f"[Student-t] s_agg = {s_agg:.6g}")
        print(f"[Student-t] weight stats: "
              f"min={weights_norm.min():.6g}, "
              f"median={np.median(weights_norm):.6g}, "
              f"max={weights_norm.max():.6g}")
        print(f"[Student-t] energy stats: "
              f"min={energies.min():.4g}, "
              f"median={np.median(energies):.4g}, "
              f"max={energies.max():.4g}")
        print(f"[Student-t] mean(diag(C_agg))={np.mean(np.diag(Cagg_norm)):.6g}, "
              f"mean(diag(C_pred))={np.mean(np.diag(Cpred_norm)):.6g}")
        print(f"[Student-t] total elapsed: {t_total:.2f}s")

    return StudentTResult(
        y_mean_real=mu_real,
        y_mean_norm=mu_agg,
        y_std_real=std_real,
        y_std_norm=std_norm,
        y_cov_real=Cagg_real,
        y_cov_norm=Cagg_norm,
        y_std_predictive_real=std_pred_real,
        y_std_predictive_norm=std_pred_norm,
        y_cov_predictive_real=Cpred_real,
        y_cov_predictive_norm=Cpred_norm,
        sigma_btw_squared=sigma2_btw,
        nu=nu,
        weights=weights_norm,
        weights_raw=weights_raw,
        energies=energies,
        n_models=R,
        n_points=N,
        iterations=iterations,
        converged=converged,
        max_weight_delta=max_weight_delta,
        weight_history=weight_history,
        curve_history=curve_history,
        energy_history=energy_history,
        sigma_btw_history=sigma_btw_history,
        nu_history=nu_history,
        s_agg=s_agg,
    )
