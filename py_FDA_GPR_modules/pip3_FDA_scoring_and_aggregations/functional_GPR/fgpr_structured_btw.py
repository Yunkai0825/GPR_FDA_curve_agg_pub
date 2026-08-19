# functional_GPR/fgpr_structured_btw.py
"""
Structured Between-Curve Covariance for FGPR.

Replaces the scalar σ²_btw I with a structured operator:

    Ĉ_btw(ϑ) = σ²_w Î + σ²_s K̂_smooth(ℓ_b) + σ²_o |1⟩⟨1| + σ²_d |t⟩⟨t| + σ²_sc |m_ref⟩⟨m_ref|

where:
- σ²_w Î          : white noise (independent per grid point)
- σ²_s K̂_smooth   : smooth correlated deviations (RBF kernel)
- σ²_o |1⟩⟨1|     : constant offset mode
- σ²_d |t⟩⟨t|     : linear drift mode
- σ²_sc |m_ref⟩⟨m_ref| : scale/shape mode aligned to mean

Optimization uses profile likelihood with analytic gradients (L-BFGS-B)
in log-parameterized variance space.

References:
    Section 1.4 of _ref/GPR_derivation_Dirac_notation(R).md

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructuredBtwParams:
    r"""
    Optimised parameters for structured Ĉ_btw.

    All σ² values are in normalised-space units.
    """
    sigma2_w: float         # white-noise variance
    sigma2_s: float         # smooth-kernel amplitude
    ell_b: float            # smooth-kernel length-scale
    sigma2_o: float         # offset variance (|1⟩⟨1|)
    sigma2_d: float         # drift variance  (|t⟩⟨t|)
    sigma2_sc: float        # scale variance  (|m_ref⟩⟨m_ref|)
    nll: float              # final profile NLL
    n_outer_iter: int       # actual outer-loop iterations used
    converged: bool         # L-BFGS-B optimizer converged
    m_ref_hat: np.ndarray = None  # final normalised |m_ref⟩ vector
    weight_converged: bool = False   # did weights meet convergence?
    max_weight_delta: float = 0.0    # final max |Δw| across curves
    weight_history: list = None      # list of (R,) weight arrays per iter
    curve_history: list = None       # list of (N,) mu_agg arrays per iter
    raw_result: object = None  # scipy OptimizeResult


@dataclass
class StructuredBtwConfig:
    """
    User-facing configuration for the structured Ĉ_btw model.

    Set any ``enable_*`` to False to drop that component entirely
    (its σ² is clamped to 0).

    Use ``StructuredBtwConfig.white_noise_only()`` for a clean
    white-noise-only configuration (single parameter σ²_w).
    """
    enable_white: bool = True
    enable_smooth: bool = False
    enable_offset: bool = False
    enable_drift: bool = False
    enable_scale: bool = False
    # Smooth-kernel type: "rbf" (Matérn-∞) or "matern32" / "matern52"
    kernel_type: str = "rbf"
    # Outer-loop convergence settings
    max_outer_iter: int = 100    # max weight-convergence iterations
    weight_atol: float = 1e-4    # converge when max|Δw| < this
    # L-BFGS-B settings
    maxiter: int = 200
    ftol: float = 1e-8
    gtol: float = 1e-6
    # Numerical jitter
    jitter: float = 1e-10

    @classmethod
    def white_noise_only(cls, **kwargs) -> "StructuredBtwConfig":
        r"""Factory for the white-noise-only model: Ĉ_btw = σ²_w Î_J.

        Uses the full operator profile likelihood as the optimisation
        target (not the diagonal NLL heuristic).  Single parameter a_w
        = log σ²_w, optimised via L-BFGS-B with analytic gradient.
        """
        defaults = dict(
            enable_white=True,
            enable_smooth=False,
            enable_offset=False,
            enable_drift=False,
            enable_scale=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @property
    def needs_outer_iteration(self) -> bool:
        """Whether the outer weight-convergence loop is needed.

        The outer loop is only required when a component depends on
        the current aggregate (i.e. the |m_ref⟩⟨m_ref| scale mode,
        which uses |m_agg⟩/‖m_agg‖ as its basis vector).
        All other components are fixed w.r.t. the data and do not
        require re-optimisation after updating m_agg.
        """
        return self.enable_scale


# ====================================================================== #
#  Basis vectors in H_J                                                    #
# ====================================================================== #

def _build_ones_vec(N: int) -> np.ndarray:
    r"""Normalised constant vector |1⟩ = (1/√N) Σ_i |J_i⟩."""
    v = np.ones(N) / np.sqrt(N)
    return v


def _build_drift_vec(t_grid: np.ndarray) -> np.ndarray:
    r"""Normalised centred drift vector |t⟩ = (t - t̄)/‖t - t̄‖."""
    t_centred = t_grid - t_grid.mean()
    norm = np.linalg.norm(t_centred)
    if norm < 1e-30:
        return np.zeros_like(t_centred)
    return t_centred / norm


def _build_mref_vec(m_ref: np.ndarray) -> np.ndarray:
    r"""Normalised reference-mean vector |m_ref⟩ = m / ‖m‖."""
    norm = np.linalg.norm(m_ref)
    if norm < 1e-30:
        return np.zeros_like(m_ref)
    return m_ref / norm


# ====================================================================== #
#  Smooth kernel matrix                                                    #
# ====================================================================== #

def _rbf_kernel(t_grid: np.ndarray, ell: float) -> np.ndarray:
    """RBF kernel K_ij = exp(-|t_i - t_j|² / (2 ℓ²))."""
    sq_dists = (t_grid[:, None] - t_grid[None, :]) ** 2
    return np.exp(-sq_dists / (2.0 * ell ** 2))


def _rbf_kernel_dell(t_grid: np.ndarray, ell: float,
                     K: Optional[np.ndarray] = None) -> np.ndarray:
    """Derivative of RBF kernel w.r.t. ℓ:  dK/dℓ = K ⊙ (Δ² / ℓ³)."""
    sq_dists = (t_grid[:, None] - t_grid[None, :]) ** 2
    if K is None:
        K = np.exp(-sq_dists / (2.0 * ell ** 2))
    return K * sq_dists / (ell ** 3)


# ====================================================================== #
#  Build structured C_btw                                                  #
# ====================================================================== #

def build_C_btw(
    log_params: np.ndarray,
    t_grid: np.ndarray,
    m_ref_hat: np.ndarray,
    v_ones: np.ndarray,
    v_drift: np.ndarray,
    cfg: StructuredBtwConfig,
) -> np.ndarray:
    """
    Assemble Ĉ_btw(ϑ) from log-parameters.

    Parameters
    ----------
    log_params : (P,) array
        Log-parameterised hyperparameters [a_w, a_s, log_ell_b, a_o, a_d, a_sc]
        (only active components are present; order follows ``_param_indices``).
    t_grid : (N,) array
        Common x-grid in transformed space.
    m_ref_hat : (N,) array
        Current normalised reference-mean vector |m_ref⟩.
    v_ones : (N,) array
        Normalised constant vector |1⟩.
    v_drift : (N,) array
        Normalised drift vector |t⟩.
    cfg : StructuredBtwConfig
        Which components are active.

    Returns
    -------
    C_btw : (N, N) array
    """
    N = len(t_grid)
    idx = _param_indices(cfg)
    C = np.zeros((N, N))

    if cfg.enable_white:
        sigma2_w = np.exp(log_params[idx['a_w']])
        C += sigma2_w * np.eye(N)

    if cfg.enable_smooth:
        sigma2_s = np.exp(log_params[idx['a_s']])
        ell_b = np.exp(log_params[idx['log_ell']])
        K = _rbf_kernel(t_grid, ell_b)
        C += sigma2_s * K

    if cfg.enable_offset:
        sigma2_o = np.exp(log_params[idx['a_o']])
        C += sigma2_o * np.outer(v_ones, v_ones)

    if cfg.enable_drift:
        sigma2_d = np.exp(log_params[idx['a_d']])
        C += sigma2_d * np.outer(v_drift, v_drift)

    if cfg.enable_scale:
        sigma2_sc = np.exp(log_params[idx['a_sc']])
        C += sigma2_sc * np.outer(m_ref_hat, m_ref_hat)

    return C


# ====================================================================== #
#  Parameter index map                                                     #
# ====================================================================== #

def _param_indices(cfg: StructuredBtwConfig) -> Dict[str, int]:
    """Map component names → positions in log_params vector."""
    idx: Dict[str, int] = {}
    pos = 0
    if cfg.enable_white:
        idx['a_w'] = pos; pos += 1
    if cfg.enable_smooth:
        idx['a_s'] = pos; pos += 1
        idx['log_ell'] = pos; pos += 1
    if cfg.enable_offset:
        idx['a_o'] = pos; pos += 1
    if cfg.enable_drift:
        idx['a_d'] = pos; pos += 1
    if cfg.enable_scale:
        idx['a_sc'] = pos; pos += 1
    return idx


def _n_params(cfg: StructuredBtwConfig) -> int:
    """Number of free parameters."""
    n = 0
    if cfg.enable_white:  n += 1
    if cfg.enable_smooth: n += 2  # amplitude + length-scale
    if cfg.enable_offset: n += 1
    if cfg.enable_drift:  n += 1
    if cfg.enable_scale:  n += 1
    return n


# ====================================================================== #
#  Profile NLL + analytic gradient                                         #
# ====================================================================== #

def _profile_nll_and_grad(
    log_params: np.ndarray,
    m_list: List[np.ndarray],
    Cpost_list: List[np.ndarray],
    t_grid: np.ndarray,
    m_ref_hat: np.ndarray,
    v_ones: np.ndarray,
    v_drift: np.ndarray,
    cfg: StructuredBtwConfig,
) -> Tuple[float, np.ndarray]:
    """
    Compute the profile negative log-likelihood and its gradient
    with respect to log-params.

    For fixed ϑ_btw, m_agg is profiled out in closed form:
        Ĉ_agg⁻¹ = Σ_r Ĉ_{e,r}⁻¹
        |m_agg⟩  = Ĉ_agg Σ_r Ĉ_{e,r}⁻¹ |m_{post,r}⟩

    Then NLL = ½ Σ_r [ ⟨ε_r| Ĉ_{e,r}⁻¹ |ε_r⟩ + log det Ĉ_{e,r} ]

    Gradient (envelope theorem — m_agg treated as fixed):
        ∂L/∂a_k = ½ Σ_r [ -α_r^T D_k α_r + tr(Ĉ_{e,r}⁻¹ D_k) ]
    where α_r = Ĉ_{e,r}⁻¹ ε_r  and  D_k = ∂Ĉ_btw/∂a_k.
    """
    R = len(m_list)
    N = m_list[0].size
    I_N = np.eye(N)
    jitter = cfg.jitter
    idx = _param_indices(cfg)
    P = len(log_params)

    # --- Build C_btw ---
    C_btw = build_C_btw(log_params, t_grid, m_ref_hat, v_ones, v_drift, cfg)

    # --- Per-curve: C_e,r = C_post,r + C_btw, factorise ---
    Ce_inv_list = []
    Ce_logdet_list = []
    for Cpost in Cpost_list:
        Ce = Cpost + C_btw + jitter * I_N
        Ce = 0.5 * (Ce + Ce.T)  # ensure symmetry
        try:
            cf = cho_factor(Ce, lower=True, check_finite=False)
            Ce_inv = cho_solve(cf, I_N, check_finite=False)
            logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
        except np.linalg.LinAlgError:
            # Fallback: eigendecomposition
            eigvals = np.linalg.eigvalsh(Ce)
            eigvals = np.clip(eigvals, jitter, None)
            logdet = float(np.sum(np.log(eigvals)))
            Ce_inv = np.linalg.pinv(Ce)
        Ce_inv_list.append(Ce_inv)
        Ce_logdet_list.append(logdet)

    # --- Profiled m_agg (closed form) ---
    A = np.zeros((N, N))       # Ĉ_agg⁻¹ = Σ_r Ĉ_{e,r}⁻¹
    b = np.zeros(N)            # Σ_r Ĉ_{e,r}⁻¹ m_r
    for Ce_inv, m in zip(Ce_inv_list, m_list):
        A += Ce_inv
        b += Ce_inv @ m

    try:
        cf_A = cho_factor(A, lower=True, check_finite=False)
        m_agg = cho_solve(cf_A, b, check_finite=False)
    except np.linalg.LinAlgError:
        m_agg = np.linalg.solve(A, b)

    # --- NLL ---
    nll = 0.0
    alpha_list = []  # α_r = Ĉ_{e,r}⁻¹ ε_r
    for Ce_inv, logdet, m in zip(Ce_inv_list, Ce_logdet_list, m_list):
        eps = m - m_agg
        alpha = Ce_inv @ eps
        alpha_list.append(alpha)
        nll += 0.5 * (eps @ alpha + logdet)
    nll += 0.5 * R * N * np.log(2.0 * np.pi)

    # --- Gradient computation ---
    # For each parameter a_k, D_k = ∂C_btw/∂a_k
    # ∂L/∂a_k = 0.5 Σ_r [ -α_r^T D_k α_r + tr(Ĉ_{e,r}⁻¹ D_k) ]
    grad = np.zeros(P)

    # Precompute the derivative matrices D_k
    # (and the Σ_r α_r α_r^T and Σ_r Ĉ_{e,r}⁻¹ for efficiency)
    sum_Ce_inv = np.zeros((N, N))
    sum_aaT = np.zeros((N, N))
    for Ce_inv, alpha in zip(Ce_inv_list, alpha_list):
        sum_Ce_inv += Ce_inv
        sum_aaT += np.outer(alpha, alpha)

    # Q = Σ_r (Ĉ_{e,r}⁻¹ - α_r α_r^T)
    # Then ∂L/∂a_k = 0.5 tr(Q · D_k)
    Q = sum_Ce_inv - sum_aaT

    # White noise: D_w = σ²_w · I  (since ∂C_btw/∂a_w = σ²_w I)
    if cfg.enable_white:
        sigma2_w = np.exp(log_params[idx['a_w']])
        # tr(Q · σ²_w I) = σ²_w tr(Q)
        grad[idx['a_w']] = 0.5 * sigma2_w * np.trace(Q)

    # Smooth kernel: D_s = σ²_s · K,  D_ell = σ²_s · ℓ_b · dK/dℓ_b
    if cfg.enable_smooth:
        sigma2_s = np.exp(log_params[idx['a_s']])
        ell_b = np.exp(log_params[idx['log_ell']])
        K = _rbf_kernel(t_grid, ell_b)
        dK_dell = _rbf_kernel_dell(t_grid, ell_b, K=K)

        # ∂L/∂a_s = 0.5 σ²_s tr(Q K)
        grad[idx['a_s']] = 0.5 * sigma2_s * np.sum(Q * K)

        # ∂L/∂log_ell = 0.5 σ²_s ℓ_b tr(Q dK/dℓ_b)
        #             = 0.5 σ²_s tr(Q · ℓ_b dK/dℓ)
        grad[idx['log_ell']] = 0.5 * sigma2_s * ell_b * np.sum(Q * dK_dell)

    # Offset: D_o = σ²_o · |1⟩⟨1|
    if cfg.enable_offset:
        sigma2_o = np.exp(log_params[idx['a_o']])
        # tr(Q · σ²_o |1⟩⟨1|) = σ²_o · 1^T Q 1 = σ²_o v^T Q v
        grad[idx['a_o']] = 0.5 * sigma2_o * (v_ones @ Q @ v_ones)

    # Drift: D_d = σ²_d · |t⟩⟨t|
    if cfg.enable_drift:
        sigma2_d = np.exp(log_params[idx['a_d']])
        grad[idx['a_d']] = 0.5 * sigma2_d * (v_drift @ Q @ v_drift)

    # Scale: D_sc = σ²_sc · |m_ref⟩⟨m_ref|
    if cfg.enable_scale:
        sigma2_sc = np.exp(log_params[idx['a_sc']])
        grad[idx['a_sc']] = 0.5 * sigma2_sc * (m_ref_hat @ Q @ m_ref_hat)

    return nll, grad


# ====================================================================== #
#  Full optimisation: fit_structured_btw                                   #
# ====================================================================== #

def fit_structured_btw(
    m_list: List[np.ndarray],
    Cpost_list: List[np.ndarray],
    t_grid: np.ndarray,
    *,
    cfg: Optional[StructuredBtwConfig] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StructuredBtwParams]:
    """
    Optimise structured Ĉ_btw and compute aggregated posterior.

    Implements the full algorithm from Section 7 of the derivation:

    1. Initialise ϑ_btw (small σ's, data-driven ℓ_b).
    2. For each outer iteration:
       a. Build Ĉ_btw(ϑ_btw).
       b. Profile out m_agg in closed form, evaluate NLL + gradient.
       c. Optimise ϑ_btw via L-BFGS-B.
       d. Update |m_ref⟩ ← |m_agg⟩ / ‖m_agg‖.

    Parameters
    ----------
    m_list : list of (N,) arrays
        GP posterior means per curve (normalised units).
    Cpost_list : list of (N, N) arrays
        GP posterior covariances per curve (normalised units).
    t_grid : (N,) array
        Common x-grid in transformed space.
    cfg : StructuredBtwConfig, optional
        Config for enabled components and optimiser settings.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    C_btw_opt : (N, N) array
        Optimised between-curve covariance.
    mu_agg : (N,) array
        Aggregated posterior mean (normalised units).
    Cagg : (N, N) array
        Aggregated posterior covariance (normalised units).
    params : StructuredBtwParams
        Optimised hyperparameters and diagnostics.
    """
    if cfg is None:
        cfg = StructuredBtwConfig()

    m_list = [np.asarray(m, float).ravel() for m in m_list]
    Cpost_list = [np.asarray(C, float) for C in Cpost_list]
    R = len(m_list)
    N = m_list[0].size
    I_N = np.eye(N)
    jitter = cfg.jitter

    if verbose:
        print(f"[FGPR-Structured] {R} curves, {N}-point grid")
        enabled = []
        if cfg.enable_white:  enabled.append("white")
        if cfg.enable_smooth: enabled.append("smooth")
        if cfg.enable_offset: enabled.append("offset")
        if cfg.enable_drift:  enabled.append("drift")
        if cfg.enable_scale:  enabled.append("scale")
        print(f"[FGPR-Structured] active components: {', '.join(enabled)}")

    # --- Basis vectors ---
    v_ones = _build_ones_vec(N)
    v_drift = _build_drift_vec(t_grid)

    # --- Initial m_ref from simple mean ---
    m_stack = np.vstack(m_list)  # (R, N)
    m_simple = m_stack.mean(axis=0)
    m_ref_hat = _build_mref_vec(m_simple)

    # Outer loop is only needed when a component depends on the current
    # aggregate (the |m_ref⟩⟨m_ref| scale mode).  For white-noise-only
    # or any config without scale, one optimisation pass suffices.
    max_outer = max(1, cfg.max_outer_iter) if cfg.needs_outer_iteration else 1

    # --- Data-driven initialisation ---
    spread = np.var(m_stack, axis=0).mean()        # between-curve variance
    within = np.mean([np.mean(np.diag(C)) for C in Cpost_list])
    t_range = float(t_grid.max() - t_grid.min()) if N > 1 else 1.0

    # Initial log-params
    init_vals: Dict[str, float] = {
        'a_w': np.log(max(spread * 0.1, 1e-12)),
        'a_s': np.log(max(spread * 0.5, 1e-12)),
        'log_ell': np.log(max(t_range * 0.25, 1e-6)),
        'a_o': np.log(max(spread * 0.2, 1e-12)),
        'a_d': np.log(max(spread * 0.1, 1e-12)),
        'a_sc': np.log(max(spread * 0.1, 1e-12)),
    }

    if verbose:
        print(f"[FGPR-Structured] data-driven init: "
              f"spread={spread:.4g}, within={within:.4g}, t_range={t_range:.4g}")
        if cfg.needs_outer_iteration:
            print(f"[FGPR-Structured] weight convergence: "
                  f"max_iter={max_outer}, atol={cfg.weight_atol:.2e}")
        else:
            print(f"[FGPR-Structured] no outer iteration needed "
                  f"(no m_ref-dependent components)")

    best_nll = np.inf
    best_log_params = None
    best_m_agg = m_simple.copy()
    best_Cagg = None

    # --- Weight convergence tracking ---
    prev_weights = np.ones(R) / R   # uniform initialisation
    weight_history: list = []
    curve_history: list = []
    weight_converged = False
    max_weight_delta = 0.0

    outer_i = 0
    while True:
        # Build initial log_params vector
        idx = _param_indices(cfg)
        P = _n_params(cfg)
        x0 = np.zeros(P)
        for key, pos in idx.items():
            x0[pos] = init_vals[key]

        # Bounds: log-variances in [-30, 20], log-ell in [-10, log(t_range*5)]
        bounds = []
        for key, pos in sorted(idx.items(), key=lambda kv: kv[1]):
            if key == 'log_ell':
                bounds.append((np.log(1e-6), np.log(max(t_range * 5.0, 1.0))))
            else:
                bounds.append((-30.0, 20.0))

        if verbose and outer_i == 0:
            print(f"[FGPR-Structured] optimising {P} params "
                  f"(bounds: {[(f'{lo:.1f}',f'{hi:.1f}') for lo,hi in bounds]})")

        # --- Run L-BFGS-B ---
        result = minimize(
            fun=_profile_nll_and_grad,
            x0=x0,
            args=(m_list, Cpost_list, t_grid, m_ref_hat, v_ones, v_drift, cfg),
            method='L-BFGS-B',
            jac=True,  # function returns (nll, grad)
            bounds=bounds,
            options=dict(
                maxiter=cfg.maxiter,
                ftol=cfg.ftol,
                gtol=cfg.gtol,
                disp=False,
            ),
        )

        opt_log_params = result.x
        opt_nll = result.fun

        # --- Aggregation at optimal params ---
        C_btw_opt = build_C_btw(opt_log_params, t_grid, m_ref_hat,
                                v_ones, v_drift, cfg)

        A = np.zeros((N, N))
        b_vec = np.zeros(N)
        for m, Cpost in zip(m_list, Cpost_list):
            Ce = Cpost + C_btw_opt + jitter * I_N
            Ce = 0.5 * (Ce + Ce.T)
            try:
                cf = cho_factor(Ce, lower=True, check_finite=False)
                Ce_inv = cho_solve(cf, I_N, check_finite=False)
            except np.linalg.LinAlgError:
                Ce_inv = np.linalg.pinv(Ce)
            A += Ce_inv
            b_vec += Ce_inv @ m

        try:
            cf_A = cho_factor(A, lower=True, check_finite=False)
            Cagg = cho_solve(cf_A, I_N, check_finite=False)
        except np.linalg.LinAlgError:
            Cagg = np.linalg.pinv(A)
        mu_agg = Cagg @ b_vec

        # --- Compute weights at this iteration ---
        cur_weights = np.zeros(R)
        for r_idx in range(R):
            Ce_r = Cpost_list[r_idx] + C_btw_opt + jitter * I_N
            Ce_r = 0.5 * (Ce_r + Ce_r.T)
            try:
                Ce_r_inv = np.linalg.inv(Ce_r)
            except np.linalg.LinAlgError:
                Ce_r_inv = np.linalg.pinv(Ce_r)
            w_tilde_r = Cagg @ Ce_r_inv
            cur_weights[r_idx] = float(w_tilde_r.sum())
        cur_weights = cur_weights / cur_weights.sum()

        # --- Weight convergence check ---
        max_weight_delta = float(np.max(np.abs(cur_weights - prev_weights)))

        # Record history
        weight_history.append(cur_weights.copy())
        curve_history.append(mu_agg.copy())

        # --- Per-iteration logging ---
        if verbose:
            w_str = ', '.join(f'{w:.6f}' for w in cur_weights)
            print(f"[FGPR-Structured] iter {outer_i+1}/{max_outer}: "
                  f"NLL={opt_nll:.6g}, max|dw|={max_weight_delta:.2e}, "
                  f"L-BFGS-B={result.success}, "
                  f"weights=[{w_str}]")

        if opt_nll < best_nll:
            best_nll = opt_nll
            best_log_params = opt_log_params.copy()
            best_m_agg = mu_agg.copy()
            best_Cagg = Cagg.copy()

        # --- Check convergence ---
        # For configs without outer iteration (e.g. white-noise-only),
        # a single pass is exact — declare converged immediately.
        # For configs requiring iteration (scale mode), require at least
        # two passes to measure weight stability against a real baseline.
        if not cfg.needs_outer_iteration:
            weight_converged = True
            if verbose:
                print(f"[FGPR-Structured] single-pass optimisation complete "
                      f"(no outer iteration needed)")
            break

        if outer_i > 0 and max_weight_delta < cfg.weight_atol:
            weight_converged = True
            if verbose:
                print(f"[FGPR-Structured] weights converged after "
                      f"{outer_i+1} iteration(s) (max|dw|={max_weight_delta:.2e} "
                      f"< atol={cfg.weight_atol:.2e})")
            break

        outer_i += 1
        if outer_i >= max_outer:
            if verbose:
                print(f"[FGPR-Structured] max iterations ({max_outer}) reached "
                      f"(max|dw|={max_weight_delta:.2e})")
            break

        # --- Update for next iteration ---
        prev_weights = cur_weights.copy()
        if cfg.needs_outer_iteration:
            m_ref_hat = _build_mref_vec(mu_agg)
            if verbose:
                print(f"[FGPR-Structured] updated |m_ref> "
                      f"(||m_agg||={np.linalg.norm(mu_agg):.4g})")

        # Warm-start next iteration from current optimum
        for key, pos in idx.items():
            init_vals[key] = float(opt_log_params[pos])

    n_outer_done = outer_i + 1

    # --- Extract final parameters ---
    idx = _param_indices(cfg)
    lp = best_log_params

    sigma2_w = float(np.exp(lp[idx['a_w']])) if cfg.enable_white else 0.0
    sigma2_s = float(np.exp(lp[idx['a_s']])) if cfg.enable_smooth else 0.0
    ell_b = float(np.exp(lp[idx['log_ell']])) if cfg.enable_smooth else 0.0
    sigma2_o = float(np.exp(lp[idx['a_o']])) if cfg.enable_offset else 0.0
    sigma2_d = float(np.exp(lp[idx['a_d']])) if cfg.enable_drift else 0.0
    sigma2_sc = float(np.exp(lp[idx['a_sc']])) if cfg.enable_scale else 0.0

    params = StructuredBtwParams(
        sigma2_w=sigma2_w,
        sigma2_s=sigma2_s,
        ell_b=ell_b,
        sigma2_o=sigma2_o,
        sigma2_d=sigma2_d,
        sigma2_sc=sigma2_sc,
        m_ref_hat=m_ref_hat.copy(),
        nll=best_nll,
        n_outer_iter=n_outer_done,
        converged=result.success,
        weight_converged=weight_converged,
        max_weight_delta=max_weight_delta,
        weight_history=weight_history,
        curve_history=curve_history,
        raw_result=result,
    )

    if verbose:
        print(f"[FGPR-Structured] === Final parameters ===")
        print(f"  s2_w  (white)  = {sigma2_w:.6g}")
        print(f"  s2_s  (smooth) = {sigma2_s:.6g},  ell_b = {ell_b:.6g}")
        print(f"  s2_o  (offset) = {sigma2_o:.6g}")
        print(f"  s2_d  (drift)  = {sigma2_d:.6g}")
        print(f"  s2_sc (scale)  = {sigma2_sc:.6g}")
        print(f"  NLL = {best_nll:.6g}")
        print(f"  weight_converged = {weight_converged} "
              f"({n_outer_done} iters, max|dw|={max_weight_delta:.2e})")

    # Rebuild final C_btw at best params
    C_btw_final = build_C_btw(best_log_params, t_grid, m_ref_hat,
                              v_ones, v_drift, cfg)

    return C_btw_final, best_m_agg, best_Cagg, params


# ====================================================================== #
#  Compute the "effective σ²_btw" scalar for backward compatibility       #
# ====================================================================== #

def effective_sigma2_btw(C_btw: np.ndarray) -> float:
    """
    Mean diagonal of the structured Ĉ_btw, serving as a scalar
    summary compatible with the original σ²_btw I model.
    """
    return float(np.mean(np.diag(C_btw)))
