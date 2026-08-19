"""
Functional Gaussian Process Regression (FGPR) helpers for curve aggregation.

Implements the FGPR aggregation method from the derivation in
_ref/GPR_derivation_Dirac_notation(R).md, Section 1.2:

    m_agg = (Σ_r C_e,r^{-1})^{-1} Σ_r C_e,r^{-1} m_post,r
    C_agg = (Σ_r C_e,r^{-1})^{-1}

where C_e,r = C_post,r + σ_btw² I  and σ_btw² is optimized via
profile negative log-likelihood (1D bounded scalar optimization).

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from scipy.optimize import minimize_scalar
from scipy.linalg import cho_factor, cho_solve

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FGPRResult:
    """
    Result from Functional GPR aggregation.

    Attributes
    ----------
    y_mean_real : np.ndarray
        Aggregated posterior mean in real (original) scale.
    y_mean_norm : np.ndarray
        Aggregated posterior mean in normalized scale.
    y_std_real : np.ndarray
        Aggregated posterior std (diagonal of C_agg) in real scale.
    y_std_norm : np.ndarray
        Aggregated posterior std in normalized scale.
    y_cov_real : np.ndarray
        Full posterior covariance in real scale, shape (N, N).
    y_cov_norm : np.ndarray
        Full posterior covariance in normalized scale, shape (N, N).
    y_std_predictive_real : np.ndarray
        Predictive std in real scale:
        sqrt(diag(s_agg² · (C̃_agg + σ²_btw I))).
    y_std_predictive_norm : np.ndarray
        Predictive std in normalized scale: sqrt(diag(C̃_agg + σ²_btw I)).
    y_cov_predictive_real : np.ndarray
        Full predictive covariance in real scale, shape (N, N).
    y_cov_predictive_norm : np.ndarray
        Full predictive covariance in normalized scale, shape (N, N).
    sigma_btw_squared : float
        Optimized between-curve variance (σ_btw²) in normalized units.
    nll_optimized : float
        Negative log-likelihood at the optimized σ_btw².
    weights : np.ndarray
        Precision-derived weights per curve (normalized to sum to 1).
    n_models : int
        Number of curves aggregated.
    n_points : int
        Number of grid points.
    optimizer_result : object
        Raw result from scipy.optimize.minimize_scalar.
    """
    y_mean_real: np.ndarray
    y_mean_norm: np.ndarray
    y_std_real: np.ndarray
    y_std_norm: np.ndarray
    y_cov_real: np.ndarray
    y_cov_norm: np.ndarray
    # Predictive covariance: C_pred = C_agg + sigma_btw^2 * I
    # Describes expected spread of a new curve around the aggregated mean
    y_std_predictive_real: np.ndarray
    y_std_predictive_norm: np.ndarray
    y_cov_predictive_real: np.ndarray
    y_cov_predictive_norm: np.ndarray
    sigma_btw_squared: float
    nll_optimized: float
    weights: np.ndarray
    n_models: int
    n_points: int
    optimizer_result: object
    # Per-curve operators (4.23a-b) and rescaling factor
    curve_operators: Optional[List] = None
    s_agg: float = 1.0              # Weighted-average scale factor
    sum_s_r_squared: float = 0.0    # Deprecated: kept for back-compat
    # Iterative weight convergence history (structured C_btw only)
    weight_history: Optional[List[np.ndarray]] = None
    curve_history: Optional[List[np.ndarray]] = None
    weight_converged: bool = False
    max_weight_delta: float = 0.0
    n_weight_iters: int = 0


# ---------------------------------------------------------------------------
# Per-curve operator container
# ---------------------------------------------------------------------------

@dataclass
class FGPRCurveOperators:
    """
    Per-curve operators from eqs (4.23a-b) of the derivation.

    All tilde quantities live in the normalised (aggregation) space.
    s_r is the per-curve scale factor satisfying:
        f̃_r = (1/s_r) f_r   (normalisation, eq 4.1)
        f_r  = s_r f̃_r       (inverse)
    """
    C_e_tilde_r: np.ndarray       # (N,N) Eq (4.23a): C̃_{e,r} = C̃_{post,r} + σ²_btw I
    C_e_tilde_inv_r: np.ndarray   # (N,N) (C̃_{e,r})⁻¹
    w_tilde_r: np.ndarray         # (N,N) Eq (4.23b): w̃_r = C̃_agg · (C̃_{e,r})⁻¹
    s_r: float                    # Scale factor for curve r
    curve_index: int              # Index r


# ---------------------------------------------------------------------------
# Internals
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
    """Weighted aggregation of ScalingInfo objects (mirrors operator_fusion version)."""
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
        return ScalingInfo.standardize(float(np.dot(w, means)), float(np.dot(w, stds)))

    if method == "minmax":
        mins = np.array([s.params.get("min_val", 0.0) for s in y_scalings])
        maxs = np.array([s.params.get("max_val", 1.0) for s in y_scalings])
        feature_range = y_scalings[0].params.get("feature_range", (0, 1))
        return ScalingInfo.minmax(float(np.dot(w, mins)), float(np.dot(w, maxs)), feature_range)

    if method.startswith("log"):
        shift = y_scalings[0].params.get("shift", 1e-9)
        base = y_scalings[0].params.get("base", "log10")
        return ScalingInfo.log_transform(shift=shift, base=base)

    if method == "identity":
        return ScalingInfo.identity()

    factors = [s.params.get("factor", 1.0) for s in y_scalings]
    agg_factor = float(np.dot(w, factors))
    return ScalingInfo.divide_by_factor(agg_factor, method_name=method)


# ---------------------------------------------------------------------------
# Per-curve operators: eqs (4.23a-b), (4.14a-b)
# ---------------------------------------------------------------------------

def build_per_curve_operators(
    Cagg_tilde: np.ndarray,
    Cpost_tilde_list: List[np.ndarray],
    sigma2_btw: float,
    factors: np.ndarray,
    epsilon: float = 1e-12,
) -> List[FGPRCurveOperators]:
    """
    Build per-curve operators following eqs (4.23a-b) of the derivation.

    For each curve r, computes:
        (4.23a)  C̃_{e,r} = C̃_{post,r} + σ²_btw I
        (4.23b)  w̃_r = C̃_agg · (C̃_{e,r})⁻¹

    Parameters
    ----------
    Cagg_tilde : (N, N) array
        Aggregated posterior covariance in normalised space, C̃_agg.
    Cpost_tilde_list : list of (N, N) arrays
        Normalised posterior covariance per curve, C̃_{post,r}.
    sigma2_btw : float
        Optimised between-curve variance σ²_btw.
    factors : (R,) array
        Scale factors s_r per curve (observation = s_r × normalised).
    epsilon : float
        Numerical jitter.

    Returns
    -------
    list of FGPRCurveOperators
    """
    N = Cagg_tilde.shape[0]
    I_N = np.eye(N)
    ops_list: List[FGPRCurveOperators] = []

    for r, Cpost_r in enumerate(Cpost_tilde_list):
        # Eq (4.23a): C̃_{e,r} = C̃_{post,r} + σ²_btw I
        C_e_tilde_r = Cpost_r + (sigma2_btw + epsilon) * I_N

        # (C̃_{e,r})⁻¹
        try:
            C_e_tilde_inv_r = np.linalg.inv(C_e_tilde_r)
        except np.linalg.LinAlgError:
            C_e_tilde_inv_r = np.linalg.pinv(C_e_tilde_r)

        # Eq (4.23b) second form: w̃_r = C̃_agg · (C̃_{e,r})⁻¹
        w_tilde_r = Cagg_tilde @ C_e_tilde_inv_r

        ops_list.append(FGPRCurveOperators(
            C_e_tilde_r=C_e_tilde_r,
            C_e_tilde_inv_r=C_e_tilde_inv_r,
            w_tilde_r=w_tilde_r,
            s_r=float(factors[r]),
            curve_index=r,
        ))

    return ops_list


def compute_scalar_weights(ops_list: List[FGPRCurveOperators]) -> np.ndarray:
    """
    Scalar weights from eq (4.14a-b):
        w_r = Σ_i Σ_j w̃_r(i,j) = 1ᵀ w̃_r 1
        Σ_r w_r = N  (un-normalised)

    Returns weights normalised to sum to 1.
    """
    raw = np.array([float(ops.w_tilde_r.sum()) for ops in ops_list])
    return raw / raw.sum()


# ---------------------------------------------------------------------------
# R̂ projection: eqs (4.23c-e)
# ---------------------------------------------------------------------------

def project_R_mean(
    ops_list: List[FGPRCurveOperators],
    m_tilde_list: List[np.ndarray],
) -> np.ndarray:
    """
    Eq (4.23c):
        ⟨J_i|R̂|m_agg⟩ = Σ_r s_r Σ_j w̃_r(i,j) m̃_{post,r}(j)

    Project aggregated mean from normalised → observation scale.
    Each curve r contributes s_r · (w̃_r @ m̃_r).
    """
    N = m_tilde_list[0].shape[0]
    mu_real = np.zeros(N)
    for ops, m_norm in zip(ops_list, m_tilde_list):
        mu_real += ops.s_r * (ops.w_tilde_r @ m_norm)
    return mu_real


def project_R_cov_agg(
    ops_list: List[FGPRCurveOperators],
    Cagg_tilde: np.ndarray,
) -> np.ndarray:
    """
    .. deprecated::
        This function uses Σ_r s_r² which grows with R, inflating the
        observation-scale covariance.  Use :func:`rescale_fgpr_to_observation`
        instead, which applies s_agg² = (Σ_r w_r s_r)².

    Original Eq (4.23d) R̂-projection (incorrect for obs-scale CI):
        C^rescaled = (Σ_r s_r²) · C̃_agg
    """
    import warnings
    warnings.warn(
        "project_R_cov_agg uses sum_r s_r^2 which scales with R, giving "
        "inflated obs-scale covariance.  Use rescale_fgpr_to_observation().",
        DeprecationWarning, stacklevel=2,
    )
    sum_s2 = sum(ops.s_r ** 2 for ops in ops_list)
    return sum_s2 * Cagg_tilde


def project_R_cov_meas(
    ops_list: List[FGPRCurveOperators],
    Cagg_tilde: np.ndarray,
    sigma2_btw: float,
) -> np.ndarray:
    """
    .. deprecated::
        Same Σ_r s_r² bug as :func:`project_R_cov_agg`.
        Use :func:`rescale_fgpr_to_observation` instead.

    Original Eq (4.23e) R̂-projection (incorrect for obs-scale CI):
        C_meas = (Σ_r s_r²) · (C̃_agg + σ̃²_btw I)
    """
    import warnings
    warnings.warn(
        "project_R_cov_meas uses sum_r s_r^2 which scales with R, giving "
        "inflated obs-scale covariance.  Use rescale_fgpr_to_observation().",
        DeprecationWarning, stacklevel=2,
    )
    N = Cagg_tilde.shape[0]
    sum_s2 = sum(ops.s_r ** 2 for ops in ops_list)
    return sum_s2 * (Cagg_tilde + sigma2_btw * np.eye(N))


# ---------------------------------------------------------------------------
# Rescaling: normalised → observation scale  (corrected)
# ---------------------------------------------------------------------------

def rescale_fgpr_to_observation(
    mu_norm: np.ndarray,
    Cagg_norm: np.ndarray,
    sigma2_btw: float,
    weights: np.ndarray,
    factors: np.ndarray,
    curve_ops: List[FGPRCurveOperators],
    y_norm_list: List[np.ndarray],
    y_scalings: Optional[List[ScalingInfo]] = None,
) -> dict:
    r"""
    Convert FGPR results from normalised space to observation scale.

    The posterior in normalised space is f̃ | data ~ N(m̃_agg, C̃_agg).
    Converting to observation scale via f_real = s_agg · f̃  gives:

        μ_real    = s_agg · m̃_agg
        C_real    = s_agg² · C̃_agg
        C_pred    = s_agg² · (C̃_agg + σ²_btw I)

    Both C̃_agg and σ²_btw I are covariance components in the same
    normalised space and must transform with the same scalar s_agg².

    The previous implementation used  W C̃_agg Wᵀ  (with
    W = Σ_r s_r w̃_r) for the posterior but s_agg² for σ²_btw.
    This is inconsistent when scale factors s_r are heterogeneous:
    W Wᵀ ≠ s_agg² I, causing the posterior CI to inflate while the
    between-curve term did not (or vice-versa).

    For homogeneous s_r = s: s_agg = s, W = sI ⟹
        C_pred = s²(C̃_agg + σ²I).  ✓

    The W matrix is still computed and returned for per-curve
    diagnostics (e.g. per-curve operator products) but is no longer
    used for the bulk covariance rescaling.

    Parameters
    ----------
    mu_norm : (N,) array
        Aggregated posterior mean in normalised space (m̃_agg).
    Cagg_norm : (N, N) array
        Aggregated posterior covariance in normalised space (C̃_agg).
    sigma2_btw : float
        Optimised between-curve variance σ²_btw.
    weights : (R,) array
        Precision-derived scalar weights per curve (sum to 1).
    factors : (R,) array
        Per-curve scale factors s_r.
    curve_ops : list of FGPRCurveOperators
        Per-curve operators (contain w̃_r matrices).
    y_norm_list : list of (N,) arrays
        Per-curve posterior means in normalised units.
    y_scalings : list of ScalingInfo, optional
        Per-curve Y-axis scaling info.  When provided, the aggregated
        ScalingInfo is built via ``_compute_aggregated_scaling`` for
        reporting / downstream inverse transforms.

    Returns
    -------
    dict with keys:
        mu_real, std_real, Cagg_real,
        std_pred_real, Cpred_real,
        W, s_agg, scaling_agg, sum_s_r_squared
    """
    N = mu_norm.shape[0]

    # --- Weighted denormalisation matrix (kept for per-curve diagnostics) --
    W = np.zeros((N, N))
    for ops in curve_ops:
        W += ops.s_r * ops.w_tilde_r

    # --- Scalar scale factor ---
    s_agg = float(np.dot(weights, factors))
    sum_s2 = float(np.sum(factors ** 2))   # kept for backward compat
    s_agg2 = s_agg ** 2

    # --- Mean: f_real = s_agg · m̃_agg ---
    mu_real = s_agg * mu_norm

    # --- Posterior covariance: s_agg² · C̃_agg --------------------------
    #   C̃_agg and σ²_btw I both live in normalised space and must
    #   transform with the same scalar s_agg² to observation scale.
    Cagg_real = s_agg2 * Cagg_norm
    Cagg_real = 0.5 * (Cagg_real + Cagg_real.T)  # enforce symmetry
    std_real = np.sqrt(np.clip(np.diag(Cagg_real), 0.0, None))

    # --- Predictive covariance: s_agg² · (C̃_agg + σ²_btw I) -----------
    Cpred_real = s_agg2 * (Cagg_norm + sigma2_btw * np.eye(N))
    Cpred_real = 0.5 * (Cpred_real + Cpred_real.T)  # enforce symmetry
    std_pred_real = np.sqrt(np.clip(np.diag(Cpred_real), 0.0, None))

    # --- Aggregated ScalingInfo (for downstream / reporting) ---
    if y_scalings is not None:
        scaling_agg = _compute_aggregated_scaling(y_scalings, weights)
    else:
        scaling_agg = ScalingInfo.divide_by_factor(s_agg)

    return dict(
        mu_real=mu_real,
        std_real=std_real,
        Cagg_real=Cagg_real,
        std_pred_real=std_pred_real,
        Cpred_real=Cpred_real,
        W=W,
        s_agg=s_agg,
        scaling_agg=scaling_agg,
        sum_s_r_squared=sum_s2,
    )


# ---------------------------------------------------------------------------
# Core FGPR: σ_btw² optimization + aggregation
# ---------------------------------------------------------------------------

def fit_sigma_btw(
    m_list: List[np.ndarray],
    Cpost_list: List[np.ndarray],
    *,
    bounds: Tuple[float, Optional[float]] = (0.0, None),
    jitter: float = 1e-12,
    verbose: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, object]:
    """
    Profile-likelihood 1-D optimization of σ_btw² (between-curve variance)
    using a diagonal (marginal-variance) NLL for robust σ_btw² estimation,
    followed by full-covariance aggregation.

    Implements the reference formula (Section 1.2 of the derivation):

        C_e,r = C_post,r + σ_btw² I
        m_agg = (Σ C_e,r⁻¹)⁻¹ · Σ C_e,r⁻¹ m_r
        C_agg = (Σ C_e,r⁻¹)⁻¹

    **σ_btw² estimation** uses the diagonal (marginal) NLL:

        NLL_diag = ½ Σ_r Σ_j [ (m_rj - μ_j)² / (v_rj + σ²)
                                + log(v_rj + σ²) ]

    where v_rj = diag(C_post,r)_j and μ_j is profiled out per grid point.
    This avoids the high-dimensional log-determinant curse that biases the
    full N×N NLL toward σ_btw² = 0 when N >> R.  It is equivalent to
    standard random-effects meta-analysis pooled across grid points.

    Parameters
    ----------
    m_list : list of (N,) arrays
        GP posterior means for each curve (normalized units).
    Cpost_list : list of (N, N) arrays
        GP posterior covariances for each curve (normalized units).
    bounds : (lo, hi)
        Search bounds for σ_btw².  If hi is None a data-driven bound is used.
    jitter : float
        Nugget for numerical stability.
    verbose : bool
        Print optimisation diagnostics.

    Returns
    -------
    sigma2_star : float
        Optimized σ_btw².
    mu : np.ndarray, shape (N,)
        Aggregated posterior mean (normalized units).
    Cagg : np.ndarray, shape (N, N)
        Aggregated posterior covariance (normalized units).
    res : OptimizeResult
        Scipy optimiser result.
    """
    m_list = [np.asarray(m, float).ravel() for m in m_list]
    Cpost_list = [np.asarray(C, float) for C in Cpost_list]
    R = len(m_list)
    N = m_list[0].size
    I = np.eye(N)

    # --- automatic upper bound -------------------------------------------
    m_stack = np.vstack(m_list)  # (R, N)
    lo = bounds[0]
    hi = bounds[1]
    if hi is None:
        spread = float(np.median(np.var(m_stack, axis=0)))
        within = float(np.median([np.median(np.diag(C)) for C in Cpost_list]))
        hi = max(1e-12, 50.0 * (spread + within))
    bounds_final = (lo, hi)

    # --- diagonal NLL (profiled over μ_j per grid point) -----------------
    # Stack marginal variances: V[r, j] = diag(C_post,r)_j
    V = np.vstack([np.diag(C) for C in Cpost_list])  # (R, N)

    def nll_diag(sigma2: float) -> float:
        sigma2 = max(sigma2, 0.0)
        S = V + sigma2 + jitter           # (R, N)  effective variance
        prec = 1.0 / S                    # (R, N)
        # Profiled mean: μ_j = Σ_r prec_rj m_rj / Σ_r prec_rj
        mu_j = (prec * m_stack).sum(axis=0) / prec.sum(axis=0)  # (N,)
        resid = m_stack - mu_j[None, :]   # (R, N)
        return float(0.5 * np.sum(resid**2 / S + np.log(S)))

    res = minimize_scalar(nll_diag, bounds=bounds_final, method="bounded")
    sigma2_star = float(res.x)

    if verbose:
        print(f"[FGPR] sigma_btw^2 optimized = {sigma2_star:.6g}  "
              f"(diag-NLL = {res.fun:.6g})")
        print(f"[FGPR] search bounds = ({bounds_final[0]:.6g}, "
              f"{bounds_final[1]:.6g})")

    # --- final aggregation at optimal sigma_btw^2 (full covariance) ------
    A = np.zeros((N, N))
    b = np.zeros(N)
    for m, Cpost in zip(m_list, Cpost_list):
        Sigma = Cpost + (sigma2_star + jitter) * I
        cf = cho_factor(Sigma, lower=True, check_finite=False)
        Sinv = cho_solve(cf, I, check_finite=False)
        A += Sinv
        b += Sinv @ m

    mu = np.linalg.solve(A, b)
    Cagg = np.linalg.solve(A, I)  # A^{-1}

    return sigma2_star, mu, Cagg, res


# ---------------------------------------------------------------------------
# Public API: compute_fgpr
# ---------------------------------------------------------------------------

def compute_fgpr(
    y_norm_list: List[np.ndarray],
    cov_norm_list: List[np.ndarray],
    y_scalings: List[ScalingInfo],
    *,
    epsilon: float = 1e-12,
    verbose: bool = True,
) -> FGPRResult:
    """
    Functional Gaussian Process Regression aggregation.

    Given R posterior curves {m_post,r, C_post,r}, finds the optimal
    between-curve variance σ_btw² by profile likelihood and returns
    the aggregated mean and full covariance.

    Parameters
    ----------
    y_norm_list : list of (N,) arrays
        Per-curve posterior means in normalized units.
    cov_norm_list : list of (N, N) arrays
        Per-curve posterior covariances in normalized units.
    y_scalings : list of ScalingInfo
        Per-curve Y-axis scaling info (for transforming back to real scale).
    epsilon : float
        Numerical jitter.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    FGPRResult
        Aggregated posterior mean, covariance, σ_btw², and diagnostics.
    """
    if len(y_norm_list) == 0:
        raise ValueError("No curves provided for FGPR aggregation")

    n_models = len(y_norm_list)
    n_points = y_norm_list[0].shape[0]

    # Validate shapes
    for idx, (y, c) in enumerate(zip(y_norm_list, cov_norm_list)):
        if y.shape != (n_points,):
            raise ValueError(
                f"Curve {idx} mean shape {y.shape} != expected ({n_points},)"
            )
        if c.shape != (n_points, n_points):
            raise ValueError(
                f"Curve {idx} covariance shape {c.shape} != expected ({n_points}, {n_points})"
            )

    # --- optimise σ_btw² and compute aggregated mean / covariance --------
    sigma2_star, mu_norm, Cagg_norm, opt_res = fit_sigma_btw(
        m_list=y_norm_list,
        Cpost_list=cov_norm_list,
        jitter=epsilon,
        verbose=verbose,
    )

    std_norm = np.sqrt(np.clip(np.diag(Cagg_norm), 0.0, None))

    # --- Build per-curve operators: eqs (4.23a-b) -----------------------
    factors = np.array(
        [s.params.get("factor", 1.0) for s in y_scalings], dtype=float,
    )

    curve_ops = build_per_curve_operators(
        Cagg_tilde=Cagg_norm,
        Cpost_tilde_list=cov_norm_list,
        sigma2_btw=sigma2_star,
        factors=factors,
        epsilon=epsilon,
    )

    # --- Scalar weights: eq (4.14a-b) -----------------------------------
    weights = compute_scalar_weights(curve_ops)

    # --- Predictive covariance in normalised space ----------------------
    I = np.eye(n_points)
    Cpred_norm = Cagg_norm + sigma2_star * I
    std_pred_norm = np.sqrt(np.clip(np.diag(Cpred_norm), 0.0, None))

    # --- Rescale to observation scale (corrected) -----------------------
    obs = rescale_fgpr_to_observation(
        mu_norm=mu_norm,
        Cagg_norm=Cagg_norm,
        sigma2_btw=sigma2_star,
        weights=weights,
        factors=factors,
        curve_ops=curve_ops,
        y_norm_list=y_norm_list,
        y_scalings=y_scalings,
    )

    if verbose:
        print(f"[FGPR] aggregated {n_models} curves on {n_points}-point grid")
        print(f"[FGPR] s_agg = {obs['s_agg']:.6g}  "
              f"(sum_r s_r^2 = {obs['sum_s_r_squared']:.6g})")
        print(f"[FGPR] weight stats: min={weights.min():.6g}, "
              f"median={np.median(weights):.6g}, max={weights.max():.6g}")
        print(f"[FGPR] mean(diag(C_agg))={np.mean(np.diag(Cagg_norm)):.6g}, "
              f"mean(diag(C_pred))={np.mean(np.diag(Cpred_norm)):.6g}")

    return FGPRResult(
        y_mean_real=obs['mu_real'],
        y_mean_norm=mu_norm,
        y_std_real=obs['std_real'],
        y_std_norm=std_norm,
        y_cov_real=obs['Cagg_real'],
        y_cov_norm=Cagg_norm,
        y_std_predictive_real=obs['std_pred_real'],
        y_std_predictive_norm=std_pred_norm,
        y_cov_predictive_real=obs['Cpred_real'],
        y_cov_predictive_norm=Cpred_norm,
        sigma_btw_squared=sigma2_star,
        nll_optimized=float(opt_res.fun),
        weights=weights,
        n_models=n_models,
        n_points=n_points,
        optimizer_result=opt_res,
        curve_operators=curve_ops,
        s_agg=obs['s_agg'],
        sum_s_r_squared=obs['sum_s_r_squared'],
    )


# ---------------------------------------------------------------------------
# Public API: compute_fgpr_structured
# ---------------------------------------------------------------------------

def compute_fgpr_structured(
    y_norm_list: List[np.ndarray],
    cov_norm_list: List[np.ndarray],
    y_scalings: List[ScalingInfo],
    t_grid: np.ndarray,
    *,
    btw_cfg: Optional["StructuredBtwConfig"] = None,
    epsilon: float = 1e-12,
    verbose: bool = True,
) -> FGPRResult:
    """
    Functional GPR aggregation with **structured** between-curve covariance.

    Replaces the scalar ``σ²_btw I`` with a multi-component operator::

        Ĉ_btw = σ²_w I + σ²_s K_smooth(ℓ_b)
                 + σ²_o |1⟩⟨1| + σ²_d |t⟩⟨t| + σ²_sc |m_ref⟩⟨m_ref|

    All other pipeline steps (per-curve operators, rescaling, weights)
    remain unchanged.

    Parameters
    ----------
    y_norm_list : list of (N,) arrays
        Per-curve posterior means in normalised units.
    cov_norm_list : list of (N, N) arrays
        Per-curve posterior covariances in normalised units.
    y_scalings : list of ScalingInfo
        Per-curve Y-axis scaling info.
    t_grid : (N,) array
        Common x-grid in **transformed** space (same grid used for GP).
    btw_cfg : StructuredBtwConfig, optional
        Config for enabled components and optimiser settings.
    epsilon : float
        Numerical jitter.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    FGPRResult
        Same structure as ``compute_fgpr`` — fully compatible with
        downstream export / plotting.  ``sigma_btw_squared`` is set
        to ``mean(diag(Ĉ_btw))`` for backward compatibility.
    """
    from .fgpr_structured_btw import (
        StructuredBtwConfig,
        fit_structured_btw,
        effective_sigma2_btw,
    )

    if btw_cfg is None:
        btw_cfg = StructuredBtwConfig()

    if len(y_norm_list) == 0:
        raise ValueError("No curves provided for FGPR-structured aggregation")

    n_models = len(y_norm_list)
    n_points = y_norm_list[0].shape[0]

    for idx, (y, c) in enumerate(zip(y_norm_list, cov_norm_list)):
        if y.shape != (n_points,):
            raise ValueError(f"Curve {idx} mean shape {y.shape} != ({n_points},)")
        if c.shape != (n_points, n_points):
            raise ValueError(f"Curve {idx} cov shape {c.shape} != ({n_points},{n_points})")

    # --- Optimise structured Ĉ_btw --------------------------------------
    C_btw_opt, mu_norm, Cagg_norm, btw_params = fit_structured_btw(
        m_list=y_norm_list,
        Cpost_list=cov_norm_list,
        t_grid=t_grid,
        cfg=btw_cfg,
        verbose=verbose,
    )

    # Effective scalar σ²_btw for backward compat
    sigma2_eff = effective_sigma2_btw(C_btw_opt)

    # --- Remove σ²_sc from C_btw for plotting covariance ----------------
    # σ²_sc |m_ref⟩⟨m_ref| captures how much individual curves follow
    # the average shape.  It is needed during optimisation (for proper
    # weighting) but should NOT inflate the final uncertainty band.
    C_btw_plot = C_btw_opt.copy()
    if btw_params.sigma2_sc > 0 and btw_params.m_ref_hat is not None:
        m_hat = btw_params.m_ref_hat
        C_btw_plot -= btw_params.sigma2_sc * np.outer(m_hat, m_hat)

    # Recompute Cagg using C_btw without σ²_sc
    I_N = np.eye(n_points)
    A_plot = np.zeros((n_points, n_points))
    b_plot = np.zeros(n_points)
    for m_r, Cpost_r in zip(y_norm_list, cov_norm_list):
        Ce_r = Cpost_r + C_btw_plot + epsilon * I_N
        Ce_r = 0.5 * (Ce_r + Ce_r.T)
        try:
            Ce_r_inv = np.linalg.inv(Ce_r)
        except np.linalg.LinAlgError:
            Ce_r_inv = np.linalg.pinv(Ce_r)
        A_plot += Ce_r_inv
        b_plot += Ce_r_inv @ m_r
    try:
        Cagg_norm = np.linalg.solve(A_plot, I_N)
    except np.linalg.LinAlgError:
        Cagg_norm = np.linalg.pinv(A_plot)
    # Keep mu_norm from the full fit (σ²_sc-aware weights gave better mean)

    if verbose:
        print(f"[FGPR-Structured] removed σ²_sc={btw_params.sigma2_sc:.6g} "
              f"from plotting covariance")

    std_norm = np.sqrt(np.clip(np.diag(Cagg_norm), 0.0, None))

    # --- Build per-curve operators using structured C_btw ----------------
    factors = np.array(
        [s.params.get("factor", 1.0) for s in y_scalings], dtype=float,
    )

    curve_ops = build_per_curve_operators(
        Cagg_tilde=Cagg_norm,
        Cpost_tilde_list=cov_norm_list,
        sigma2_btw=sigma2_eff,          # scalar approximation for ops
        factors=factors,
        epsilon=epsilon,
    )

    # Override per-curve operators to use C_btw_plot (without σ²_sc)
    # Ce,r = Cpost,r + C_btw_plot  (not just + σ²I)
    for i_op, (ops, Cpost) in enumerate(zip(curve_ops, cov_norm_list)):
        Ce_tilde = Cpost + C_btw_plot + epsilon * I_N
        try:
            Ce_inv = np.linalg.inv(Ce_tilde)
        except np.linalg.LinAlgError:
            Ce_inv = np.linalg.pinv(Ce_tilde)
        w_tilde = Cagg_norm @ Ce_inv
        ops.C_e_tilde_r = Ce_tilde
        ops.C_e_tilde_inv_r = Ce_inv
        ops.w_tilde_r = w_tilde

    weights = compute_scalar_weights(curve_ops)

    # --- Predictive covariance: C_pred = C_agg + C_btw (without σ²_sc) ---
    Cpred_norm = Cagg_norm + C_btw_plot
    Cpred_norm = 0.5 * (Cpred_norm + Cpred_norm.T)
    std_pred_norm = np.sqrt(np.clip(np.diag(Cpred_norm), 0.0, None))

    # --- Rescale to observation scale -----------------------------------
    obs = rescale_fgpr_to_observation(
        mu_norm=mu_norm,
        Cagg_norm=Cagg_norm,
        sigma2_btw=sigma2_eff,   # scalar used only for legacy Cpred path
        weights=weights,
        factors=factors,
        curve_ops=curve_ops,
        y_norm_list=y_norm_list,
        y_scalings=y_scalings,
    )

    # Override predictive with structured version
    s_agg = obs['s_agg']
    s_agg2 = s_agg ** 2
    Cpred_real = s_agg2 * Cpred_norm
    Cpred_real = 0.5 * (Cpred_real + Cpred_real.T)
    std_pred_real = np.sqrt(np.clip(np.diag(Cpred_real), 0.0, None))

    if verbose:
        print(f"[FGPR-Structured] aggregated {n_models} curves on "
              f"{n_points}-point grid")
        print(f"[FGPR-Structured] s_agg = {s_agg:.6g}")
        print(f"[FGPR-Structured] weight stats: min={weights.min():.6g}, "
              f"median={np.median(weights):.6g}, max={weights.max():.6g}")
        print(f"[FGPR-Structured] s2_eff (mean diag C_btw) = {sigma2_eff:.6g}")

    # Build optimizer_result placeholder with structured params
    class _StructuredOptResult:
        """Mimics scipy OptimizeResult for backward compatibility."""
        def __init__(self, params):
            self.fun = params.nll
            self.x = None
            self.success = params.converged
            self.structured_params = params
    opt_res_compat = _StructuredOptResult(btw_params)

    return FGPRResult(
        y_mean_real=obs['mu_real'],
        y_mean_norm=mu_norm,
        y_std_real=obs['std_real'],
        y_std_norm=std_norm,
        y_cov_real=obs['Cagg_real'],
        y_cov_norm=Cagg_norm,
        y_std_predictive_real=std_pred_real,
        y_std_predictive_norm=std_pred_norm,
        y_cov_predictive_real=Cpred_real,
        y_cov_predictive_norm=Cpred_norm,
        sigma_btw_squared=sigma2_eff,
        nll_optimized=btw_params.nll,
        weights=weights,
        n_models=n_models,
        n_points=n_points,
        optimizer_result=opt_res_compat,
        curve_operators=curve_ops,
        s_agg=obs['s_agg'],
        sum_s_r_squared=obs['sum_s_r_squared'],
        weight_history=btw_params.weight_history,
        curve_history=btw_params.curve_history,
        weight_converged=btw_params.weight_converged,
        max_weight_delta=btw_params.max_weight_delta,
        n_weight_iters=btw_params.n_outer_iter,
    )
