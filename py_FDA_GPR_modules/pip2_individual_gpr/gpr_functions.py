# pip2_individual_gpr/gpr_functions.py
"""
Low-level GPR fitting functions for individual curves.

This module provides functions for:
- Fitting GPR models to preprocessed curves
- Validating GPR models
- Generating predictions with uncertainty
- Computing local uncertainty estimates
- Computing full posterior covariance (new v2 framework)

Theoretical Framework Reference:
--------------------------------
See: GPR_derivation_Dirac_notation(R).md

Key equations implemented:
- Posterior mean: |m_post⟩ = C_g,try (C_g,try + σ_f² I)^(-1) |f⟩  [with m_try = 0]
- Posterior cov:  C_post = (σ_f^(-2) I + C_g,try^(-1))^(-1)
- Log-marginal likelihood optimization for hyperparameters

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from scipy.linalg import cholesky, solve_triangular
import scipy.sparse as sp

from .gpr_config import GPRCfg


# =============================================================================
# Data Classes for Structured Results (v2 Framework)
# =============================================================================

@dataclass
class PosteriorResult:
    """
    Container for GPR posterior distribution results.
    
    Follows theoretical framework from Dirac notation derivation.
    
    Attributes (in normalized/transformed space):
    ----------------------------------------------
    mean : np.ndarray
        Posterior mean |m_post,θ⟩ in normalized space. Shape: (n_pred,)
    std : np.ndarray
        Posterior standard deviation (sqrt of diagonal of C_post). Shape: (n_pred,)
    covariance : Optional[np.ndarray]
        Full posterior covariance matrix C_post. Shape: (n_pred, n_pred)
        Only populated if store_posterior_covariance=True.
    covariance_cholesky : Optional[np.ndarray]
        Cholesky factor L where C_post = LL^T. Shape: (n_pred, n_pred)
    covariance_mode : str
        How covariance is stored: 'full', 'diagonal', 'sparse', 'cholesky', 'none'.
    
    Scaling factors (for transforming back to original units):
    ----------------------------------------------------------
    physical_scale_factor : float
        The s_r from per-curve normalization (e.g., steady-state current).
        To get original units: y_original = mean * physical_scale_factor
    statistical_scaler_mean : float
        Mean used by StandardScaler on normalized data.
    statistical_scaler_std : float
        Std used by StandardScaler on normalized data.
    """
    # Posterior in normalized space
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    std: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Covariance storage (optional, memory-intensive)
    covariance: Optional[np.ndarray] = None
    covariance_cholesky: Optional[np.ndarray] = None
    covariance_sparse: Optional[sp.spmatrix] = None
    covariance_mode: str = "full"
    
    # Scaling information
    physical_scale_factor: float = 1.0
    statistical_scaler_mean: float = 0.0
    statistical_scaler_std: float = 1.0
    
    def get_mean_original_units(self) -> np.ndarray:
        """Get posterior mean in original physical units (e.g., A/cm²)."""
        # First undo statistical scaling, then physical scaling
        mean_normalized = self.mean * self.statistical_scaler_std + self.statistical_scaler_mean
        return mean_normalized * self.physical_scale_factor
    
    def get_mean_normalized(self) -> np.ndarray:
        """Get posterior mean in normalized space (after user's preprocessing, before StandardScaler)."""
        # Undo statistical scaling only
        return self.mean * self.statistical_scaler_std + self.statistical_scaler_mean
    
    def get_std_original_units(self) -> np.ndarray:
        """Get posterior std in original physical units."""
        # Variance scales by (s_r * σ_stat)²
        # Use abs() because scale factor can be negative (for negative currents)
        std_normalized = self.std * self.statistical_scaler_std
        return std_normalized * abs(self.physical_scale_factor)
    
    def get_std_normalized(self) -> np.ndarray:
        """Get posterior std in normalized space (after user's preprocessing, before StandardScaler)."""
        # Undo statistical scaling only
        return self.std * self.statistical_scaler_std
    
    def get_covariance_original_units(self) -> Optional[np.ndarray]:
        """Get full covariance in original physical units."""
        if self.covariance is None:
            return None
        # Covariance scales by (s_r * σ_stat)²
        # Use abs() because scale factor can be negative
        scale = (abs(self.physical_scale_factor) * self.statistical_scaler_std) ** 2
        return self.covariance * scale
    
    def get_covariance_normalized(self) -> Optional[np.ndarray]:
        """Get full covariance in normalized space (after user's preprocessing, before StandardScaler)."""
        if self.covariance is None:
            return None
        # Covariance scales by σ_stat²
        scale = self.statistical_scaler_std ** 2
        return self.covariance * scale
    
    def get_covariance_diagonal(self) -> np.ndarray:
        """Get variance (diagonal of covariance) regardless of storage mode."""
        if self.covariance_mode == "diagonal" or self.covariance is None:
            return self.std ** 2
        elif self.covariance_mode == "full":
            return np.diag(self.covariance)
        elif self.covariance_mode == "cholesky" and self.covariance_cholesky is not None:
            # C = LL^T, diag(C) = sum(L_ij^2, j)
            return np.sum(self.covariance_cholesky ** 2, axis=1)
        elif self.covariance_mode == "sparse" and self.covariance_sparse is not None:
            return np.array(self.covariance_sparse.diagonal())
        return self.std ** 2


# =============================================================================
# Core GPR Functions - Based on Dirac Notation Derivation
# =============================================================================

def perform_gpr(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    gpr_cfg: GPRCfg,
) -> Tuple[Optional[GaussianProcessRegressor], Optional[StandardScaler], 
           Optional[StandardScaler], Optional[str], Optional[Dict[str, Any]]]:
    """
    Perform Gaussian Process Regression with theoretical framework alignment.
    
    This version explicitly tracks the relationship between:
    - Physical normalization (s_r): Done in pip1 preprocessing
    - Statistical normalization: StandardScaler applied here
    - GPR kernel hyperparameters (θ): Optimized via L-BFGS-B
    
    Theoretical context (from Dirac notation derivation):
    -----------------------------------------------------
    The GPR trains on downsampled data to optimize hyperparameters:
    |ϑ*⟩ = arg(max_ϑ(P(|f⟩||ϑ⟩)))
    
    where log-marginal likelihood is:
    log(P(|f⟩||ϑ⟩)) = -½⟨f|K(ϑ)^(-1)|f⟩ - ½log|K(ϑ)| - N/2·log(2π)
    
    Parameters
    ----------
    X_raw : np.ndarray
        Input features (n_samples, 1) - typically log-transformed time |J⟩.
    y_raw : np.ndarray
        Target values (n_samples,) - normalized current (already scaled by s_r in pip1).
    gpr_cfg : GPRCfg
        GPR configuration.
        
    Returns
    -------
    Tuple containing:
        - gpr: Fitted GaussianProcessRegressor
        - scaler_X: StandardScaler for X (statistical normalization)
        - scaler_y: StandardScaler for y (statistical normalization)  
        - optimized_hyperparams_str: String representation of optimized kernel K(ϑ*)
        - hyperparams: Dict of kernel hyperparameters
    """
    # Same fitting logic as v1, but with clearer documentation
    kernel = gpr_cfg.kernel
    n_restarts_optimizer = gpr_cfg.n_restarts_optimizer
    alpha = gpr_cfg.alpha  # This is σ_f² (measurement noise variance)
    normalize_y = gpr_cfg.normalize_y

    # Statistical scaling (separate from physical s_r scaling done in pip1)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    # Theory alignment (Eqn 2.19a-b, 5.1c-d):
    #   kernel = k_m,try + σ_m² I  (ConstantKernel*Matern + WhiteKernel)
    #   alpha  = σ_f²              (observation noise, separate from model noise)
    # When WhiteKernel is present, it learns σ_m² (model noise). Setting
    # alpha=0 here avoids double-counting during fitting. For our test,
    # σ_f = 0 (observations treated as ground truth |f⟩ = |g⟩).
    effective_alpha = alpha
    try:
        def _has_white(kern):
            if kern is None:
                return False
            if kern.__class__.__name__ == "WhiteKernel":
                return True
            return any(_has_white(getattr(kern, child, None)) for child in ("k1", "k2"))
        if _has_white(kernel):
            effective_alpha = 0.0
    except Exception:
        effective_alpha = alpha

    gpr = GaussianProcessRegressor(
        kernel=kernel,  # type: ignore[arg-type]
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=effective_alpha,
        normalize_y=normalize_y
    )

    # ------------------------------------------------------------------
    # Fit with retry-on-MemoryError strategy
    # ------------------------------------------------------------------
    # On Windows, numpy can fail to allocate the kernel gradient array
    # (n, n, n_params) even when plenty of RAM is available, likely due
    # to address-space fragmentation or SMB/network-drive issues.
    # Strategy:
    #   1) Try full data with original n_restarts_optimizer.
    #   2) On MemoryError → retry with n_restarts_optimizer=0.
    #   3) Still fails → retry with subsampled training data (max 200 pts)
    #      and n_restarts_optimizer=0.
    # ------------------------------------------------------------------
    fit_X, fit_y = X, y
    for attempt in range(3):
        try:
            gpr.fit(fit_X, fit_y)
            if attempt > 0:
                print(f"  GPR fit succeeded on attempt {attempt + 1} "
                      f"(n_train={fit_X.shape[0]})")
            optimized_hyperparams_str = str(gpr.kernel_)
            hyperparams = gpr.kernel_.get_params()
            hyperparams['log_marginal_likelihood'] = gpr.log_marginal_likelihood_value_
            if attempt > 0:
                hyperparams['fit_attempt'] = attempt + 1
                hyperparams['fit_n_train'] = int(fit_X.shape[0])
            return gpr, scaler_X, scaler_y, optimized_hyperparams_str, hyperparams
        except MemoryError as me:
            if attempt == 0:
                # Retry 1: same data, no restarts
                print(f"  MemoryError (n={fit_X.shape[0]}, restarts="
                      f"{n_restarts_optimizer}), retrying with 0 restarts...")
                gpr = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(kernel.theta),
                    n_restarts_optimizer=0,
                    alpha=effective_alpha,
                    normalize_y=normalize_y,
                )
            elif attempt == 1:
                # Retry 2: subsample + no restarts
                max_pts = min(200, fit_X.shape[0] // 2)
                if max_pts < 20:
                    print(f"Error fitting GPR: too few points for subsample")
                    return None, None, None, None, None
                idx = np.round(np.linspace(0, fit_X.shape[0] - 1, max_pts)).astype(int)
                fit_X, fit_y = X[idx], y[idx]
                print(f"  MemoryError persists, retrying with {max_pts} "
                      f"subsampled points...")
                gpr = GaussianProcessRegressor(
                    kernel=kernel.clone_with_theta(kernel.theta),
                    n_restarts_optimizer=0,
                    alpha=effective_alpha,
                    normalize_y=normalize_y,
                )
            else:
                print(f"Error fitting GPR: {me}")
                return None, None, None, None, None
        except Exception as e:
            print(f"Error fitting GPR: {e}")
            return None, None, None, None, None


def generate_predictions(
    gpr: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    X_pred: np.ndarray,
    physical_scale_factor: float,
    *,
    gpr_cfg: GPRCfg,
) -> PosteriorResult:
    """
    Generate predictions with full posterior distribution.
    
    Theoretical context (from Dirac notation derivation):
    -----------------------------------------------------
    Posterior distribution: P(|g⟩||f⟩) ∝ exp(-½⟨g-m_post|C_post^(-1)|g-m_post⟩)
    
    Posterior mean:
        |m_post⟩ = C_g,try (C_g,try + σ_f² I)^(-1) |f⟩   [with m_try = 0]
    
    Posterior covariance (ground truth / latent function):
        C_g,post = K** - K*0 (K00 + α I)^(-1) K0*
    
    where K = k_m,try + σ_m² I  (Eqn 2.19a-b, the full sklearn kernel).
    sklearn's predict(return_cov=True) returns exactly C_g,post.
    
    Observation posterior covariance:
        C_f,post = C_g,post + σ_f² I   (only if σ_f > 0)
    
    For FGPR (Eqn 3.5a): C_e,r = C_post,r + σ_btw² I, where C_post,r
    is C_g,post output by this function. With σ_f = 0, C_f,post = C_g,post.
    
    Scaling back to original units:
        For normalized data ỹ = y/s_r, the posterior in original units is:
        |g_original⟩ ~ N(s_r |m̃_post⟩, s_r² C̃_post)
    
    Parameters
    ----------
    gpr : GaussianProcessRegressor
        Fitted GPR model with optimized kernel K(ϑ*).
    scaler_X : StandardScaler
        Statistical scaler for X.
    scaler_y : StandardScaler
        Statistical scaler for y.
    X_pred : np.ndarray
        Prediction points (n_points, 1) in transformed x-space (e.g., log-time).
    physical_scale_factor : float
        The s_r factor from pip1 normalization (e.g., steady-state current).
    gpr_cfg : GPRCfg
        Configuration for covariance storage options.
        
    Returns
    -------
    PosteriorResult
        Complete posterior distribution with mean, std, and optionally full covariance.
    """
    X_pred_scaled = scaler_X.transform(X_pred)
    n_pred = len(X_pred_scaled)
    
    # Get statistical scaler parameters
    stat_mean = scaler_y.mean_[0] if scaler_y.mean_ is not None else 0.0
    stat_std = scaler_y.scale_[0] if scaler_y.scale_ is not None else 1.0
    
    # Determine what covariance information to compute
    store_cov = gpr_cfg.store_posterior_covariance
    cov_mode = gpr_cfg.covariance_storage_mode
    
    # Extract observation noise σ_f² (alpha) ONLY.
    #
    # CRITICAL: Do NOT include WhiteKernel noise_level (σ_m²) here.
    # The WhiteKernel represents model uncertainty σ_m², which is part of
    # the prior kernel C_g,try = k_m,try + σ_m² I  (Eqn 2.19a-b).
    # sklearn's posterior C_g,post already accounts for σ_m² through
    # K(X*, X*) in:  C_g,post = K** - K*0 (K00 + α I)^{-1} K0*
    # Adding σ_m² again would double-count model noise.
    #
    # For C_f,post = C_g,post + σ_f² I, only α (= σ_f²) should be added.
    # With σ_f = 0 (our test: observations treated as ground truth),
    # C_f,post = C_g,post and no noise is added.
    # See: GPR_derivation_Dirac_notation(R4).md, Eqn.(2.19), Eqn.(5.1c-d)
    obs_noise_var = float(getattr(gpr, 'alpha', 0.0) or 0.0)

    if store_cov and cov_mode == "full":
        # Get full posterior covariance from sklearn (latent function covariance)
        y_pred_scaled, cov_scaled = gpr.predict(X_pred_scaled, return_cov=True)

        # Add observation noise variance on the diagonal for predictive covariance of observations
        if obs_noise_var and obs_noise_var > 0.0:
            cov_scaled = cov_scaled + (obs_noise_var * np.eye(n_pred))

        y_std_scaled = np.sqrt(np.diag(cov_scaled))

        return PosteriorResult(
            mean=y_pred_scaled,
            std=y_std_scaled,
            covariance=cov_scaled,
            covariance_mode="full",
            physical_scale_factor=physical_scale_factor,
            statistical_scaler_mean=stat_mean,
            statistical_scaler_std=stat_std,
        )
    
    elif store_cov and cov_mode == "cholesky":
        # Get full covariance, then store Cholesky factor
        y_pred_scaled, cov_scaled = gpr.predict(X_pred_scaled, return_cov=True)

        # Add observation noise variance on the diagonal
        if obs_noise_var and obs_noise_var > 0.0:
            cov_scaled = cov_scaled + (obs_noise_var * np.eye(n_pred))

        y_std_scaled = np.sqrt(np.diag(cov_scaled))
        
        try:
            # Add small jitter for numerical stability
            cov_scaled_stable = cov_scaled + 1e-10 * np.eye(n_pred)
            L = cholesky(cov_scaled_stable, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to diagonal if Cholesky fails
            L = None
            cov_mode = "diagonal"
        
        return PosteriorResult(
            mean=y_pred_scaled,
            std=y_std_scaled,
            covariance_cholesky=L,
            covariance_mode=cov_mode,
            physical_scale_factor=physical_scale_factor,
            statistical_scaler_mean=stat_mean,
            statistical_scaler_std=stat_std,
        )
    
    elif store_cov and cov_mode == "sparse":
        # Get full covariance, then sparsify
        y_pred_scaled, cov_scaled = gpr.predict(X_pred_scaled, return_cov=True)

        # Add observation noise variance on the diagonal
        if obs_noise_var and obs_noise_var > 0.0:
            cov_scaled = cov_scaled + (obs_noise_var * np.eye(n_pred))

        y_std_scaled = np.sqrt(np.diag(cov_scaled))
        
        # Threshold small values
        threshold = gpr_cfg.covariance_sparse_threshold
        cov_sparse = sp.csr_matrix(np.where(np.abs(cov_scaled) < threshold, 0, cov_scaled))
        
        return PosteriorResult(
            mean=y_pred_scaled,
            std=y_std_scaled,
            covariance_sparse=cov_sparse,
            covariance_mode="sparse",
            physical_scale_factor=physical_scale_factor,
            statistical_scaler_mean=stat_mean,
            statistical_scaler_std=stat_std,
        )
    
    else:
        # Diagonal only (default, memory-efficient)
        pred_result = gpr.predict(X_pred_scaled, return_std=True)
        y_pred_scaled, y_std_scaled = pred_result[0], pred_result[1]
        # Include observation noise variance in returned std
        if obs_noise_var and obs_noise_var > 0.0:
            y_std_scaled = np.sqrt(y_std_scaled ** 2 + obs_noise_var)

        return PosteriorResult(
            mean=y_pred_scaled,
            std=y_std_scaled,
            covariance_mode="diagonal",
            physical_scale_factor=physical_scale_factor,
            statistical_scaler_mean=stat_mean,
            statistical_scaler_std=stat_std,
        )


def validate_gpr(
    X_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
    gpr: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
) -> Dict[str, float]:
    """
    Validate GPR with extended metrics.
    
    Returns additional metrics useful for curve aggregation:
    - MAE, RMSE (standard)
    - Mean negative log-predictive density (NLPD) - proper probabilistic scoring
    - Calibration metrics (fraction within 1σ, 2σ)
    
    Parameters
    ----------
    X_val_raw : np.ndarray
        Validation input features in transformed space.
    y_val_raw : np.ndarray
        Validation target values in normalized space.
    gpr : GaussianProcessRegressor
        Fitted GPR model.
    scaler_X, scaler_y : StandardScaler
        Statistical scalers.
        
    Returns
    -------
    Dict[str, float]
        Validation metrics dictionary.
    """
    X_val = scaler_X.transform(X_val_raw)
    pred_result = gpr.predict(X_val, return_std=True)
    y_pred_scaled, y_std_scaled = pred_result[0], pred_result[1]
    # Only include observation noise α = σ_f² (NOT WhiteKernel σ_m²).
    # WhiteKernel noise is already in the kernel; sklearn's predict
    # accounts for it in the posterior std. See generate_predictions.
    obs_noise = float(getattr(gpr, 'alpha', 0.0) or 0.0)

    # Transform predictions back to normalized space
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Scale std, and include observation noise (scaled by statistical std)
    # NOTE: obs_noise is a VARIANCE (σ²), not a std. To convert variance from
    # scaled-y space to normalized-y space, multiply by scale². 
    stat_scale = (scaler_y.scale_[0] if scaler_y.scale_ is not None else 1.0)
    y_std = y_std_scaled * stat_scale
    if obs_noise and obs_noise > 0.0:
        # obs_noise (variance) in scaled space → multiply by scale² to get normalized space
        y_std = np.sqrt(y_std ** 2 + obs_noise * (stat_scale ** 2))
    
    # Standard metrics
    mae = mean_absolute_error(y_val_raw, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val_raw, y_pred))
    
    # Negative log-predictive density (proper probabilistic score)
    # NLPD = 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * ((y - μ)/σ)²
    residuals = y_val_raw - y_pred
    nlpd = 0.5 * np.log(2 * np.pi) + 0.5 * np.log(y_std**2 + 1e-10) + 0.5 * (residuals / (y_std + 1e-10))**2
    mean_nlpd = np.mean(nlpd)
    
    # Calibration: fraction within 1σ, 2σ (should be ~68%, ~95% for well-calibrated)
    within_1sigma = np.mean(np.abs(residuals) <= y_std)
    within_2sigma = np.mean(np.abs(residuals) <= 2 * y_std)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'NLPD': mean_nlpd,
        'within_1sigma': within_1sigma,
        'within_2sigma': within_2sigma,
    }


def compute_full_posterior_covariance(
    gpr: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    X_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute full posterior covariance matrix C_g,post(J, J').
    
    This returns the GROUND TRUTH / LATENT function posterior covariance,
    which is what FGPR (pip3) needs for aggregation (Eqn 3.5a).
    
    Theoretical context:
    -------------------
    C_g,post = K** - K*0 (K + σ_f² I)^(-1) K0*
    
    where K = k_m,try + σ_m² I (Eqn 2.19a-b).
    With σ_f = 0: C_g,post = K** - K*0 K^(-1) K0*
    
    In sklearn's implementation, this is computed via:
    C_post = kernel_(X*) - K(X*, X_train) @ L^{-T} L^{-1} @ K(X_train, X*)
    
    Parameters
    ----------
    gpr : GaussianProcessRegressor
        Fitted GPR model.
    scaler_X : StandardScaler
        Scaler for input features.
    X_pred : np.ndarray
        Prediction points (n_pred, 1).
        
    Returns
    -------
    np.ndarray
        Full posterior covariance matrix (n_pred, n_pred).
    """
    X_pred_scaled = scaler_X.transform(X_pred)
    _, cov = gpr.predict(X_pred_scaled, return_cov=True)
    return cov


# =============================================================================
# Grid Regulation Functions - For Shared Annotation Basis
# =============================================================================

def refit_with_frozen_hyperparameters(
    gpr_trained: GaussianProcessRegressor,
    X_all: np.ndarray,
    y_all: np.ndarray,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    alpha: float,
) -> GaussianProcessRegressor:
    """
    Refit GPR on ALL data with frozen (fixed) hyperparameters.
    
    This is Step B in the regulation workflow:
    - After hyperparameter tuning on train/val split, we fix θ* and refit
      on all available data to maximize information in the posterior.
    
    Theoretical context (from Dirac notation derivation):
    -----------------------------------------------------
    After optimization: θ* = arg max_θ P(|f_train⟩|θ)
    
    Refitting with frozen θ*:
    - The posterior is P(|g⟩|f_all, θ*) with kernel K(θ*) fixed
    - More data points in the conditioning set → tighter posterior
    - sklearn: setting optimizer=None freezes hyperparameters
    
    Parameters
    ----------
    gpr_trained : GaussianProcessRegressor
        GPR model with optimized kernel (from training phase).
    X_all : np.ndarray
        ALL input features (train + val + test), shape (n_total, 1).
    y_all : np.ndarray
        ALL target values, shape (n_total,).
    scaler_X : StandardScaler
        X scaler (already fitted on training data).
    scaler_y : StandardScaler
        y scaler (already fitted on training data).
    alpha : float
        Noise variance term σ_f².
        
    Returns
    -------
    GaussianProcessRegressor
        New GPR fitted on all data with frozen hyperparameters.
    """
    # Get the optimized kernel from training
    kernel_star = gpr_trained.kernel_
    
    # Scale the full dataset using the SAME scalers (fit on training data)
    X_all_scaled = scaler_X.transform(X_all)
    y_all_scaled = scaler_y.transform(y_all.reshape(-1, 1)).ravel()
    
    # Create new GPR with frozen hyperparameters
    gpr_full = GaussianProcessRegressor(
        kernel=kernel_star,  # Optimized kernel θ*
        optimizer=None,      # CRITICAL: Freeze hyperparameters
        alpha=alpha,
        normalize_y=False,   # We already normalized manually
    )
    
    # Fit on all data (only computes posterior, no hyperparameter optimization)
    gpr_full.fit(X_all_scaled, y_all_scaled)
    
    return gpr_full


def regulate_to_shared_grid(
    gpr_refitted: GaussianProcessRegressor,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    X_shared: np.ndarray,
    physical_scale_factor: float,
    *,
    gpr_cfg: GPRCfg,
) -> PosteriorResult:
    """
    Regulate GPR predictions to a shared grid J_R.
    
    This is Step C in the regulation workflow:
    - Given a GPR fitted on all data with frozen θ*, evaluate the posterior
      on a shared grid J_R that is common to all curves.
    
    Theoretical context:
    -------------------
    For regulated grid J_R:
        m_R = m_post(J_R) = mean prediction on shared grid
        C_R = C_post(J_R, J_R) = covariance on shared grid
    
    The key insight: Even though different curves have different native grids,
    we evaluate ALL posteriors on the SAME shared grid J_R. This enables:
    - Proper FDA aggregation (all curves on same functional basis)
    - Consistent covariance matrices (same dimensions for all curves)
    - Lossless transformation (full posterior information preserved)
    
    Parameters
    ----------
    gpr_refitted : GaussianProcessRegressor
        GPR model fitted on all data with frozen hyperparameters.
    scaler_X : StandardScaler
        X scaler.
    scaler_y : StandardScaler
        y scaler.
    X_shared : np.ndarray
        Shared grid J_R, shape (n_grid, 1).
        Must be in the TRANSFORMED x-space (e.g., log-time).
    physical_scale_factor : float
        The s_r from pip1 normalization.
    gpr_cfg : GPRCfg
        Configuration for covariance storage.
        
    Returns
    -------
    PosteriorResult
        Complete posterior on the shared grid.
    """
    # Use the generate_predictions function with the shared grid
    return generate_predictions(
        gpr_refitted,
        scaler_X,
        scaler_y,
        X_shared,
        physical_scale_factor,
        gpr_cfg=gpr_cfg,
    )


def full_regulation_workflow(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    X_shared: np.ndarray,
    physical_scale_factor: float,
    *,
    gpr_cfg: GPRCfg,
) -> Tuple[
    PosteriorResult,
    GaussianProcessRegressor,
    GaussianProcessRegressor,
    StandardScaler,
    StandardScaler,
    str,
    Dict[str, Any],
]:
    """
    Complete regulation workflow: train → refit → regulate.
    
    This implements the full three-step process:
    
    Step A - Hyperparameter tuning:
        Fit GPR on training data to optimize θ.
        Result: kernel_* = K(θ*)
        
    Step B - Refit with frozen θ*:
        Refit on ALL available data with optimizer disabled.
        Result: GPR conditioned on all points, kernel parameters fixed.
        
    Step C - Regulate to shared grid:
        Evaluate posterior on shared grid J_R.
        Result: m_R, C_R (mean and covariance on shared grid)
    
    Parameters
    ----------
    X_train : np.ndarray
        Training X values (for hyperparameter optimization), shape (n_train, 1).
    y_train : np.ndarray
        Training y values, shape (n_train,).
    X_all : np.ndarray
        ALL X values (train + val), shape (n_all, 1).
    y_all : np.ndarray
        ALL y values, shape (n_all,).
    X_shared : np.ndarray
        Shared grid J_R, shape (n_grid, 1).
    physical_scale_factor : float
        The s_r from pip1 normalization.
    gpr_cfg : GPRCfg
        GPR configuration.
        
    Returns
    -------
    Tuple containing:
        - posterior: PosteriorResult on shared grid
        - gpr_refitted: The refitted GPR model (all data, θ* frozen)
        - gpr_trained: The training-only GPR (θ* optimized on train split)
        - scaler_X: X scaler (fit on training data)
        - scaler_y: y scaler (fit on training data)
        - hyperparams_str: String of optimized kernel
        - hyperparams: Dict of kernel parameters
    """
    # Step A: Hyperparameter optimization on training data
    gpr_trained, scaler_X, scaler_y, hyperparams_str, hyperparams = perform_gpr(
        X_train, y_train, gpr_cfg=gpr_cfg
    )
    
    if gpr_trained is None:
        raise ValueError("GPR training failed")
    
    assert scaler_X is not None and scaler_y is not None
    
    # Step B: Refit on all data with frozen hyperparameters
    if gpr_cfg.shared_grid.enabled and gpr_cfg.shared_grid.refit_on_full_data:
        gpr_refitted = refit_with_frozen_hyperparameters(
            gpr_trained, X_all, y_all, scaler_X, scaler_y, gpr_cfg.alpha
        )
    else:
        # Skip refitting, use training-only GPR
        gpr_refitted = gpr_trained
    
    # Step C: Regulate to shared grid
    posterior = regulate_to_shared_grid(
        gpr_refitted, scaler_X, scaler_y, X_shared,
        physical_scale_factor, gpr_cfg=gpr_cfg
    )
    
    # Return both refitted (all-data) model and the training-only model so callers
    # can use the train-only GPR for out-of-sample calibration without any scaler
    # mismatch. Both models share the same scalers (fit on training data).
    return posterior, gpr_refitted, gpr_trained, scaler_X, scaler_y, hyperparams_str, hyperparams