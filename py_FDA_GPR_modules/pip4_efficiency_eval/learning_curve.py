# pip4_efficiency_eval/learning_curve.py
"""
Learning curve computation for Efficiency Evaluation.

Implements Monte-Carlo learning curve analysis to determine
how many curves are needed to reproduce the full-data summary-GPR.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import time
from typing import List, Sequence, Optional, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .efficiency_config import GlobalParams, ScaleParams
from .mc_sampling import mc_repeats, balanced_subset

if TYPE_CHECKING:
    from ..pip3_FDA_scoring_and_aggregations import IndividualGPRData


@dataclass
class SubsetResult:
    """Result from running summary GPR on a subset.

    All ``sigma_btw*`` fields are provided in **both** observation (real)
    and normalised scales.  Exported / plotted values MUST use the
    ``*_real`` variants so that different methods are comparable in
    physical units.
    """
    y_mean: np.ndarray
    y_std_real: np.ndarray
    y_std_norm: np.ndarray
    y_bar: float
    n_iterations: int
    elapsed_time: float
    # Between-curve variance — observation scale
    sigma_btw_real: Optional[float] = None
    # Between-curve variance — normalised scale
    sigma_btw_norm: Optional[float] = None
    cov_matrix: Optional[np.ndarray] = None     # aggregated covariance (N,N) — FGPR only
    # Pointwise between-var (N,) — observation scale
    sigma_btw_pointwise_real: Optional[np.ndarray] = None
    # Pointwise between-var (N,) — normalised scale
    sigma_btw_pointwise_norm: Optional[np.ndarray] = None


@dataclass
class LearningCurveResult:
    """
    Result from learning_curve computation.
    
    Attributes
    ----------
    summary : pd.DataFrame
        Aggregated statistics per subset_size (mean, median, variance, quantiles).
    detailed : pd.DataFrame
        Individual MC run results with columns:
        subset_size, mc_index, error, time_s, n_iterations, sample_indices.
    cov_matrices : dict
        {subset_size: np.ndarray} — representative aggregated covariance
        matrix (FGPR) or None. Stored for the first MC run per subset.
    sigma_btw_pointwise_arrays : dict
        {subset_size: np.ndarray} — representative pointwise between-variance
        array (iterative methods). Stored for the first MC run per subset.
    """
    summary: pd.DataFrame
    detailed: pd.DataFrame
    cov_matrices: Optional[dict] = None
    sigma_btw_pointwise_arrays: Optional[dict] = None


def fast_summary_gpr_core(
    gpr_pool: List["IndividualGPRData"],
    *,
    summary_gpr_config,
    summary_gpr_hyperparams,
    aggregation_method: str = "iterative",
) -> SubsetResult:
    """
    Run summary GPR algorithm on a pool of IndividualGPRData and return results.
    
    Supports five aggregation methods:
    1. Plain average: weight_mode="equal", normalization_summary=False
    2. Plain average (normalised): weight_mode="equal", normalization_summary=True
    3. Pointwise: weight_mode="iterative", weight_scope="point"
    4. Curvewise: weight_mode="iterative", weight_scope="curve"
    5. FGPR: aggregation_method="fgpr" (functional GPR with full covariance)
    
    Methods 1-4 are selected via summary_gpr_config fields;
    method 5 uses the aggregation_method parameter.
    
    Parameters
    ----------
    gpr_pool : List[IndividualGPRData]
        List of individual GPR data objects (from pip3 loader).
    summary_gpr_config : SummaryGPRConfig
        Configuration for summary GPR.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters for summary GPR.
    aggregation_method : str
        "iterative" (default) for methods 1-4, or "fgpr" for method 5.
        
    Returns
    -------
    SubsetResult
        Contains y_mean, y_std_real, y_std_norm, y_bar, n_iterations, elapsed_time.
    """
    # Import here to avoid circular imports
    from ..pip3_FDA_scoring_and_aggregations import compute_summary_gpr, IndividualGPRData

    if aggregation_method.lower() == "fgpr":
        return _fast_fgpr_core(
            gpr_pool,
            summary_gpr_hyperparams=summary_gpr_hyperparams,
            use_observation_scale=not summary_gpr_config.normalization_summary,
        )

    if aggregation_method.lower() == "student_t":
        return _fast_student_t_core(
            gpr_pool,
            summary_gpr_hyperparams=summary_gpr_hyperparams,
            use_observation_scale=not summary_gpr_config.normalization_summary,
        )

    # Convert IndividualGPRData list to arrays using static method
    x_pred, y_array, S_array, y_scalings, sample_ids = IndividualGPRData.to_arrays(gpr_pool)
    
    t0 = time.perf_counter()
    
    # Run summary GPR
    result = compute_summary_gpr(
        y_array=y_array,
        S_array=S_array,
        x_pred=x_pred,
        y_scalings=y_scalings,
        weight_mode=summary_gpr_config.weight_mode,
        weight_scope=summary_gpr_config.weight_scope,
        include_within=summary_gpr_config.include_within_variance,
        include_between=summary_gpr_config.include_between_variance,
        variance_scale=summary_gpr_config.variance_aggregation_scale,
        normalization_summary=summary_gpr_config.normalization_summary,
        epsilon=summary_gpr_hyperparams.epsilon,
        convergence_tol=summary_gpr_hyperparams.convergence_tol,
        max_iterations=summary_gpr_hyperparams.max_iterations,
    )
    
    elapsed = time.perf_counter() - t0
    
    # Number of iterations is length of weight_history
    n_iterations = len(result.weight_history) if result.weight_history else 1
    # Keep both real and normalised stds, and expose selected std for compatibility
    y_std_real = result.y_std_real
    y_std_norm = result.y_std_norm

    # Compute y_bar (average of mean curve)
    y_bar = float(np.mean(np.abs(result.y_mean)))
    
    # --- Compute between-curve variance diagnostics (iterative methods) ---
    # Always compute in BOTH observation and normalised spaces so that
    # exported / plotted values are in consistent physical units.
    n_models = y_array.shape[0]
    y_norm_all = np.array([y_scalings[i].transform(y_array[i]) for i in range(n_models)])
    
    weights = result.weights
    W = weights[:, np.newaxis] if weights.ndim == 1 else weights
    
    # Observation (real) space
    y_mean_real_w = (W * y_array).sum(axis=0)
    sig2_btw_pw_real = (W * (y_array - y_mean_real_w) ** 2).sum(axis=0)
    
    # Normalised space
    y_mean_norm_w = (W * y_norm_all).sum(axis=0)
    sig2_btw_pw_norm = (W * (y_norm_all - y_mean_norm_w) ** 2).sum(axis=0)
    
    sigma_btw_real = float(np.mean(sig2_btw_pw_real))
    sigma_btw_norm = float(np.mean(sig2_btw_pw_norm))
    
    return SubsetResult(
        y_mean=result.y_mean,
        y_std_real=y_std_real,
        y_std_norm=y_std_norm,
        y_bar=y_bar,
        n_iterations=n_iterations,
        elapsed_time=elapsed,
        sigma_btw_real=sigma_btw_real,
        sigma_btw_norm=sigma_btw_norm,
        cov_matrix=None,
        sigma_btw_pointwise_real=sig2_btw_pw_real,
        sigma_btw_pointwise_norm=sig2_btw_pw_norm,
    )


def _fast_fgpr_core(
    gpr_pool: List["IndividualGPRData"],
    *,
    summary_gpr_hyperparams,
    use_observation_scale: bool = False,
) -> SubsetResult:
    """
    Run FGPR aggregation on a pool of IndividualGPRData.
    
    Uses full posterior covariance matrices from each curve.
    
    Two modes:
    - **Normalized scale** (default): uses ``y_pred_normalized`` and
      ``covariance_matrix`` (normalized units), then rescales to
      observation scale via the aggregated scale factor.
    - **Observation scale** (``use_observation_scale=True``): uses
      ``y_pred`` and ``covariance_real`` directly with identity
      scalings (factor = 1).  This bypasses normalization entirely.
    
    Parameters
    ----------
    gpr_pool : List[IndividualGPRData]
        List of individual GPR data objects with covariance_matrix set.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters (epsilon used as jitter).
    use_observation_scale : bool
        If True, run FGPR on raw (observation-scale) predictions and
        covariance matrices with scale factor = 1.
        
    Returns
    -------
    SubsetResult
        FGPR aggregated result.
        
    Raises
    ------
    ValueError
        If any curve lacks a covariance matrix.
    """
    from ..pip3_FDA_scoring_and_aggregations import compute_fgpr, IndividualGPRData
    from pip1_datapreprocessing import ScalingInfo  # type: ignore

    # NOTE: Outlier filtering (near-zero scale factors) is now handled
    # upstream in process_potential_learning_curve() so that the pool is
    # cleaned before MC subset sizes are determined.  The block below is
    # kept as a defensive fallback for direct callers of _fast_fgpr_core.
    ratio_thresh = getattr(summary_gpr_hyperparams, 'fgpr_min_scale_factor_ratio', 0.0)
    if ratio_thresh > 0 and len(gpr_pool) > 1:
        abs_factors = np.array([
            abs(gpr.y_scaling.params.get("factor", 1.0)) for gpr in gpr_pool
        ])
        median_factor = float(np.median(abs_factors))
        min_factor = ratio_thresh * median_factor
        keep_mask = abs_factors >= min_factor
        n_excluded = int((~keep_mask).sum())
        if n_excluded > 0:
            keep_idx = np.where(keep_mask)[0].tolist()
            gpr_pool = [gpr_pool[i] for i in keep_idx]

    y_input_list = []
    cov_input_list = []
    y_scalings = []

    for gpr in gpr_pool:
        if gpr.covariance_matrix is None:
            raise ValueError(
                f"FGPR requires covariance matrices; curve '{gpr.sample_id}' has none."
            )
        if use_observation_scale:
            # Observation-scale mode: use raw predictions / real covariance,
            # identity scalings so scale factor = 1 for every curve.
            y_input_list.append(gpr.y_pred)
            cov_real = gpr.covariance_real
            if cov_real is None:
                raise ValueError(
                    f"FGPR observation-scale mode requires covariance_real; "
                    f"curve '{gpr.sample_id}' could not compute it."
                )
            cov_input_list.append(cov_real)
            y_scalings.append(ScalingInfo.identity())
        else:
            # Normalized-scale mode (default): use normalized predictions.
            y_input_list.append(gpr.y_pred_normalized)
            cov_input_list.append(gpr.covariance_matrix)
            y_scalings.append(gpr.y_scaling)
    
    t0 = time.perf_counter()
    
    result = compute_fgpr(
        y_norm_list=y_input_list,
        cov_norm_list=cov_input_list,
        y_scalings=y_scalings,
        epsilon=summary_gpr_hyperparams.epsilon,
        verbose=False,
    )
    
    elapsed = time.perf_counter() - t0
    
    y_bar = float(np.mean(np.abs(result.y_mean_real)))
    
    # sigma_btw_squared is always in normalised units;
    # convert to observation scale via s_agg².
    s_agg2 = float(result.s_agg ** 2)
    sigma_btw_norm = float(result.sigma_btw_squared)
    sigma_btw_real = float(s_agg2 * result.sigma_btw_squared)
    
    return SubsetResult(
        y_mean=result.y_mean_real,
        y_std_real=result.y_std_real,
        y_std_norm=result.y_std_norm,
        y_bar=y_bar,
        n_iterations=1,  # FGPR is a single-pass method
        elapsed_time=elapsed,
        sigma_btw_real=sigma_btw_real,
        sigma_btw_norm=sigma_btw_norm,
        cov_matrix=result.y_cov_norm,
        sigma_btw_pointwise_real=None,
        sigma_btw_pointwise_norm=None,
    )


def _fast_student_t_core(
    gpr_pool: List["IndividualGPRData"],
    *,
    summary_gpr_hyperparams,
    use_observation_scale: bool = False,
) -> SubsetResult:
    """
    Run Student-t robust aggregation on a pool of IndividualGPRData.

    Uses full posterior covariance matrices from each curve, similar to FGPR.

    Two modes:
    - **Normalized scale** (default): uses ``y_pred_normalized`` and
      ``covariance_matrix`` (normalized units), then rescales to
      observation scale via the aggregated scale factor.
    - **Observation scale** (``use_observation_scale=True``): uses
      ``y_pred`` and ``covariance_real`` directly with identity
      scalings (factor = 1).

    Parameters
    ----------
    gpr_pool : List[IndividualGPRData]
        List of individual GPR data objects with covariance_matrix set.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters (epsilon, student_t_nu, etc.).
    use_observation_scale : bool
        If True, run on raw (observation-scale) predictions.

    Returns
    -------
    SubsetResult
    """
    from ..pip3_FDA_scoring_and_aggregations import (
        compute_student_t_aggregation,
        IndividualGPRData,
    )
    from pip1_datapreprocessing import ScalingInfo  # type: ignore

    y_input_list = []
    cov_input_list = []
    y_scalings = []

    for gpr in gpr_pool:
        if gpr.covariance_matrix is None:
            raise ValueError(
                f"Student-t requires covariance matrices; "
                f"curve '{gpr.sample_id}' has none."
            )
        if use_observation_scale:
            y_input_list.append(gpr.y_pred)
            cov_real = gpr.covariance_real
            if cov_real is None:
                raise ValueError(
                    f"Student-t observation-scale mode requires covariance_real; "
                    f"curve '{gpr.sample_id}' could not compute it."
                )
            cov_input_list.append(cov_real)
            y_scalings.append(ScalingInfo.identity())
        else:
            y_input_list.append(gpr.y_pred_normalized)
            cov_input_list.append(gpr.covariance_matrix)
            y_scalings.append(gpr.y_scaling)

    nu = getattr(summary_gpr_hyperparams, 'student_t_nu', 5.0)
    optimize_nu = getattr(summary_gpr_hyperparams, 'student_t_optimize_nu', True)
    nu_bounds = getattr(summary_gpr_hyperparams, 'student_t_nu_bounds', (1.0, 500.0))
    nu_lb_adaptive = getattr(summary_gpr_hyperparams, 'student_t_nu_lb_adaptive', False)
    max_iter = getattr(summary_gpr_hyperparams, 'student_t_max_iterations', 100)
    conv_tol = getattr(summary_gpr_hyperparams, 'student_t_convergence_tol', 1e-6)
    eps = getattr(summary_gpr_hyperparams, 'epsilon', 1e-12)

    t0 = time.perf_counter()

    result = compute_student_t_aggregation(
        y_norm_list=y_input_list,
        cov_norm_list=cov_input_list,
        y_scalings=y_scalings,
        nu=nu,
        optimize_nu=optimize_nu,
        nu_bounds=nu_bounds,
        nu_lb_adaptive=nu_lb_adaptive,
        max_iterations=max_iter,
        convergence_tol=conv_tol,
        epsilon=eps,
        verbose=False,
    )

    elapsed = time.perf_counter() - t0

    y_bar = float(np.mean(np.abs(result.y_mean_real)))

    # sigma_btw_squared is always in normalised units;
    # convert to observation scale via s_agg².
    s_agg2 = float(result.s_agg ** 2)
    sigma_btw_norm = float(result.sigma_btw_squared)
    sigma_btw_real = float(s_agg2 * result.sigma_btw_squared)

    return SubsetResult(
        y_mean=result.y_mean_real,
        y_std_real=result.y_std_real,
        y_std_norm=result.y_std_norm,
        y_bar=y_bar,
        n_iterations=result.iterations,
        elapsed_time=elapsed,
        sigma_btw_real=sigma_btw_real,
        sigma_btw_norm=sigma_btw_norm,
        cov_matrix=result.y_cov_norm,
        sigma_btw_pointwise_real=None,
        sigma_btw_pointwise_norm=None,
    )


def error_metric(
    ref: np.ndarray,
    ave_ref: float,
    cand: np.ndarray,
    ave_bar: float,
    kind: str,
    scapara: ScaleParams
) -> float:
    """
    Compute distance metric between reference and candidate curves.
    
    Parameters
    ----------
    ref : np.ndarray
        Reference curve (full data).
    ave_ref : float
        Average value of reference for normalization.
    cand : np.ndarray
        Candidate curve (subset).
    ave_bar : float
        Average value of candidate.
    kind : str
        Metric type: "rmse", "mae", or "max".
    scapara : ScaleParams
        Scaling parameters.
        
    Returns
    -------
    float
        Computed error metric.
    """
    if kind.lower() == "rmse":
        err = float(np.sqrt(np.mean((ref - cand) ** 2)))
    elif kind.lower() == "mae":
        err = float(np.mean(np.abs(ref - cand)))
    elif kind.lower() == "max":
        err = float(np.max(np.abs(ref - cand)))
    else:
        raise ValueError(f"Unknown metric: {kind}")

    if scapara.normalize_w_rbar:
        err = np.abs(err / ave_ref)

    if scapara.use_log_error:
        if scapara.log_base_error == "10":
            err = np.log10(err + scapara.eps_error)
        elif scapara.log_base_error == "e":
            err = np.log(err + scapara.eps_error)
        else:
            raise ValueError("log_base_error must be 'e' or '10'")

    return float(err)


def learning_curve(
    gpr_pool: List["IndividualGPRData"],
    ref_mean: np.ndarray,
    ref_bar: float,
    sizes: Sequence[int],
    *,
    summary_gpr_config,
    summary_gpr_hyperparams,
    globpara: GlobalParams,
    scapara: ScaleParams,
    verbose: bool = True,
    aggregation_method: str = "iterative",
) -> LearningCurveResult:
    """
    Compute Monte-Carlo learning curve with iteration statistics.
    
    Parameters
    ----------
    gpr_pool : List[IndividualGPRData]
        All available GPR data objects.
    ref_mean : np.ndarray
        Reference mean from full data.
    ref_bar : float
        Reference average for normalization.
    sizes : Sequence[int]
        Subset sizes to evaluate.
    summary_gpr_config : SummaryGPRConfig
        Configuration for summary GPR.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters for summary GPR.
    globpara : GlobalParams
        Global experiment parameters.
    scapara : ScaleParams
        Scaling parameters.
    verbose : bool
        Print progress.
    aggregation_method : str
        "iterative" for variance-based methods (1-4), "fgpr" for functional GPR (5).
        
    Returns
    -------
    LearningCurveResult
        Contains:
        - summary: DataFrame with aggregated stats per subset_size
        - detailed: DataFrame with individual MC run results
    """
    metric = globpara.metric
    q_low = globpara.q_low
    q_high = globpara.q_high

    idx = np.arange(len(gpr_pool))
    rng = np.random.default_rng(globpara.random_seed)

    summary_rows = []
    detailed_rows = []  # Individual MC run results
    cov_mats = {}       # {subset_size: cov_matrix} for representative FGPR runs
    sbtw_pw = {}        # {subset_size: sigma_btw_pointwise} for representative iterative runs

    total_sizes = len(sizes)
    lc_t0 = time.perf_counter()

    for size_idx, subset_size in enumerate(sizes):
        if verbose:
            print(f"    [{size_idx+1}/{total_sizes}] Evaluating subset size {subset_size}...", flush=True)
        
        errs: List[float] = []
        dts: List[float] = []
        iters: List[int] = []  # Track iteration counts
        sigma_btws: List[float] = []  # Track between-model variance
        
        row: dict = {
            "subset_size": subset_size,
            # Error statistics
            "mean_error": None,
            "median_error": None,
            "variance_error": None,
            "q_error_low": None,
            "q_error_high": None,
            # Time statistics
            "avg_subset_time[s]": None,
            "avg_time_per_elem[s]": None,
            "median_subset_time[s]": None,
            "variance_subset_time[s]": None,
            "q_subset_time_low[s]": None,
            "q_subset_time_high[s]": None,
            # Iteration statistics
            "mean_iterations": None,
            "median_iterations": None,
            "min_iterations": None,
            "max_iterations": None,
            "std_iterations": None,
            # Between-model variance statistics
            "mean_sigma_btw": None,
            "median_sigma_btw": None,
            "std_sigma_btw": None,
            "q_sigma_btw_low": None,
            "q_sigma_btw_high": None,
            "min_sigma_btw": None,
            "max_sigma_btw": None,
        }

        if subset_size == len(gpr_pool):
            # Full dataset case
            result = fast_summary_gpr_core(
                gpr_pool,
                summary_gpr_config=summary_gpr_config,
                summary_gpr_hyperparams=summary_gpr_hyperparams,
                aggregation_method=aggregation_method,
            )
            err = 0.0
            errs.append(err)
            dts.append(result.elapsed_time)
            iters.append(result.n_iterations)
            if result.sigma_btw_real is not None:
                sigma_btws.append(result.sigma_btw_real)
            row["iter0"] = err
            row["iter0_n_iterations"] = result.n_iterations
            
            # Add detailed row for full dataset
            det_row = {
                "subset_size": subset_size,
                "mc_index": 0,
                "error": err,
                "y_std_real": float(np.mean(np.abs(result.y_std_real))),
                "y_std_normalised": float(np.mean(np.abs(result.y_std_norm))),
                "time_s": result.elapsed_time,
                "n_iterations": result.n_iterations,
                "sample_indices": str(list(range(len(gpr_pool)))),
            }
            if result.sigma_btw_real is not None:
                det_row["sigma_btw"] = result.sigma_btw_real
                det_row["sigma_btw_norm"] = result.sigma_btw_norm
            if result.sigma_btw_pointwise_real is not None:
                det_row["mean_sigma_btw_pointwise"] = float(np.mean(result.sigma_btw_pointwise_real))
                det_row["mean_sigma_btw_pointwise_norm"] = float(np.mean(result.sigma_btw_pointwise_norm))
            detailed_rows.append(det_row)

            # Store representative covariance / pointwise arrays
            if result.cov_matrix is not None:
                cov_mats[subset_size] = result.cov_matrix
            if result.sigma_btw_pointwise_real is not None:
                sbtw_pw[subset_size] = result.sigma_btw_pointwise_real
        else:
            iteration_num = mc_repeats(len(gpr_pool), subset_size, globpara=globpara)
            occ = np.zeros(len(gpr_pool), dtype=int)
            
            if verbose:
                print(f"      MC repeats: {iteration_num}", flush=True)
            
            mc_t0 = time.perf_counter()
            # Determine progress reporting interval (every ~10% or at least every 50 iters)
            report_interval = max(1, min(50, iteration_num // 10))
            
            for iteration in range(iteration_num):
                sel = balanced_subset(idx, occ, subset_size, rng)
                occ[sel] += 1

                result = fast_summary_gpr_core(
                    [gpr_pool[i] for i in sel],
                    summary_gpr_config=summary_gpr_config,
                    summary_gpr_hyperparams=summary_gpr_hyperparams,
                    aggregation_method=aggregation_method,
                )
                
                err = error_metric(ref_mean, ref_bar, result.y_mean, result.y_bar, metric, scapara)
                
                # Progress reporting within MC loop
                if verbose and (iteration + 1) % report_interval == 0:
                    mc_elapsed = time.perf_counter() - mc_t0
                    mc_rate = (iteration + 1) / mc_elapsed if mc_elapsed > 0 else 0
                    mc_eta = (iteration_num - iteration - 1) / mc_rate if mc_rate > 0 else float('inf')
                    print(f"        MC iter {iteration+1}/{iteration_num}  "
                          f"({100*(iteration+1)/iteration_num:.0f}%)  "
                          f"elapsed={mc_elapsed:.1f}s  "
                          f"ETA={mc_eta:.0f}s  "
                          f"err={err:.4f}", flush=True)

                errs.append(err)
                dts.append(result.elapsed_time)
                iters.append(result.n_iterations)
                if result.sigma_btw_real is not None:
                    sigma_btws.append(result.sigma_btw_real)
                row[f"iter{iteration}"] = float(err)
                row[f"iter{iteration}_n_iterations"] = result.n_iterations
                
                # Add detailed row for this MC run
                det_row = {
                    "subset_size": subset_size,
                    "mc_index": iteration,
                    "error": float(err),
                    "y_std_real": float(np.mean(np.abs(result.y_std_real))),
                    "y_std_normalised": float(np.mean(np.abs(result.y_std_norm))),
                    "time_s": result.elapsed_time,
                    "n_iterations": result.n_iterations,
                    "sample_indices": str(sorted(sel.tolist())),
                }
                if result.sigma_btw_real is not None:
                    det_row["sigma_btw"] = result.sigma_btw_real
                    det_row["sigma_btw_norm"] = result.sigma_btw_norm
                if result.sigma_btw_pointwise_real is not None:
                    det_row["mean_sigma_btw_pointwise"] = float(np.mean(result.sigma_btw_pointwise_real))
                    det_row["mean_sigma_btw_pointwise_norm"] = float(np.mean(result.sigma_btw_pointwise_norm))
                detailed_rows.append(det_row)

                # Store first MC run's cov / pointwise arrays per subset size
                if subset_size not in cov_mats and result.cov_matrix is not None:
                    cov_mats[subset_size] = result.cov_matrix
                if subset_size not in sbtw_pw and result.sigma_btw_pointwise_real is not None:
                    sbtw_pw[subset_size] = result.sigma_btw_pointwise_real

        # Error statistics
        arr = np.asarray(errs, dtype=float)
        row.update({
            "mean_error": float(arr.mean()),
            "median_error": float(np.median(arr)),
            "variance_error": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
            "q_error_low": float(np.quantile(arr, q_low)),
            "q_error_high": float(np.quantile(arr, q_high)),
        })
        
        # Time statistics
        time_arr = np.asarray(dts, dtype=float)
        row.update({
            "avg_subset_time[s]": float(time_arr.mean()),
            "avg_time_per_elem[s]": float(time_arr.mean() / subset_size),
            "median_subset_time[s]": float(np.median(time_arr)),
            "variance_subset_time[s]": float(time_arr.var(ddof=1)) if time_arr.size > 1 else 0.0,
            "q_subset_time_low[s]": float(np.quantile(time_arr, q_low)),
            "q_subset_time_high[s]": float(np.quantile(time_arr, q_high)),
        })
        
        # Iteration statistics
        iter_arr = np.asarray(iters, dtype=int)
        row.update({
            "mean_iterations": float(iter_arr.mean()),
            "median_iterations": float(np.median(iter_arr)),
            "min_iterations": int(iter_arr.min()),
            "max_iterations": int(iter_arr.max()),
            "std_iterations": float(iter_arr.std(ddof=1)) if iter_arr.size > 1 else 0.0,
        })
        
        # Between-model variance statistics
        if sigma_btws:
            sbtw_arr = np.asarray(sigma_btws, dtype=float)
            row.update({
                "mean_sigma_btw": float(sbtw_arr.mean()),
                "median_sigma_btw": float(np.median(sbtw_arr)),
                "std_sigma_btw": float(sbtw_arr.std(ddof=1)) if sbtw_arr.size > 1 else 0.0,
                "q_sigma_btw_low": float(np.quantile(sbtw_arr, q_low)),
                "q_sigma_btw_high": float(np.quantile(sbtw_arr, q_high)),
                "min_sigma_btw": float(sbtw_arr.min()),
                "max_sigma_btw": float(sbtw_arr.max()),
            })

        if verbose:
            size_elapsed = time.perf_counter() - lc_t0
            sbtw_str = f", mean_sigma_btw={row['mean_sigma_btw']:.4g}" if row.get('mean_sigma_btw') is not None else ""
            print(f"      => subset_size={subset_size} done: "
                  f"mean_err={row['mean_error']:.4f}, "
                  f"mean_iters={row['mean_iterations']:.1f}{sbtw_str}, "
                  f"total_elapsed={size_elapsed:.1f}s", flush=True)
        
        summary_rows.append(row)

    return LearningCurveResult(
        summary=pd.DataFrame(summary_rows),
        detailed=pd.DataFrame(detailed_rows),
        cov_matrices=cov_mats if cov_mats else None,
        sigma_btw_pointwise_arrays=sbtw_pw if sbtw_pw else None,
    )


# =========================================================================
# Layer-by-layer Monte-Carlo learning curve
# =========================================================================

def _default_plot_layers(max_layers: int) -> set:
    """
    Return the set of **1-based** layer numbers at which to regenerate
    the learning-curve plot.

    Schedule: 1, 2, 3, 5, 10, 20, 40, 80, …, max_layers
    (powers of 2 after 10, always includes the final layer).
    """
    layers = {1, 2, 3, 5, 10}
    v = 20
    while v < max_layers:
        layers.add(v)
        v = int(v * 2)
    layers.add(max_layers)
    return {l for l in layers if 1 <= l <= max_layers}


def learning_curve_layered(
    gpr_pool: List["IndividualGPRData"],
    ref_mean: np.ndarray,
    ref_bar: float,
    sizes: Sequence[int],
    *,
    summary_gpr_config,
    summary_gpr_hyperparams,
    globpara: GlobalParams,
    scapara: ScaleParams,
    output_dir: str,
    csv_stem: str,
    verbose: bool = True,
    aggregation_method: str = "iterative",
    resume: bool = True,
    plot_callback=None,
) -> LearningCurveResult:
    """
    Layer-by-layer Monte-Carlo learning curve with real-time CSV persistence.

    Instead of completing **all** MC iterations for one subset_size before
    moving to the next (the behaviour of :func:`learning_curve`), this
    function sweeps one MC *layer* across **all** subset sizes, then moves
    to the next layer.

    Benefits
    --------
    * After layer 1, a (rough) learning curve covering **every** subset size
      is already available.
    * Results are flushed to a CSV file after every layer, so the run can be
      interrupted and resumed later with zero loss.
    * A learning-curve plot is regenerated at logarithmically-spaced layer
      numbers (1, 2, 3, 5, 10, 20, 40, …, max_layers).

    Reproducibility
    ---------------
    Each subset_size gets its own independent RNG (via ``SeedSequence.spawn``)
    so that the sampling sequence for any given size is deterministic and does
    **not** depend on the processing order of other sizes.  On resume, the
    RNG and occurrence counters are replayed from completed rows.

    Parameters
    ----------
    gpr_pool : List[IndividualGPRData]
        All available GPR data objects.
    ref_mean : np.ndarray
        Reference mean from full data.
    ref_bar : float
        Reference average for normalisation.
    sizes : Sequence[int]
        Subset sizes to evaluate.
    summary_gpr_config : SummaryGPRConfig
        Configuration for summary GPR.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters for summary GPR.
    globpara : GlobalParams
        Global experiment parameters (includes ``base_repeats``,
        ``max_enum``, ``random_seed``).
    scapara : ScaleParams
        Scaling parameters.
    output_dir : str
        Directory for the real-time CSV file.
    csv_stem : str
        Base name for output files (e.g. ``"LearningCurve_-1.95"``).
    verbose : bool
        Print progress.
    aggregation_method : str
        ``"iterative"`` for variance-based methods, ``"fgpr"`` for
        functional GPR.
    resume : bool
        If *True* and a CSV from a previous run is found, skip completed
        layers and continue from where the last run stopped.
    plot_callback : callable, optional
        ``plot_callback(df_detailed, layer)`` is called after selected
        layers to generate an interim learning-curve figure.

    Returns
    -------
    LearningCurveResult
        Contains ``summary``, ``detailed``, ``cov_matrices``,
        ``sigma_btw_pointwise_arrays`` — same contract as
        :func:`learning_curve`.
    """
    from .efficiency_plotting import aggregate_detailed_to_summary

    metric = globpara.metric
    N = len(gpr_pool)
    idx = np.arange(N)

    # ---- MC schedule: how many repeats per size ----
    mc_schedule: dict[int, int] = {}
    for s in sizes:
        mc_schedule[s] = 1 if s == N else mc_repeats(N, s, globpara=globpara)
    max_layers = max(mc_schedule.values())

    if verbose:
        total_mc = sum(mc_schedule.values())
        print(f"    Layer-by-layer mode: {len(sizes)} sizes, "
              f"max {max_layers} layers, {total_mc} total MC runs")

    # ---- CSV path ----
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{csv_stem}_detailed.csv")

    # ---- Independent RNG per subset_size ----
    seed_seq = np.random.SeedSequence(globpara.random_seed)
    child_seeds = seed_seq.spawn(len(sizes))
    rngs = {s: np.random.default_rng(seed) for s, seed in zip(sizes, child_seeds)}
    occs = {s: np.zeros(N, dtype=int) for s in sizes}

    # ---- Resume from existing CSV ----
    done_per_size: dict[int, int] = {s: 0 for s in sizes}
    csv_exists = False

    if resume and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty:
                # Safety: check that existing sizes are a subset of expected
                existing_sizes = set(existing_df["subset_size"].unique())
                expected_sizes = set(sizes)
                if not existing_sizes.issubset(expected_sizes):
                    raise ValueError(
                        f"Existing CSV has unexpected subset sizes "
                        f"{existing_sizes - expected_sizes}"
                    )
                for s in sizes:
                    done_per_size[s] = int(
                        (existing_df["subset_size"] == s).sum()
                    )
                total_done = sum(done_per_size.values())
                total_needed = sum(mc_schedule.values())
                csv_exists = True

                if verbose:
                    print(f"    Resuming: {total_done}/{total_needed} MC runs "
                          f"already completed in {csv_path}")

                # All done?
                if total_done >= total_needed:
                    if verbose:
                        print("    All MC runs already completed — "
                              "skipping computation.")
                    summary_df = aggregate_detailed_to_summary(
                        existing_df,
                        q_low=globpara.q_low,
                        q_high=globpara.q_high,
                    )
                    return LearningCurveResult(
                        summary=summary_df,
                        detailed=existing_df,
                        cov_matrices=None,
                        sigma_btw_pointwise_arrays=None,
                    )

                # Replay RNG + occ for completed iterations
                for s in sizes:
                    if s == N:
                        continue  # full-dataset never uses balanced_subset
                    for _ in range(done_per_size[s]):
                        sel = balanced_subset(idx, occs[s], s, rngs[s])
                        occs[s][sel] += 1
        except Exception as e:
            if verbose:
                print(f"    [Warning] Could not resume from {csv_path}: {e}")
                print("    Starting fresh.")
            done_per_size = {s: 0 for s in sizes}
            csv_exists = False
            # Reset RNG + occ
            rngs = {
                s: np.random.default_rng(seed)
                for s, seed in zip(sizes, child_seeds)
            }
            occs = {s: np.zeros(N, dtype=int) for s in sizes}

    if not resume and os.path.exists(csv_path):
        os.remove(csv_path)
        csv_exists = False

    # ---- Plot schedule ----
    plot_layer_set = _default_plot_layers(max_layers)

    # ---- Supplementary outputs (first MC run per size) ----
    cov_mats: dict = {}
    sbtw_pw: dict = {}

    lc_t0 = time.perf_counter()
    total_new_runs = sum(mc_schedule[s] - done_per_size[s] for s in sizes)
    runs_done = 0

    # ---- Layer-by-layer iteration ----
    for layer in range(max_layers):
        active_sizes = [
            s for s in sizes
            if layer < mc_schedule[s] and layer >= done_per_size[s]
        ]
        if not active_sizes:
            continue

        # Progress reporting
        if verbose:
            elapsed = time.perf_counter() - lc_t0
            pct = runs_done / total_new_runs * 100 if total_new_runs > 0 else 100
            eta = (elapsed / runs_done * (total_new_runs - runs_done)
                   if runs_done > 0 else float("inf"))
            eta_str = f"{eta:.0f}s" if eta < float("inf") else "?"
            print(
                f"    Layer {layer+1}/{max_layers}  "
                f"({len(active_sizes)} active sizes)  "
                f"{pct:.0f}% done  "
                f"elapsed={elapsed:.1f}s  ETA={eta_str}",
                flush=True,
            )

        layer_rows: list[dict] = []

        for s in active_sizes:
            if s == N:
                # Full-dataset → single run with error = 0
                result = fast_summary_gpr_core(
                    gpr_pool,
                    summary_gpr_config=summary_gpr_config,
                    summary_gpr_hyperparams=summary_gpr_hyperparams,
                    aggregation_method=aggregation_method,
                )
                err = 0.0
                sel_list = list(range(N))
            else:
                sel = balanced_subset(idx, occs[s], s, rngs[s])
                occs[s][sel] += 1
                result = fast_summary_gpr_core(
                    [gpr_pool[i] for i in sel],
                    summary_gpr_config=summary_gpr_config,
                    summary_gpr_hyperparams=summary_gpr_hyperparams,
                    aggregation_method=aggregation_method,
                )
                err = error_metric(
                    ref_mean, ref_bar,
                    result.y_mean, result.y_bar,
                    metric, scapara,
                )
                sel_list = sorted(sel.tolist())

            det_row: dict = {
                "subset_size": s,
                "mc_index": layer,
                "error": float(err),
                "y_std_real": float(np.mean(np.abs(result.y_std_real))),
                "y_std_normalised": float(np.mean(np.abs(result.y_std_norm))),
                "time_s": result.elapsed_time,
                "n_iterations": result.n_iterations,
                "sample_indices": str(sel_list),
            }
            if result.sigma_btw_real is not None:
                det_row["sigma_btw"] = result.sigma_btw_real
                det_row["sigma_btw_norm"] = result.sigma_btw_norm
            if result.sigma_btw_pointwise_real is not None:
                det_row["mean_sigma_btw_pointwise"] = float(
                    np.mean(result.sigma_btw_pointwise_real)
                )
                det_row["mean_sigma_btw_pointwise_norm"] = float(
                    np.mean(result.sigma_btw_pointwise_norm)
                )
            layer_rows.append(det_row)

            # Store representative cov / pointwise (first run per size)
            if s not in cov_mats and result.cov_matrix is not None:
                cov_mats[s] = result.cov_matrix
            if s not in sbtw_pw and result.sigma_btw_pointwise_real is not None:
                sbtw_pw[s] = result.sigma_btw_pointwise_real

        runs_done += len(layer_rows)

        # ---- Flush layer to CSV ----
        layer_df = pd.DataFrame(layer_rows)
        if not csv_exists:
            layer_df.to_csv(csv_path, index=False, mode="w")
            csv_exists = True
        else:
            layer_df.to_csv(csv_path, index=False, mode="a", header=False)

        # ---- Interim learning-curve plot ----
        if plot_callback is not None and (layer + 1) in plot_layer_set:
            try:
                all_df = pd.read_csv(csv_path)
                plot_callback(all_df, layer + 1)
                if verbose:
                    print(f"      [Plot updated at layer {layer+1}]", flush=True)
            except Exception as exc:
                if verbose:
                    print(
                        f"      [Warning] Plot generation failed "
                        f"at layer {layer+1}: {exc}"
                    )

    # ---- Final result ----
    if verbose:
        elapsed = time.perf_counter() - lc_t0
        print(f"    Layer-by-layer complete: {runs_done} new MC runs "
              f"in {elapsed:.1f}s", flush=True)

    all_df = pd.read_csv(csv_path)
    summary_df = aggregate_detailed_to_summary(
        all_df, q_low=globpara.q_low, q_high=globpara.q_high
    )

    return LearningCurveResult(
        summary=summary_df,
        detailed=all_df,
        cov_matrices=cov_mats if cov_mats else None,
        sigma_btw_pointwise_arrays=sbtw_pw if sbtw_pw else None,
    )


def summarize_learning_curve(
    df_long: pd.DataFrame,
    csv_stem: str,
    out_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Convert the tall Monte-Carlo table into a wide summary table.
    
    Parameters
    ----------
    df_long : pd.DataFrame
        Long-form learning curve DataFrame.
    csv_stem : str
        Base name for output CSV.
    out_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame or None
        Wide-form summary, or None if input is empty.
    """
    if df_long.empty:
        return None

    needed = {
        "mean_error", "median_error", "variance_error",
        "q_error_low", "q_error_high",
        "avg_subset_time[s]", "avg_time_per_elem[s]",
        "median_subset_time[s]", "variance_subset_time[s]",
        "q_subset_time_low[s]", "q_subset_time_high[s]",
        "mean_iterations", "median_iterations", "min_iterations",
        "max_iterations", "std_iterations",
        "mean_sigma_btw", "median_sigma_btw", "std_sigma_btw",
        "q_sigma_btw_low", "q_sigma_btw_high",
        "min_sigma_btw", "max_sigma_btw",
    }
    
    available = needed.intersection(df_long.columns)
    
    wide = (
        df_long
        .set_index("subset_size")[list(available)]
        .T
    )

    wide.index = pd.CategoricalIndex(wide.index, categories=list(wide.index), ordered=True)
    wide = wide.sort_index()

    fname = os.path.join(out_dir, f"{csv_stem}_summary.csv")
    wide.to_csv(fname)
    print(f"    -> Wrote wide summary: {fname}")
    return wide
