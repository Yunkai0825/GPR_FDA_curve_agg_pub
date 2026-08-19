# pip3_FDA_scoring_and_aggregations/pip3_summary_gpr_orchestrator.py
"""
High-level orchestrator for Summary GPR processing.

This module ties together:
- Loading individual GPRs with metadata
- Interpolating to common X grid
- Running core aggregation algorithm
- Saving results and plots

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from scipy.interpolate import interp1d

from .summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
from .summary_gpr_loader import (
    IndividualGPRData,
    load_all_individual_gprs,
    group_gprs_by_key,
)

# Method orchestrators
from .iterative_weight_sum_GPR.iterative_orchestrator import IterativeGPROrchestrator
from .operator_fusion_noweight.operator_fusion_orchestrator import OperatorFusionOrchestrator
from .functional_GPR.fgpr_orchestrator import FGPROrchestrator
from .student_t_agg_iterative.student_t_orchestrator import StudentTOrchestrator
from .comparison_orchestrator import ComparisonOrchestrator

# Result types (still needed for SummaryGPRResult fields)
from .operator_fusion_noweight.operator_fusion_weight_helpers import OperatorFusionResult
from .functional_GPR.fgpr_helpers import FGPRResult
from .student_t_agg_iterative.student_t_core import StudentTResult

# Import ScalingInfo from pip1
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from pip1_datapreprocessing import ScalingInfo


@dataclass
class SummaryGPRResult:
    """
    Result from Summary GPR aggregation for one group.
    
    Attributes
    ----------
    group_key : str
        Group identifier (e.g., "potential=-1.95").
    x_pred_transformed : np.ndarray
        Common X grid in transformed space.
    x_pred_original : np.ndarray
        Common X grid in original space.
    y_mean : np.ndarray
        Aggregated mean prediction.
    y_mean_norm : np.ndarray
        Aggregated mean prediction in normalized scale.
    y_std_real : np.ndarray
        Aggregated standard deviation using real-scale variance.
    y_std_norm : np.ndarray
        Aggregated standard deviation using normalized-scale variance.
    weights : np.ndarray
        Final weights (curve-level or point-level).
    weight_history : List[np.ndarray]
        Weight history from optimization.
    curve_history : List[np.ndarray]
        Summary curve evolution during optimization.
    n_curves : int
        Number of curves aggregated.
    sample_ids : List[str]
        Sample identifiers in the group.
    y_scaling : ScalingInfo
        Aggregated Y-axis scaling info.
    x_scaling : ScalingInfo
        X-axis scaling info (shared across group).
    """
    group_key: str
    x_pred_transformed: np.ndarray
    x_pred_original: np.ndarray
    y_mean: np.ndarray
    y_mean_norm: np.ndarray
    y_std_real: np.ndarray
    y_std_norm: np.ndarray
    weights: np.ndarray
    weight_history: List[np.ndarray]
    curve_history: List[np.ndarray]
    n_curves: int
    sample_ids: List[str]
    y_scaling: ScalingInfo = field(default_factory=ScalingInfo.identity)
    x_scaling: ScalingInfo = field(default_factory=ScalingInfo.identity)

    # Optional operator-fusion outputs
    operator_mean: Optional[np.ndarray] = None
    operator_std: Optional[np.ndarray] = None
    operator_mean_norm: Optional[np.ndarray] = None
    operator_cov: Optional[np.ndarray] = None
    operator_cov_norm: Optional[np.ndarray] = None
    operator_weights: Optional[np.ndarray] = None
    operator_iterations: Optional[int] = None
    operator_weight_history: Optional[List[np.ndarray]] = None
    operator_between_variance: Optional[float] = None
    operator_within_variance_norm: Optional[float] = None
    operator_within_variance_real: Optional[float] = None

    # Optional FGPR outputs
    fgpr_mean: Optional[np.ndarray] = None
    fgpr_std: Optional[np.ndarray] = None
    fgpr_mean_norm: Optional[np.ndarray] = None
    fgpr_std_norm: Optional[np.ndarray] = None
    fgpr_cov: Optional[np.ndarray] = None
    fgpr_cov_norm: Optional[np.ndarray] = None
    fgpr_weights: Optional[np.ndarray] = None
    fgpr_sigma_btw_squared: Optional[float] = None
    fgpr_nll: Optional[float] = None
    fgpr_within_variance_norm: Optional[float] = None
    fgpr_within_variance_real: Optional[float] = None
    # FGPR predictive: C_pred = C_agg + sigma_btw^2 * I
    fgpr_std_predictive: Optional[np.ndarray] = None   # real scale
    fgpr_std_predictive_norm: Optional[np.ndarray] = None
    fgpr_cov_predictive_norm: Optional[np.ndarray] = None
    fgpr_predictive_variance_norm: Optional[float] = None
    fgpr_predictive_variance_real: Optional[float] = None
    # Structured C_btw params (when fgpr_structured_btw=True)
    fgpr_structured_btw_params: Optional[object] = None
    # FGPR iteration convergence history
    fgpr_weight_history: Optional[List[np.ndarray]] = None
    fgpr_curve_history: Optional[List[np.ndarray]] = None
    fgpr_weight_converged: bool = False
    fgpr_max_weight_delta: float = 0.0
    fgpr_n_weight_iters: int = 0

    # Optional Student-t outputs
    student_t_mean: Optional[np.ndarray] = None
    student_t_std: Optional[np.ndarray] = None
    student_t_mean_norm: Optional[np.ndarray] = None
    student_t_std_norm: Optional[np.ndarray] = None
    student_t_cov: Optional[np.ndarray] = None
    student_t_cov_norm: Optional[np.ndarray] = None
    student_t_weights: Optional[np.ndarray] = None
    student_t_sigma_btw_squared: Optional[float] = None
    student_t_nu: Optional[float] = None
    student_t_within_variance_norm: Optional[float] = None
    student_t_within_variance_real: Optional[float] = None
    # Student-t predictive: C_pred = C_agg + sigma_btw^2 * I
    student_t_std_predictive: Optional[np.ndarray] = None
    student_t_std_predictive_norm: Optional[np.ndarray] = None
    student_t_cov_predictive_norm: Optional[np.ndarray] = None
    student_t_predictive_variance_norm: Optional[float] = None
    student_t_predictive_variance_real: Optional[float] = None
    # Student-t iteration convergence history
    student_t_weight_history: Optional[List[np.ndarray]] = None
    student_t_curve_history: Optional[List[np.ndarray]] = None
    student_t_energy_history: Optional[List[np.ndarray]] = None
    student_t_sigma_btw_history: Optional[List[float]] = None
    student_t_nu_history: Optional[List[float]] = None
    student_t_converged: bool = False
    student_t_max_weight_delta: float = 0.0
    student_t_n_iters: int = 0
    student_t_energies: Optional[np.ndarray] = None
    student_t_weights_raw: Optional[np.ndarray] = None


class SummaryGPROrchestrator:
    """
    Orchestrator for Summary GPR processing.
    
    Example
    -------
    >>> cfg = SummaryGPRConfig(input_directory=Path("output/"))
    >>> hp = SummaryGPRHyperParams()
    >>> orchestrator = SummaryGPROrchestrator(cfg, hp, verbose=True)
    >>> results = orchestrator.process_all()
    >>> for group_key, result in results.items():
    ...     print(f"{group_key}: {result.n_curves} curves")
    """
    
    def __init__(
        self,
        config: SummaryGPRConfig,
        hyperparams: SummaryGPRHyperParams,
        verbose: bool = True,
    ):
        """
        Initialize the orchestrator.
        
        Parameters
        ----------
        config : SummaryGPRConfig
            Configuration for aggregation.
        hyperparams : SummaryGPRHyperParams
            Algorithm hyperparameters.
        verbose : bool
            Print progress messages.
        """
        self.cfg = config
        self.hp = hyperparams
        self.verbose = verbose

        # Sub-orchestrators
        self._iterative = IterativeGPROrchestrator(config, hyperparams, verbose)
        self._operator = OperatorFusionOrchestrator(hyperparams, verbose)
        self._fgpr = FGPROrchestrator(config, hyperparams, verbose)
        self._student_t = StudentTOrchestrator(config, hyperparams, verbose)
        self._comparison = ComparisonOrchestrator(config, hyperparams, verbose)
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def _interpolate_to_common_grid(
        self,
        gprs: List[IndividualGPRData],
        n_points: int,
        x_target: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate all GPRs to a common X grid.
        
        Parameters
        ----------
        gprs : List[IndividualGPRData]
            Individual GPR data objects.
        n_points : int
            Number of interpolation points.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - x_common: Common X grid (transformed)
            - y_interp: Interpolated Y values (n_models, n_points)
            - S_interp: Interpolated std values (n_models, n_points)
        """
        # Find common X range (intersection of all)
        if x_target is not None:
            x_common = np.asarray(x_target)
        else:
            # Fallback: use data-range intersection.
            # NOTE: all curves should already share the same pip2 shared grid.
            import warnings
            warnings.warn(
                "No x_target provided to _interpolate_to_common_grid. "
                "Falling back to data-range intersection grid. "
                "This bypasses the JSON shared_grid settings. "
                "Ensure pip2 shared_grid is enabled and covariance data is available.",
                stacklevel=2,
            )
            x_min = max(gpr.x_pred_transformed.min() for gpr in gprs)
            x_max = min(gpr.x_pred_transformed.max() for gpr in gprs)
            x_common = np.linspace(x_min, x_max, n_points)
        
        # Interpolate each curve
        y_interp_list = []
        S_interp_list = []
        
        for gpr in gprs:
            # Sort by x for interpolation
            sort_idx = np.argsort(gpr.x_pred_transformed)
            x_sorted = gpr.x_pred_transformed[sort_idx]
            y_sorted = gpr.y_pred[sort_idx]
            S_sorted = gpr.y_std[sort_idx]
            
            # Create interpolators
            f_y = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')  # type: ignore[arg-type]
            f_S = interp1d(x_sorted, S_sorted, kind='linear', fill_value='extrapolate')  # type: ignore[arg-type]
            
            y_interp_list.append(f_y(x_common))
            S_interp_list.append(np.abs(f_S(x_common)))  # Ensure positive
        
        y_interp = np.array(y_interp_list)
        S_interp = np.array(S_interp_list)
        
        return x_common, y_interp, S_interp
    
    def process_group(
        self,
        group_key: str,
        gprs: List[IndividualGPRData],
        export_results: bool = True,
        plot_results: bool = True,
    ) -> Optional[SummaryGPRResult]:
        """
        Process one group of GPRs.
        
        Parameters
        ----------
        group_key : str
            Group identifier.
        gprs : List[IndividualGPRData]
            GPR data for this group.
        export_results : bool
            Whether to save CSV results.
        plot_results : bool
            Whether to generate plots.
            
        Returns
        -------
        SummaryGPRResult or None
            Aggregation result, or None if processing failed.
        """
        n_curves = len(gprs)
        self._log(f"\nProcessing group '{group_key}' ({n_curves} curves)")
        
        if n_curves == 0:
            self._log("  No curves in group, skipping")
            return None
        
        sample_ids = [gpr.sample_id for gpr in gprs]
        index_ids = [gpr.index_id for gpr in gprs]
        
        # ----- General outlier filtering: exclude curves with near-zero -----
        # scale factors.  Applies to ALL aggregation methods (iterative,
        # operator fusion, FGPR) so that outlier curves with |factor| << median
        # don't corrupt simple averages or sigma_btw estimation.
        ratio_thresh = self.hp.fgpr_min_scale_factor_ratio
        if ratio_thresh > 0 and n_curves > 1:
            abs_factors = np.array([
                abs(g.y_scaling.params.get("factor", 1.0)) for g in gprs
            ])
            median_factor = float(np.median(abs_factors))
            min_factor = ratio_thresh * median_factor
            keep_mask = abs_factors >= min_factor
            n_excluded = int((~keep_mask).sum())

            if n_excluded > 0:
                self._log(
                    f"  Excluding {n_excluded}/{n_curves} outlier curves "
                    f"with |factor| < {min_factor:.4g} "
                    f"(threshold = {ratio_thresh} \u00d7 median {median_factor:.4g})"
                )
                for idx in np.where(~keep_mask)[0]:
                    self._log(
                        f"    - [{idx}] {gprs[idx].sample_id}  "
                        f"|factor|={abs_factors[idx]:.4g}"
                    )
                keep_idx = np.where(keep_mask)[0].tolist()
                gprs = [gprs[i] for i in keep_idx]
                n_curves = len(gprs)
                sample_ids = [gpr.sample_id for gpr in gprs]
                index_ids = [gpr.index_id for gpr in gprs]
                self._log(f"  Continuing with {n_curves} curves")

                if n_curves == 0:
                    self._log("  No curves left after filtering, skipping")
                    return None

        # When FGPR is the aggregation method, skip the iterative path entirely
        fgpr_only = self.cfg.enable_fgpr and not self.cfg.enable_operator_fusion

        # Single curve case
        if n_curves == 1:
            gpr = gprs[0]
            result = SummaryGPRResult(
                group_key=group_key,
                x_pred_transformed=gpr.x_pred_transformed,
                x_pred_original=gpr.x_pred_original,
                y_mean=gpr.y_pred,
                y_mean_norm=gpr.y_scaling.transform(gpr.y_pred),
                y_std_real=gpr.y_std,
                y_std_norm=gpr.y_std,
                weights=np.array([1.0]),
                weight_history=[],
                curve_history=[gpr.y_pred],
                n_curves=1,
                sample_ids=sample_ids,
                y_scaling=gpr.y_scaling,
                x_scaling=gpr.x_scaling,
            )
            
            if export_results and not fgpr_only:
                output_dir = self.cfg.output_directory
                assert output_dir is not None
                self._iterative.export(result, [gprs[0].index_id],
                                       output_dir / 'iterative')
            if plot_results and not fgpr_only:
                output_dir = self.cfg.output_directory
                assert output_dir is not None
                iterative_dir = output_dir / 'iterative'
                self._iterative.plot(result, gprs, iterative_dir)
            
            return result
        
        # Check covariance grid for operator fusion, FGPR, and/or Student-t
        need_covariance = (self.cfg.enable_operator_fusion
                           or self.cfg.enable_fgpr
                           or self.cfg.enable_student_t)
        if not need_covariance:
            cov_enabled, operator_grid = False, None
            self._log("  Operator fusion, FGPR, and Student-t all disabled by config")
        else:
            cov_enabled, operator_grid = self._operator.check_covariance_grid(gprs)

        # Interpolate to common grid; align to operator grid if available
        x_common, y_interp, S_interp = self._interpolate_to_common_grid(
            gprs,
            self.hp.num_interp_points,
            x_target=operator_grid,
        )
        
        y_scalings = [gpr.y_scaling for gpr in gprs]
        
        # --- Method 1: Iterative weighted-sum (skip when FGPR-only) ---
        core_result = None
        if not fgpr_only:
            core_result = self._iterative.run(x_common, y_interp, S_interp, y_scalings)

        # --- Methods 2, 3, & 4: Operator Fusion, FGPR, Student-t (share data prep) ---
        operator_result: Optional[OperatorFusionResult] = None
        fgpr_result: Optional[FGPRResult] = None
        student_t_result: Optional[StudentTResult] = None
        fgpr_gprs = gprs  # may be narrowed by outlier filtering below

        # Prepare covariance data (shared by both methods)
        y_norm_list = None
        cov_norm_list = None
        if cov_enabled and operator_grid is not None:
            try:
                y_norm_list, cov_norm_list = self._operator.prepare_data(gprs, operator_grid)
            except ValueError as e:
                self._log(f"  {e}")
                cov_enabled = False

        # Method 2: Operator Fusion (only if enabled)
        if cov_enabled and self.cfg.enable_operator_fusion and y_norm_list is not None:
            operator_result = self._operator.run(y_norm_list, cov_norm_list, y_scalings)

        # Method 3: FGPR (independent of operator fusion)
        if cov_enabled and self.cfg.enable_fgpr and y_norm_list is not None:
            # Outlier filtering already applied upstream (general filtering).
            # Just alias for FGPR variables.
            fgpr_gprs = gprs
            fgpr_y_norm = y_norm_list
            fgpr_cov_norm = cov_norm_list
            fgpr_scalings = y_scalings

            # Determine whether to run in observation scale or normalised scale
            use_observation_scale = not self.cfg.normalization_summary
            if use_observation_scale:
                # Observation-scale FGPR: use real-scale predictions + real covariance
                # with identity scale factors, so FGPR operates entirely in real units.
                y_real_list = [gpr.y_pred for gpr in fgpr_gprs]
                cov_real_list = [gpr.covariance_real for gpr in fgpr_gprs]
                identity_scalings = [ScalingInfo.identity() for _ in fgpr_gprs]
                self._log(f"  FGPR: using observation-scale data ({len(fgpr_gprs)} curves, identity scalings)")
                fgpr_result = self._fgpr.run(y_real_list, cov_real_list, identity_scalings, t_grid=x_common)
            else:
                self._log(f"  FGPR: using normalised-scale data ({len(fgpr_gprs)} curves)")
                fgpr_result = self._fgpr.run(fgpr_y_norm, fgpr_cov_norm, fgpr_scalings, t_grid=x_common)

        # Method 4: Student-t robust aggregation (independent of FGPR / operator fusion)
        if cov_enabled and self.cfg.enable_student_t and y_norm_list is not None:
            st_y_norm = y_norm_list
            st_cov_norm = cov_norm_list
            st_scalings = y_scalings
            self._log(f"  Student-t: using normalised-scale data ({len(gprs)} curves, nu={self.hp.student_t_nu})")
            student_t_result = self._student_t.run(st_y_norm, st_cov_norm, st_scalings)
        
        # Convert x to original space
        x_scaling = gprs[0].x_scaling
        x_original = x_scaling.inverse_transform(x_common)
        
        # When FGPR-only and the iterative method was skipped, use the
        # covariance grid as x_common (it's the 500-point shared grid).
        if fgpr_only and operator_grid is not None:
            x_common = operator_grid
            x_original = x_scaling.inverse_transform(x_common)

        # Build unified result
        result = self._build_result(
            group_key, x_common, x_original, x_scaling,
            core_result, operator_result, fgpr_result, student_t_result,
            n_curves, sample_ids,
        )
        
        # Export
        if export_results:
            output_dir = self.cfg.output_directory
            assert output_dir is not None
            if not fgpr_only:
                self._iterative.export(result, index_ids, output_dir / 'iterative', gprs=gprs)
            if operator_result is not None:
                self._operator.export(result, index_ids, output_dir / 'operator_fusion')
            if fgpr_result is not None:
                fgpr_index_ids = [g.index_id for g in fgpr_gprs]
                self._fgpr.export(result, fgpr_index_ids, output_dir / 'fgpr', gprs=fgpr_gprs)
            if student_t_result is not None:
                self._student_t.export(result, index_ids, output_dir / 'student_t', gprs=gprs)
        
        # Plot
        if plot_results:
            output_dir = self.cfg.output_directory
            assert output_dir is not None
            if not fgpr_only:
                self._iterative.plot(result, gprs, output_dir / 'iterative')
            if operator_result is not None:
                self._operator.plot(result, output_dir / 'operator_fusion')
            if fgpr_result is not None:
                self._fgpr.plot(result, output_dir / 'fgpr', gprs=fgpr_gprs)
            if student_t_result is not None:
                self._student_t.plot(result, output_dir / 'student_t', gprs=gprs)
            if not fgpr_only:
                self._comparison.plot(result, gprs, output_dir)       # root
        
        return result
    
    def _build_result(
        self,
        group_key: str,
        x_common: np.ndarray,
        x_original: np.ndarray,
        x_scaling: ScalingInfo,
        core_result,
        operator_result: Optional[OperatorFusionResult],
        fgpr_result: Optional[FGPRResult],
        student_t_result: Optional[StudentTResult],
        n_curves: int,
        sample_ids: List[str],
    ) -> SummaryGPRResult:
        """Assemble a SummaryGPRResult from the four method outputs."""
        # When iterative was skipped (FGPR-only), use FGPR data for the
        # base fields so comparisons / downstream code still works.
        if core_result is not None:
            y_mean = core_result.y_mean
            y_mean_norm = core_result.y_mean_norm
            y_std_real = core_result.y_std_real
            y_std_norm = core_result.y_std_norm
            weights = core_result.weights
            weight_history = core_result.weight_history
            curve_history = core_result.curve_history
            y_scaling = core_result.y_scaling
        elif fgpr_result is not None:
            y_mean = fgpr_result.y_mean_real
            y_mean_norm = fgpr_result.y_mean_norm
            y_std_real = fgpr_result.y_std_real
            y_std_norm = fgpr_result.y_std_norm
            weights = fgpr_result.weights
            weight_history = []
            curve_history = [fgpr_result.y_mean_real]
            y_scaling = ScalingInfo.identity()
        else:
            raise ValueError("Both core_result and fgpr_result are None")

        return SummaryGPRResult(
            group_key=group_key,
            x_pred_transformed=x_common,
            x_pred_original=x_original,
            y_mean=y_mean,
            y_mean_norm=y_mean_norm,
            y_std_real=y_std_real,
            y_std_norm=y_std_norm,
            weights=weights,
            weight_history=weight_history,
            curve_history=curve_history,
            n_curves=n_curves,
            sample_ids=sample_ids,
            y_scaling=y_scaling,
            x_scaling=x_scaling,
            operator_mean=operator_result.y_mean_real if operator_result else None,
            operator_std=operator_result.y_std_real if operator_result else None,
            operator_mean_norm=operator_result.y_mean_norm if operator_result else None,
            operator_cov=operator_result.y_cov_real if operator_result else None,
            operator_cov_norm=operator_result.y_cov_norm if operator_result else None,
            operator_weights=operator_result.weights if operator_result else None,
            operator_iterations=operator_result.iterations if operator_result else None,
            operator_weight_history=operator_result.weight_history if operator_result else None,
            operator_between_variance=operator_result.between_variance_used if operator_result else None,
            operator_within_variance_norm=float(np.mean(np.diag(operator_result.y_cov_norm))) if operator_result else None,
            operator_within_variance_real=float(np.mean(np.diag(operator_result.y_cov_real))) if operator_result else None,
            fgpr_mean=fgpr_result.y_mean_real if fgpr_result else None,
            fgpr_std=fgpr_result.y_std_real if fgpr_result else None,
            fgpr_mean_norm=fgpr_result.y_mean_norm if fgpr_result else None,
            fgpr_std_norm=fgpr_result.y_std_norm if fgpr_result else None,
            fgpr_cov=fgpr_result.y_cov_real if fgpr_result else None,
            fgpr_cov_norm=fgpr_result.y_cov_norm if fgpr_result else None,
            fgpr_weights=fgpr_result.weights if fgpr_result else None,
            fgpr_sigma_btw_squared=fgpr_result.sigma_btw_squared if fgpr_result else None,
            fgpr_nll=fgpr_result.nll_optimized if fgpr_result else None,
            fgpr_within_variance_norm=float(np.mean(np.diag(fgpr_result.y_cov_norm))) if fgpr_result else None,
            fgpr_within_variance_real=float(np.mean(np.diag(fgpr_result.y_cov_real))) if fgpr_result else None,
            fgpr_std_predictive=fgpr_result.y_std_predictive_real if fgpr_result else None,
            fgpr_std_predictive_norm=fgpr_result.y_std_predictive_norm if fgpr_result else None,
            fgpr_cov_predictive_norm=fgpr_result.y_cov_predictive_norm if fgpr_result else None,
            fgpr_predictive_variance_norm=float(np.mean(np.diag(fgpr_result.y_cov_predictive_norm))) if fgpr_result else None,
            fgpr_predictive_variance_real=float(np.mean(np.diag(fgpr_result.y_cov_predictive_real))) if fgpr_result else None,
            fgpr_structured_btw_params=(
                fgpr_result.optimizer_result.structured_params
                if fgpr_result and hasattr(getattr(fgpr_result, 'optimizer_result', None), 'structured_params')
                else None
            ),
            fgpr_weight_history=(
                fgpr_result.weight_history
                if fgpr_result and getattr(fgpr_result, 'weight_history', None)
                else None
            ),
            fgpr_curve_history=(
                fgpr_result.curve_history
                if fgpr_result and getattr(fgpr_result, 'curve_history', None)
                else None
            ),
            fgpr_weight_converged=(
                fgpr_result.weight_converged
                if fgpr_result else False
            ),
            fgpr_max_weight_delta=(
                fgpr_result.max_weight_delta
                if fgpr_result else 0.0
            ),
            fgpr_n_weight_iters=(
                fgpr_result.n_weight_iters
                if fgpr_result else 0
            ),
            # --- Student-t fields ---
            student_t_mean=student_t_result.y_mean_real if student_t_result else None,
            student_t_std=student_t_result.y_std_real if student_t_result else None,
            student_t_mean_norm=student_t_result.y_mean_norm if student_t_result else None,
            student_t_std_norm=student_t_result.y_std_norm if student_t_result else None,
            student_t_cov=student_t_result.y_cov_real if student_t_result else None,
            student_t_cov_norm=student_t_result.y_cov_norm if student_t_result else None,
            student_t_weights=student_t_result.weights if student_t_result else None,
            student_t_sigma_btw_squared=student_t_result.sigma_btw_squared if student_t_result else None,
            student_t_nu=student_t_result.nu if student_t_result else None,
            student_t_within_variance_norm=float(np.mean(np.diag(student_t_result.y_cov_norm))) if student_t_result else None,
            student_t_within_variance_real=float(np.mean(np.diag(student_t_result.y_cov_real))) if student_t_result else None,
            student_t_std_predictive=student_t_result.y_std_predictive_real if student_t_result else None,
            student_t_std_predictive_norm=student_t_result.y_std_predictive_norm if student_t_result else None,
            student_t_cov_predictive_norm=student_t_result.y_cov_predictive_norm if student_t_result else None,
            student_t_predictive_variance_norm=float(np.mean(np.diag(student_t_result.y_cov_predictive_norm))) if student_t_result else None,
            student_t_predictive_variance_real=float(np.mean(np.diag(student_t_result.y_cov_predictive_real))) if student_t_result else None,
            student_t_weight_history=student_t_result.weight_history if student_t_result else None,
            student_t_curve_history=student_t_result.curve_history if student_t_result else None,
            student_t_energy_history=student_t_result.energy_history if student_t_result else None,
            student_t_sigma_btw_history=student_t_result.sigma_btw_history if student_t_result else None,
            student_t_nu_history=student_t_result.nu_history if student_t_result else None,
            student_t_converged=student_t_result.converged if student_t_result else False,
            student_t_max_weight_delta=student_t_result.max_weight_delta if student_t_result else 0.0,
            student_t_n_iters=student_t_result.iterations if student_t_result else 0,
            student_t_energies=student_t_result.energies if student_t_result else None,
            student_t_weights_raw=student_t_result.weights_raw if student_t_result else None,
        )
    
    def process_all(
        self,
        export_results: bool = True,
        plot_results: bool = True,
    ) -> Dict[str, SummaryGPRResult]:
        """
        Process all groups in the input directory.
        
        Parameters
        ----------
        export_results : bool
            Whether to save CSV results.
        plot_results : bool
            Whether to generate plots.
            
        Returns
        -------
        Dict[str, SummaryGPRResult]
            Results keyed by group key.
        """
        self._log(f"Loading individual GPRs from: {self.cfg.input_directory}")
        
        # Load all GPRs
        all_gprs = load_all_individual_gprs(
            self.cfg.input_directory,
            self.cfg.file_pattern,
            verbose=self.verbose,
        )
        
        if not all_gprs:
            self._log("No GPR files found!")
            return {}
        
        self._log(f"Loaded {len(all_gprs)} individual GPRs")
        
        # Group by key
        grouped = group_gprs_by_key(all_gprs)
        self._log(f"Found {len(grouped)} groups: {list(grouped.keys())}")
        
        # Filter groups if specified
        if self.cfg.process_groups:
            grouped = {k: v for k, v in grouped.items() if k in self.cfg.process_groups}
            self._log(f"Filtering to {len(grouped)} groups")
        
        # Process each group
        results = {}
        for group_key, gprs in sorted(grouped.items()):
            try:
                result = self.process_group(
                    group_key, gprs,
                    export_results=export_results,
                    plot_results=plot_results,
                )
                if result is not None:
                    results[group_key] = result
            except Exception as e:
                self._log(f"Error processing group '{group_key}': {e}")
                import traceback
                traceback.print_exc()
        
        self._log(f"\nSummary GPR complete: {len(results)} groups processed")
        
        return results
