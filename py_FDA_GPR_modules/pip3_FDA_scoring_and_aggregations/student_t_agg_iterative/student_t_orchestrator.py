# student_t_agg_iterative/student_t_orchestrator.py
"""
Orchestrator for Student-t robust curve aggregation.

Owns:
- Running core Student-t algorithm (compute_student_t_aggregation)
- Exporting weights, diagnostics, curve, covariance, iteration history
- Plotting curve, weights, convergence, covariance, sigma calibration

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .student_t_core import (
    compute_student_t_aggregation,
    StudentTResult,
)
from .student_t_export_helpers import (
    export_student_t_weights_csv,
    export_student_t_diagnostics,
    export_student_t_curve_csv,
    export_student_t_covariance_csv,
    export_student_t_iteration_history_csv,
    export_student_t_hyperparameter_history_csv,
    export_student_t_sigma_calibration_csv,
)
from .student_t_plot_helpers import (
    plot_student_t_curve,
    plot_student_t_weight_distribution,
    plot_student_t_covariance_heatmap,
    plot_student_t_diagonal_std,
    plot_student_t_sigma_calibration,
    plot_student_t_weight_convergence,
    plot_student_t_curve_iterations,
    plot_student_t_energy_convergence,
    plot_student_t_hyperparameter_history,
)

if TYPE_CHECKING:
    from ..summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
    from ..summary_gpr_loader import IndividualGPRData
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


class StudentTOrchestrator:
    """
    Orchestrates the Student-t robust curve aggregation method.

    Usage from the main orchestrator::

        orch = StudentTOrchestrator(config, hyperparams, verbose=True)
        st_result = orch.run(y_norm_list, cov_norm_list, y_scalings)
        orch.export(result, index_ids, output_dir)
        orch.plot(result, output_dir)
    """

    def __init__(
        self,
        config: "SummaryGPRConfig",
        hyperparams: "SummaryGPRHyperParams",
        verbose: bool = True,
    ):
        self.cfg = config
        self.hp = hyperparams
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        y_norm_list: List[np.ndarray],
        cov_norm_list: List[np.ndarray],
        y_scalings: List[ScalingInfo],
    ) -> Optional[StudentTResult]:
        """
        Run Student-t aggregation.

        Parameters
        ----------
        y_norm_list : list of shape-(N,) arrays
            Normalized means for each curve.
        cov_norm_list : list of shape-(N, N) arrays
            Covariance matrices for each curve.
        y_scalings : list of ScalingInfo
            Y-axis scaling info per curve.

        Returns
        -------
        StudentTResult or None on failure.
        """
        try:
            return compute_student_t_aggregation(
                y_norm_list=y_norm_list,
                cov_norm_list=cov_norm_list,
                y_scalings=y_scalings,
                nu=self.hp.student_t_nu,
                optimize_nu=self.hp.student_t_optimize_nu,
                nu_bounds=self.hp.student_t_nu_bounds,
                nu_lb_adaptive=self.hp.student_t_nu_lb_adaptive,
                max_iterations=self.hp.student_t_max_iterations,
                convergence_tol=self.hp.student_t_convergence_tol,
                epsilon=self.hp.epsilon,
                verbose=self.verbose,
            )
        except Exception as e:
            self._log(f"  Student-t aggregation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(
        self,
        result: "SummaryGPRResult",
        index_ids: List[int],
        output_dir: Path,
        gprs: Optional[List["IndividualGPRData"]] = None,
    ) -> None:
        """Export Student-t weights, diagnostics, curve, covariance, and iteration history CSVs."""
        log_fn = self._log
        sample_ids = [g.sample_id for g in gprs] if gprs else None
        export_student_t_weights_csv(result, index_ids, output_dir, log_fn=log_fn, sample_ids=sample_ids)
        export_student_t_diagnostics(result, output_dir, log_fn=log_fn)
        export_student_t_curve_csv(
            result, output_dir,
            confidence_level=self.hp.confidence_level,
            log_fn=log_fn,
        )
        export_student_t_covariance_csv(result, output_dir, log_fn=log_fn)
        export_student_t_sigma_calibration_csv(
            result, output_dir, gprs=gprs, log_fn=log_fn,
        )
        export_student_t_iteration_history_csv(result, output_dir, log_fn=log_fn)
        export_student_t_hyperparameter_history_csv(result, output_dir, log_fn=log_fn)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(
        self,
        result: "SummaryGPRResult",
        output_dir: Path,
        gprs: Optional[List["IndividualGPRData"]] = None,
    ) -> None:
        """Generate all Student-t plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_student_t_curve(
            result, output_dir, gprs=gprs,
            confidence_level=self.hp.confidence_level,
            individual_alpha=self.cfg.individual_curve_alpha,
            min_time_cap=self.cfg.min_time_cap,
            max_time_cap=self.cfg.max_time_cap,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            verbose=self.verbose,
        )
        plot_student_t_weight_distribution(result, output_dir, verbose=self.verbose)
        plot_student_t_covariance_heatmap(result, output_dir, verbose=self.verbose)
        plot_student_t_diagonal_std(result, output_dir, verbose=self.verbose)
        plot_student_t_sigma_calibration(
            result, output_dir, gprs=gprs,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            verbose=self.verbose,
        )
        plot_student_t_weight_convergence(result, output_dir, verbose=self.verbose)
        plot_student_t_curve_iterations(result, output_dir, verbose=self.verbose)
        plot_student_t_energy_convergence(result, output_dir, verbose=self.verbose)
        plot_student_t_hyperparameter_history(result, output_dir, verbose=self.verbose)
