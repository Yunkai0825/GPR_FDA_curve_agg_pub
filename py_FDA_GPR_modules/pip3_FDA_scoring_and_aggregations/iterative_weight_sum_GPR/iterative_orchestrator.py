# iterative_weight_sum_GPR/iterative_orchestrator.py
"""
Orchestrator for Iterative Weighted-Sum GPR aggregation.

Owns:
- Running core iterative algorithm (compute_summary_gpr)
- Exporting iterative weights + history
- Plotting summary GPR, curve iterations, weight distribution/convergence

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .iterative_gpr_core import compute_summary_gpr, SummaryGPRCoreResult
from .iterative_export_helpers import (
    export_iterative_weights_csv,
    export_iterative_history_csv,
    export_iterative_sigma_calibration_csv,
    export_iterative_summary_csv,
)
from .iterative_plot_helpers import (
    plot_summary_gpr_from_csv,
    plot_weight_distribution_from_csv,
    plot_weight_convergence_from_csv,
    plot_summary_curve_iterations_from_csv,
    plot_iterative_sigma_calibration,
)

if TYPE_CHECKING:
    from ..summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
    from ..summary_gpr_loader import IndividualGPRData
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


class IterativeGPROrchestrator:
    """
    Orchestrates the iterative weighted-sum GPR aggregation method.

    Usage from the main orchestrator::

        orch = IterativeGPROrchestrator(cfg, hp, verbose=True)
        core = orch.run(x_common, y_interp, S_interp, y_scalings)
        # ... build SummaryGPRResult ...
        orch.export(result, index_ids, output_dir)
        orch.plot(result, gprs, output_dir)
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
        x_common: np.ndarray,
        y_interp: np.ndarray,
        S_interp: np.ndarray,
        y_scalings: List[ScalingInfo],
    ) -> SummaryGPRCoreResult:
        """
        Run the iterative weighted-sum GPR aggregation.

        Parameters
        ----------
        x_common : shape (N,)
            Common X grid (transformed space).
        y_interp : shape (M, N)
            Interpolated Y means for each model.
        S_interp : shape (M, N)
            Interpolated std for each model.
        y_scalings : list of ScalingInfo
            Y-axis scaling info per model.

        Returns
        -------
        SummaryGPRCoreResult
        """
        return compute_summary_gpr(
            y_array=y_interp,
            S_array=S_interp,
            x_pred=x_common,
            y_scalings=y_scalings,
            weight_mode=self.cfg.weight_mode,
            weight_scope=self.cfg.weight_scope,
            include_within=self.cfg.include_within_variance,
            include_between=self.cfg.include_between_variance,
            variance_scale=self.cfg.variance_aggregation_scale,
            normalization_summary=self.cfg.normalization_summary,
            epsilon=self.hp.epsilon,
            convergence_tol=self.hp.convergence_tol,
            max_iterations=self.hp.max_iterations,
        )

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
        """Export iterative summary curve, weights, iteration history, and sigma calibration CSVs."""
        log_fn = self._log
        export_iterative_summary_csv(
            result, output_dir,
            confidence_level=self.hp.confidence_level,
            log_fn=log_fn,
        )
        export_iterative_weights_csv(result, index_ids, output_dir, log_fn=log_fn)
        export_iterative_history_csv(result, output_dir, log_fn=log_fn)
        export_iterative_sigma_calibration_csv(
            result, output_dir, gprs=gprs, log_fn=log_fn,
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(
        self,
        result: "SummaryGPRResult",
        gprs: List["IndividualGPRData"],
        output_dir: Path,
    ) -> None:
        """Generate iterative-method-specific plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        safe_key = result.group_key.replace('|', '_').replace('=', '_')

        # 1) Summary GPR plot (mean + CI + variance comparison) from CSV
        summary_csv = output_dir / f'Summary_GPR_{safe_key}.csv'
        individual_csvs = None
        if self.cfg.plot_individual_gprs and gprs:
            individual_csvs = [gpr.filepath for gpr in gprs if gpr.filepath]

        plot_summary_gpr_from_csv(
            summary_csv_path=summary_csv,
            output_directory=output_dir,
            individual_gpr_csvs=individual_csvs,
            group_key=result.group_key,
            confidence_level=self.hp.confidence_level,
            plot_individuals=self.cfg.plot_individual_gprs,
            individual_alpha=self.cfg.individual_curve_alpha,
            min_time_cap=self.cfg.min_time_cap,
            max_time_cap=self.cfg.max_time_cap,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            verbose=self.verbose,
        )

        # 2) Weight distribution (skip for single-curve groups)
        if result.n_curves > 1:
            weights_csv = output_dir / f'Converged_Weights_{safe_key}.csv'
            plot_weight_distribution_from_csv(
                csv_path=weights_csv,
                output_directory=output_dir,
                group_key=result.group_key,
                verbose=self.verbose,
            )

        # 3) Weight convergence (only if multi-iteration)
        if result.n_curves > 1 and result.weight_history and len(result.weight_history) > 1:
            weight_csv = output_dir / f'Weight_History_{safe_key}.csv'
            plot_weight_convergence_from_csv(
                csv_path=weight_csv,
                output_directory=output_dir,
                group_key=result.group_key,
                verbose=self.verbose,
            )

        # 4) Curve iteration history
        if result.n_curves > 1 and result.curve_history and len(result.curve_history) > 1:
            curve_csv = output_dir / f'Curve_History_{safe_key}.csv'
            y_transform = getattr(self.cfg, 'y_transform_method', '')
            if self.cfg.normalization_summary and y_transform:
                iter_y_label = f"{self.cfg.y_axis_label} [normalized: {y_transform}]"
            elif self.cfg.normalization_summary:
                iter_y_label = f"{self.cfg.y_axis_label} [normalized]"
            else:
                iter_y_label = self.cfg.y_axis_label
            plot_summary_curve_iterations_from_csv(
                csv_path=curve_csv,
                output_directory=output_dir,
                group_key=result.group_key,
                min_time_cap=self.cfg.min_time_cap,
                max_time_cap=self.cfg.max_time_cap,
                x_axis_label=self.cfg.x_axis_label,
                y_axis_label=iter_y_label,
                x_is_log=True,
                verbose=self.verbose,
            )

        # 5) Sigma calibration (real + normalized)
        if result.n_curves > 1:
            plot_iterative_sigma_calibration(
                result, output_dir, gprs=gprs,
                x_axis_label=self.cfg.x_axis_label,
                y_axis_label=self.cfg.y_axis_label,
                verbose=self.verbose,
            )
