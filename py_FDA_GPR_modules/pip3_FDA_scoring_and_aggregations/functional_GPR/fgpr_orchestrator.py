# functional_GPR/fgpr_orchestrator.py
"""
Orchestrator for Functional GPR (FGPR) aggregation.

Owns:
- Running core FGPR algorithm (compute_fgpr)
- Exporting FGPR weights + diagnostics
- Plotting FGPR weight distribution

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .fgpr_helpers import (
    compute_fgpr,
    compute_fgpr_structured,
    FGPRResult,
)
from .fgpr_export_helpers import (
    export_fgpr_weights_csv,
    export_fgpr_diagnostics,
    export_fgpr_curve_csv,
    export_fgpr_covariance_csv,
    export_fgpr_sigma_calibration_csv,
    export_fgpr_iteration_history_csv,
)
from .fgpr_plot_helpers import (
    plot_fgpr_weight_distribution,
    plot_fgpr_covariance_heatmap,
    plot_fgpr_diagonal_std,
    plot_fgpr_curve,
    plot_fgpr_sigma_calibration,
    plot_fgpr_weight_convergence,
    plot_fgpr_curve_iterations,
)

if TYPE_CHECKING:
    from ..summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
    from ..summary_gpr_loader import IndividualGPRData
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


class FGPROrchestrator:
    """
    Orchestrates the Functional GPR (FGPR) aggregation method.

    Usage from the main orchestrator::

        orch = FGPROrchestrator(hp, verbose=True)
        fgpr_result = orch.run(y_norm_list, cov_norm_list, y_scalings)
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
        t_grid: Optional[np.ndarray] = None,
    ) -> Optional[FGPRResult]:
        """
        Run FGPR aggregation.

        Parameters
        ----------
        y_norm_list : list of shape-(N,) arrays
            Normalized means for each model.
        cov_norm_list : list of shape-(N, N) arrays
            Covariance matrices for each model.
        y_scalings : list of ScalingInfo
            Y-axis scaling info per model.
        t_grid : (N,) array, optional
            Common x-grid in transformed space.  Required when
            ``hyperparams.fgpr_structured_btw`` is True.

        Returns
        -------
        FGPRResult or None on failure.
        """
        try:
            if self.hp.fgpr_structured_btw:
                if t_grid is None:
                    raise ValueError(
                        "t_grid is required for structured C_btw mode"
                    )
                from .fgpr_structured_btw import StructuredBtwConfig
                btw_cfg = StructuredBtwConfig()
                self._log("  FGPR: using structured C_btw model")
                return compute_fgpr_structured(
                    y_norm_list=y_norm_list,
                    cov_norm_list=cov_norm_list,
                    y_scalings=y_scalings,
                    t_grid=t_grid,
                    btw_cfg=btw_cfg,
                    epsilon=self.hp.epsilon,
                    verbose=self.verbose,
                )
            else:
                return compute_fgpr(
                    y_norm_list=y_norm_list,
                    cov_norm_list=cov_norm_list,
                    y_scalings=y_scalings,
                    epsilon=self.hp.epsilon,
                    verbose=self.verbose,
                )
        except Exception as e:
            self._log(f"  FGPR failed: {e}")
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
        """Export FGPR weights, diagnostics, aggregated curve, covariance, sigma calibration, and iteration history CSVs."""
        log_fn = self._log
        fgpr_sample_ids = [g.sample_id for g in gprs] if gprs else None
        export_fgpr_weights_csv(result, index_ids, output_dir, log_fn=log_fn, sample_ids=fgpr_sample_ids)
        export_fgpr_diagnostics(result, output_dir, log_fn=log_fn)
        export_fgpr_curve_csv(
            result, output_dir,
            confidence_level=self.hp.confidence_level,
            log_fn=log_fn,
        )
        export_fgpr_covariance_csv(result, output_dir, log_fn=log_fn)
        export_fgpr_sigma_calibration_csv(
            result, output_dir, gprs=gprs, log_fn=log_fn,
        )
        export_fgpr_iteration_history_csv(result, output_dir, log_fn=log_fn)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(
        self,
        result: "SummaryGPRResult",
        output_dir: Path,
        gprs: Optional[List["IndividualGPRData"]] = None,
    ) -> None:
        """Generate all FGPR plots: curve, weight distribution, covariance, diagonal std, sigma calibration."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_fgpr_curve(
            result, output_dir, gprs=gprs,
            confidence_level=self.hp.confidence_level,
            individual_alpha=self.cfg.individual_curve_alpha,
            min_time_cap=self.cfg.min_time_cap,
            max_time_cap=self.cfg.max_time_cap,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            verbose=self.verbose,
        )
        plot_fgpr_weight_distribution(result, output_dir, verbose=self.verbose)
        plot_fgpr_covariance_heatmap(result, output_dir, verbose=self.verbose)
        plot_fgpr_diagonal_std(result, output_dir, verbose=self.verbose)
        plot_fgpr_sigma_calibration(
            result, output_dir, gprs=gprs,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            verbose=self.verbose,
        )
        plot_fgpr_weight_convergence(result, output_dir, verbose=self.verbose)
        plot_fgpr_curve_iterations(result, output_dir, verbose=self.verbose)
