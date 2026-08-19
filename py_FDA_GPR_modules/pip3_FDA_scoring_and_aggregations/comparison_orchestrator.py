# pip3_FDA_scoring_and_aggregations/comparison_orchestrator.py
"""
Orchestrator for cross-method comparison plotting.

Owns:
- Baseline vs Operator Fusion vs FGPR comparison plot (real scale)
- Normalized-space comparison plot
- Sigma calibration deviation plots (real + normalized)

These plots span multiple aggregation methods and therefore live at the
parent level rather than inside any single method folder.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from scipy.stats import norm as sp_norm

from .summary_gpr_plotting import (
    plot_summary_comparison,
    plot_summary_comparison_normalized,
)

if TYPE_CHECKING:
    from .summary_gpr_config import SummaryGPRConfig, SummaryGPRHyperParams
    from .summary_gpr_loader import IndividualGPRData
    from .pip3_summary_gpr_orchestrator import SummaryGPRResult


class ComparisonOrchestrator:
    """
    Orchestrates cross-method comparison plots and sigma-calibration diagnostics.

    Usage from the main orchestrator::

        comp = ComparisonOrchestrator(cfg, hp, verbose=True)
        comp.plot(result, gprs, output_dir)
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
    # Public entry point
    # ------------------------------------------------------------------
    def plot(
        self,
        result: "SummaryGPRResult",
        gprs: List["IndividualGPRData"],
        output_dir: Path,
    ) -> None:
        """
        Generate all cross-method comparison plots.

        Requires operator-fusion results to be present on *result*;
        silently returns if they are not.
        """
        if result.operator_mean is None or result.operator_std is None:
            return

        z = sp_norm.ppf(0.5 + self.hp.confidence_level / 2)

        # --- build common data ---
        individual_curves_real, individual_curves_norm = self._build_individual_curves(gprs)

        # --- real-scale comparison ---
        self._plot_comparison_real(result, z, individual_curves_real, output_dir)

        # --- normalized-scale comparison ---
        self._plot_comparison_normalized(result, z, individual_curves_norm, output_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_individual_curves(self, gprs: List["IndividualGPRData"]):
        """Return (curves_real, curves_norm) or (None, None)."""
        if not self.cfg.plot_individual_gprs or not gprs:
            return None, None

        curves_real = []
        curves_norm = []
        for gpr in gprs:
            x_orig = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
            curves_real.append((x_orig, gpr.y_pred))
            curves_norm.append((x_orig, gpr.y_pred_normalized))
        return curves_real, curves_norm

    # ------------------------------------------------------------------
    def _plot_comparison_real(self, result, z, individual_curves_real, output_dir):
        """Baseline vs operator vs FGPR comparison in real scale."""
        base_upper = result.y_mean + z * (result.y_std_real if result.y_std_real is not None else 0)
        base_lower = result.y_mean - z * (result.y_std_real if result.y_std_real is not None else 0)

        op_upper = result.operator_mean + z * result.operator_std
        op_lower = result.operator_mean - z * result.operator_std

        fgpr_mean_plot = fgpr_lower_plot = fgpr_upper_plot = None
        if result.fgpr_mean is not None and result.fgpr_std_predictive is not None:
            fgpr_mean_plot = result.fgpr_mean
            fgpr_upper_plot = result.fgpr_mean + z * result.fgpr_std_predictive
            fgpr_lower_plot = result.fgpr_mean - z * result.fgpr_std_predictive
        elif result.fgpr_mean is not None and result.fgpr_std is not None:
            fgpr_mean_plot = result.fgpr_mean
            fgpr_upper_plot = result.fgpr_mean + z * result.fgpr_std
            fgpr_lower_plot = result.fgpr_mean - z * result.fgpr_std

        plot_summary_comparison(
            x_real=result.x_pred_original,
            baseline_mean=result.y_mean,
            baseline_lower=base_lower,
            baseline_upper=base_upper,
            operator_mean=result.operator_mean,
            operator_lower=op_lower,
            operator_upper=op_upper,
            output_directory=output_dir,
            group_key=result.group_key,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            min_time_cap=self.cfg.min_time_cap,
            max_time_cap=self.cfg.max_time_cap,
            individual_curves=individual_curves_real,
            individual_alpha=self.cfg.individual_curve_alpha,
            fgpr_mean=fgpr_mean_plot,
            fgpr_lower=fgpr_lower_plot,
            fgpr_upper=fgpr_upper_plot,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    def _plot_comparison_normalized(self, result, z, individual_curves_norm, output_dir):
        """Normalized-space comparison plot with individual GPRs."""
        base_std_norm = result.y_std_norm if result.y_std_norm is not None else np.zeros_like(result.y_mean_norm)
        base_upper_norm = result.y_mean_norm + z * base_std_norm
        base_lower_norm = result.y_mean_norm - z * base_std_norm

        op_mean_norm_plot = result.operator_mean_norm
        op_std_norm = (
            np.sqrt(np.clip(np.diag(result.operator_cov_norm), 0.0, None))
            if result.operator_cov_norm is not None
            else None
        )
        op_upper_norm = op_mean_norm_plot + z * op_std_norm if op_std_norm is not None else None
        op_lower_norm = op_mean_norm_plot - z * op_std_norm if op_std_norm is not None else None

        fgpr_mean_norm_plot = result.fgpr_mean_norm
        fgpr_std_norm = (
            np.sqrt(np.clip(np.diag(result.fgpr_cov_predictive_norm), 0.0, None))
            if result.fgpr_cov_predictive_norm is not None
            else None
        )
        fgpr_upper_norm = fgpr_mean_norm_plot + z * fgpr_std_norm if fgpr_std_norm is not None else None
        fgpr_lower_norm = fgpr_mean_norm_plot - z * fgpr_std_norm if fgpr_std_norm is not None else None

        plot_summary_comparison_normalized(
            x_real=result.x_pred_original,
            baseline_mean_norm=result.y_mean_norm,
            baseline_lower_norm=base_lower_norm,
            baseline_upper_norm=base_upper_norm,
            operator_mean_norm=op_mean_norm_plot,
            operator_lower_norm=op_lower_norm,
            operator_upper_norm=op_upper_norm,
            fgpr_mean_norm=fgpr_mean_norm_plot,
            fgpr_lower_norm=fgpr_lower_norm,
            fgpr_upper_norm=fgpr_upper_norm,
            output_directory=output_dir,
            group_key=result.group_key,
            x_axis_label=self.cfg.x_axis_label,
            y_axis_label=self.cfg.y_axis_label,
            min_time_cap=self.cfg.min_time_cap,
            max_time_cap=self.cfg.max_time_cap,
            individual_curves_norm=individual_curves_norm,
            individual_alpha=self.cfg.individual_curve_alpha,
            verbose=self.verbose,
        )
