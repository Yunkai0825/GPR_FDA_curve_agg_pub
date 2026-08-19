# operator_fusion_noweight/operator_fusion_orchestrator.py
"""
Orchestrator for Operator-Fusion GPR aggregation.

Owns:
- Preparing normalized means / covariance lists from GPR data
- Running core operator-fusion algorithm (compute_operator_fusion)
- Exporting operator weights + history
- Plotting operator weight convergence

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from scipy.interpolate import interp1d

from .operator_fusion_weight_helpers import (
    compute_operator_fusion,
    OperatorFusionResult,
)
from .operator_fusion_export_helpers import (
    export_operator_weights_csv,
    export_operator_history_csv,
    export_operator_curve_csv,
)
from .operator_fusion_plot_helpers import (
    plot_operator_weight_convergence,
)

if TYPE_CHECKING:
    from ..summary_gpr_config import SummaryGPRHyperParams
    from ..summary_gpr_loader import IndividualGPRData
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent))
from pip1_datapreprocessing import ScalingInfo  # type: ignore


class OperatorFusionOrchestrator:
    """
    Orchestrates the operator-fusion (no-weight, precision-space) aggregation method.

    Usage from the main orchestrator::

        orch = OperatorFusionOrchestrator(hp, verbose=True)
        enabled, grid = orch.check_covariance_grid(gprs)
        if enabled:
            y_norm, cov_norm = orch.prepare_data(gprs, grid)
            op_result = orch.run(y_norm, cov_norm, y_scalings)
            orch.export(result, index_ids, output_dir)
            orch.plot(result, output_dir)
    """

    def __init__(
        self,
        hyperparams: "SummaryGPRHyperParams",
        verbose: bool = True,
    ):
        self.hp = hyperparams
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Grid validation
    # ------------------------------------------------------------------
    def check_covariance_grid(
        self,
        gprs: List["IndividualGPRData"],
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check whether all GPRs share an aligned covariance grid.

        Returns
        -------
        (enabled, grid)
            enabled=True and grid=<shared grid> if all covariance matrices
            exist and grids agree; otherwise enabled=False, grid=None.
        """
        n_curves = len(gprs)
        if not all(gpr.covariance_matrix is not None for gpr in gprs):
            self._log("  Some curves lack covariance matrices; operator/FGPR disabled")
            return False, None

        grids = [gpr.covariance_grid for gpr in gprs if gpr.covariance_grid is not None]
        if len(grids) != n_curves:
            self._log("  Covariance grids missing on some curves; operator/FGPR disabled")
            return False, None

        base_grid = grids[0]
        if not all(np.allclose(base_grid, g, atol=1e-10) for g in grids[1:]):
            self._log("  Covariance grids mismatch; operator/FGPR disabled for this group")
            return False, None

        return True, base_grid

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def prepare_data(
        self,
        gprs: List["IndividualGPRData"],
        operator_grid: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Interpolate normalized means to the operator grid and collect
        covariance matrices.

        Returns
        -------
        (y_norm_list, cov_norm_list)
        """
        y_norm_list: List[np.ndarray] = []
        cov_norm_list: List[np.ndarray] = []

        for gpr in gprs:
            sort_idx = np.argsort(gpr.x_pred_transformed)
            x_sorted = gpr.x_pred_transformed[sort_idx]
            y_norm_sorted = gpr.y_scaling.transform(gpr.y_pred)[sort_idx]
            f_y_norm = interp1d(
                x_sorted,
                y_norm_sorted,
                kind='linear',
                fill_value='extrapolate',
            )
            y_norm_list.append(f_y_norm(operator_grid))

            assert gpr.covariance_matrix is not None
            if gpr.covariance_matrix.shape[0] != operator_grid.shape[0]:
                raise ValueError(
                    f"Covariance shape {gpr.covariance_matrix.shape} does not match "
                    f"operator grid length {operator_grid.shape[0]}"
                )
            cov_norm_list.append(gpr.covariance_matrix)

        return y_norm_list, cov_norm_list

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(
        self,
        y_norm_list: List[np.ndarray],
        cov_norm_list: List[np.ndarray],
        y_scalings: List[ScalingInfo],
    ) -> Optional[OperatorFusionResult]:
        """
        Run operator-fusion aggregation.

        Returns
        -------
        OperatorFusionResult or None on failure.
        """
        try:
            return compute_operator_fusion(
                y_norm_list=y_norm_list,
                cov_norm_list=cov_norm_list,
                y_scalings=y_scalings,
                epsilon=self.hp.epsilon,
                return_history=True,
            )
        except Exception as e:
            self._log(f"  Operator fusion failed: {e}")
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
    ) -> None:
        """Export operator curve CSV, weights, and history/diagnostics."""
        log_fn = self._log
        export_operator_curve_csv(
            result, output_dir,
            confidence_level=self.hp.confidence_level,
            log_fn=log_fn,
        )
        export_operator_weights_csv(result, index_ids, output_dir, log_fn=log_fn)
        export_operator_history_csv(result, output_dir, log_fn=log_fn)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def plot(
        self,
        result: "SummaryGPRResult",
        output_dir: Path,
    ) -> None:
        """Generate operator-fusion weight convergence plot."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_operator_weight_convergence(result, output_dir, verbose=self.verbose)
