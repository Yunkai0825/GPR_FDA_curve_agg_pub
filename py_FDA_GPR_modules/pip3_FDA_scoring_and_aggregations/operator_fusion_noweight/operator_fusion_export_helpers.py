# operator_fusion_noweight/operator_fusion_export_helpers.py
"""
Export helpers for Operator Fusion aggregation.

Handles CSV/text export of:
- Operator fusion summary curve (mean + CI)
- Operator fusion weights
- Operator weight history (per iteration)
- Between-variance / within-variance diagnostics

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult


def export_operator_curve_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    confidence_level: float = 0.75,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export operator-fusion aggregation summary curve to CSV.

    Contains operator-fusion-method columns: x coordinates,
    mean, std, CI bounds in real scale, and variance diagnostics.
    """
    from scipy.stats import norm

    if result.operator_mean is None or result.operator_std is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    z = norm.ppf(0.5 + confidence_level / 2)

    data: dict = {
        'x_transformed': result.x_pred_transformed,
        'x_real': result.x_pred_original,
        'Operator_mean_real': result.operator_mean,
        'Operator_std_real': result.operator_std,
        'Operator_upper_real': result.operator_mean + z * result.operator_std,
        'Operator_lower_real': result.operator_mean - z * result.operator_std,
    }

    # Scalar diagnostics (broadcast to column length)
    n = len(result.x_pred_transformed)
    if result.operator_weights is not None and len(result.operator_weights) > 0:
        data['Operator_weight_mean'] = np.full(n, float(np.mean(result.operator_weights)))
        data['Operator_weight_min'] = np.full(n, float(np.min(result.operator_weights)))
        data['Operator_weight_max'] = np.full(n, float(np.max(result.operator_weights)))
    data['Operator_iterations'] = np.full(
        n, result.operator_iterations if result.operator_iterations is not None else np.nan,
    )
    if result.operator_between_variance is not None:
        data['Operator_between_variance_norm'] = np.full(n, result.operator_between_variance)
    if result.operator_within_variance_norm is not None:
        data['Operator_within_variance_norm'] = np.full(n, result.operator_within_variance_norm)
    if result.operator_within_variance_real is not None:
        data['Operator_within_variance_real'] = np.full(n, result.operator_within_variance_real)

    df = pd.DataFrame(data)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    filename = output_dir / f'Operator_Curve_{safe_key}.csv'
    df.to_csv(filename, index=False)

    if log_fn:
        log_fn(f"  Saved operator curve CSV to {filename}")


def export_operator_weights_csv(
    result: "SummaryGPRResult",
    index_ids: List[int],
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export operator-fusion weights to CSV."""
    if result.operator_weights is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    op_df = pd.DataFrame({
        'sample_id': result.sample_ids,
        'index_id': index_ids,
        'operator_weight': result.operator_weights,
    })
    op_filename = output_dir / f'Operator_Weights_{safe_key}.csv'
    op_df.to_csv(op_filename, index=False)
    if log_fn:
        log_fn(f"  Saved operator weights to {op_filename}")


def export_operator_history_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export operator weight history and between/within variance diagnostics."""
    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # Operator weight history
    if result.operator_weight_history and len(result.operator_weight_history) > 0:
        op_weight_df = pd.DataFrame(
            result.operator_weight_history,
            columns=[f'weight_{sid}' for sid in result.sample_ids],
        )
        op_weight_df.insert(0, 'iteration', range(1, len(op_weight_df) + 1))
        op_weight_filename = output_dir / f'Operator_Weight_History_{safe_key}.csv'
        op_weight_df.to_csv(op_weight_filename, index=False)
        if log_fn:
            log_fn(f"  Saved operator weight history to {op_weight_filename}")

    # Between-variance / within-variance diagnostics
    if result.operator_between_variance is not None:
        op_btw_filename = output_dir / f'Operator_Between_Variance_{safe_key}.txt'
        lines = [
            f"between_variance_norm: {result.operator_between_variance}",
        ]
        if result.operator_within_variance_norm is not None:
            lines.append(f"within_variance_norm: {result.operator_within_variance_norm}")
        if result.operator_within_variance_real is not None:
            lines.append(f"within_variance_real: {result.operator_within_variance_real}")
        op_btw_filename.write_text("\n".join(lines) + "\n")
        if log_fn:
            log_fn(f"  Saved operator between/within variance to {op_btw_filename}")
