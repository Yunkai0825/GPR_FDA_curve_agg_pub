# iterative_weight_sum_GPR/iterative_export_helpers.py
"""
Export helpers for Iterative Weighted-Sum GPR aggregation.

Handles CSV export of:
- Summary curve (mean + CI)
- Converged weights
- Weight history (per iteration)
- Curve history (per iteration)
- Sigma calibration data

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
    from ..summary_gpr_loader import IndividualGPRData


def export_iterative_summary_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    confidence_level: float = 0.75,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export iterative aggregation summary curve to CSV.

    Contains only iterative-method columns: x coordinates,
    mean, std, and CI bounds in both real and normalised scales.
    """
    from scipy.stats import norm

    os.makedirs(output_dir, exist_ok=True)
    z = norm.ppf(0.5 + confidence_level / 2)

    y_std_real = getattr(result, "y_std_real", None)
    y_std_norm = getattr(result, "y_std_norm", None)

    data: dict = {
        'x_transformed': result.x_pred_transformed,
        'x_real': result.x_pred_original,
        'y_real': result.y_mean,
        'y_normalised': result.y_mean_norm,
    }

    if y_std_real is not None:
        data['Std_real'] = y_std_real
        data['Upper_CI_real'] = result.y_mean + z * y_std_real
        data['Lower_CI_real'] = result.y_mean - z * y_std_real
    if y_std_norm is not None:
        data['Std_normalised'] = y_std_norm
        # Convert normalised std to observation units via the aggregated
        # scaling so the comparison CI lives on the same axis as the
        # real-variance CI.
        y_scaling = getattr(result, 'y_scaling', None)
        if y_scaling is not None and hasattr(y_scaling, 'inverse_transform_std'):
            std_norm_as_real = y_scaling.inverse_transform_std(y_std_norm)
        else:
            # Fallback: use the raw normalised std (will be in wrong units)
            std_norm_as_real = y_std_norm
        data['Upper_CI_normalised'] = result.y_mean + z * std_norm_as_real
        data['Lower_CI_normalised'] = result.y_mean - z * std_norm_as_real

    df = pd.DataFrame(data)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    filename = output_dir / f'Summary_GPR_{safe_key}.csv'
    df.to_csv(filename, index=False)

    if log_fn:
        log_fn(f"  Saved iterative summary CSV to {filename}")


def export_iterative_weights_csv(
    result: "SummaryGPRResult",
    index_ids: List[int],
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export converged iterative weights to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Average weights if pointwise
    if result.weights.ndim == 2:
        weights = result.weights.mean(axis=1)
    else:
        weights = result.weights

    df = pd.DataFrame({
        'sample_id': result.sample_ids,
        'index_id': index_ids,
        'converged_weight': weights,
    })

    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    filename = output_dir / f'Converged_Weights_{safe_key}.csv'
    df.to_csv(filename, index=False)

    if log_fn:
        log_fn(f"  Saved weights to {filename}")


def export_iterative_history_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export iterative weight history and curve history CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # Weight history
    if result.weight_history and len(result.weight_history) > 0:
        weight_df = pd.DataFrame(
            result.weight_history,
            columns=[f'weight_{sid}' for sid in result.sample_ids],
        )
        weight_df.insert(0, 'iteration', range(1, len(weight_df) + 1))
        weight_filename = output_dir / f'Weight_History_{safe_key}.csv'
        weight_df.to_csv(weight_filename, index=False)
        if log_fn:
            log_fn(f"  Saved weight history to {weight_filename}")

    # Curve history
    if result.curve_history and len(result.curve_history) > 0:
        curve_dict = {'x_transformed': result.x_pred_transformed}
        for i, curve in enumerate(result.curve_history):
            curve_dict[f'iter_{i+1}'] = curve
        curve_df = pd.DataFrame(curve_dict)
        curve_filename = output_dir / f'Curve_History_{safe_key}.csv'
        curve_df.to_csv(curve_filename, index=False)
        if log_fn:
            log_fn(f"  Saved curve history to {curve_filename}")


def export_iterative_sigma_calibration_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[List["IndividualGPRData"]] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export iterative sigma calibration data as CSV.

    Columns: x_real, mean_real, std_real, 1sig/2sig bands,
    individual curve predictions (sample_0, ...) in real scale,
    and corresponding normalized columns if available.
    """
    from scipy.interpolate import interp1d

    if result.y_std_real is None or gprs is None or len(gprs) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original
    mean_r = result.y_mean
    std_r = result.y_std_real

    data: dict = {
        'x_real': x,
        'mean_real': mean_r,
        'std_real': std_r,
        '1sig_upper_real': mean_r + std_r,
        '1sig_lower_real': mean_r - std_r,
        '2sig_upper_real': mean_r + 2 * std_r,
        '2sig_lower_real': mean_r - 2 * std_r,
    }

    for i, gpr in enumerate(gprs):
        f_y = interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred, kind='linear', fill_value='extrapolate',
        )
        data[f'sample_{i}_real'] = f_y(x)

    # Normalized scale
    mean_n = getattr(result, 'y_mean_norm', None)
    std_n = getattr(result, 'y_std_norm', None)
    if mean_n is not None and std_n is not None:
        data['mean_norm'] = mean_n
        data['std_norm'] = std_n
        data['1sig_upper_norm'] = mean_n + std_n
        data['1sig_lower_norm'] = mean_n - std_n
        data['2sig_upper_norm'] = mean_n + 2 * std_n
        data['2sig_lower_norm'] = mean_n - 2 * std_n

        for i, gpr in enumerate(gprs):
            y_norm = getattr(gpr, 'y_pred_normalized', None)
            if y_norm is not None:
                f_y = interp1d(
                    gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
                    y_norm, kind='linear', fill_value='extrapolate',
                )
                data[f'sample_{i}_norm'] = f_y(x)

    df = pd.DataFrame(data)
    filename = output_dir / f'Sigma_Calibration_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved iterative sigma calibration CSV to {filename}")
