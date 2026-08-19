# functional_GPR/fgpr_export_helpers.py
"""
Export helpers for Functional GPR (FGPR) aggregation.

Handles CSV/text export of:
- FGPR weights
- FGPR diagnostics (sigma_btw², NLL, variance stats)

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


def export_fgpr_weights_csv(
    result: "SummaryGPRResult",
    index_ids: List[int],
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    sample_ids: Optional[List[str]] = None,
) -> None:
    """Export FGPR weights to CSV.

    Parameters
    ----------
    sample_ids : list[str], optional
        If provided, use these instead of ``result.sample_ids``.
        Needed when outlier filtering narrows the curve set.
    """
    if result.fgpr_weights is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    ids = sample_ids if sample_ids is not None else result.sample_ids
    fgpr_df = pd.DataFrame({
        'sample_id': ids,
        'index_id': index_ids,
        'fgpr_weight': result.fgpr_weights,
    })
    fgpr_filename = output_dir / f'FGPR_Weights_{safe_key}.csv'
    fgpr_df.to_csv(fgpr_filename, index=False)
    if log_fn:
        log_fn(f"  Saved FGPR weights to {fgpr_filename}")


def export_fgpr_diagnostics(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export FGPR diagnostics (sigma_btw², NLL, variance) to text file."""
    if result.fgpr_sigma_btw_squared is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    fgpr_diag_filename = output_dir / f'FGPR_Diagnostics_{safe_key}.txt'
    fgpr_lines = [
        f"sigma_btw_squared: {result.fgpr_sigma_btw_squared}",
        f"nll_optimized: {result.fgpr_nll}",
    ]
    if result.fgpr_within_variance_norm is not None:
        fgpr_lines.append(f"within_variance_norm: {result.fgpr_within_variance_norm}")
    if result.fgpr_within_variance_real is not None:
        fgpr_lines.append(f"within_variance_real: {result.fgpr_within_variance_real}")
    if result.fgpr_predictive_variance_norm is not None:
        fgpr_lines.append(f"predictive_variance_norm (C_agg + sigma_btw^2): {result.fgpr_predictive_variance_norm}")
    if result.fgpr_predictive_variance_real is not None:
        fgpr_lines.append(f"predictive_variance_real (C_agg + sigma_btw^2): {result.fgpr_predictive_variance_real}")
    if result.fgpr_weights is not None:
        fgpr_lines.append(f"n_models: {len(result.fgpr_weights)}")
        fgpr_lines.append(f"weights: {result.fgpr_weights.tolist()}")

    # Structured C_btw diagnostics
    sp = getattr(result, 'fgpr_structured_btw_params', None)
    if sp is not None:
        fgpr_lines.append("")
        fgpr_lines.append("--- Structured C_btw parameters ---")
        fgpr_lines.append(f"sigma2_w  (white):  {sp.sigma2_w}")
        fgpr_lines.append(f"sigma2_s  (smooth): {sp.sigma2_s}")
        fgpr_lines.append(f"ell_b     (length): {sp.ell_b}")
        fgpr_lines.append(f"sigma2_o  (offset): {sp.sigma2_o}")
        fgpr_lines.append(f"sigma2_d  (drift):  {sp.sigma2_d}")
        fgpr_lines.append(f"sigma2_sc (scale):  {sp.sigma2_sc}")
        fgpr_lines.append(f"n_outer_iter: {sp.n_outer_iter}")
        fgpr_lines.append(f"converged: {sp.converged}")
        fgpr_lines.append(f"weight_converged: {sp.weight_converged}")
        fgpr_lines.append(f"max_weight_delta: {sp.max_weight_delta:.2e}")

    fgpr_diag_filename.write_text("\n".join(fgpr_lines) + "\n")
    if log_fn:
        log_fn(f"  Saved FGPR diagnostics to {fgpr_diag_filename}")


def export_fgpr_curve_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    confidence_level: float = 0.95,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export FGPR aggregated curve and variance in both observation
    (real) scale and normalized scale.

    Produces one CSV per group with columns:
      x_transformed, x_real,
      fgpr_mean_real, fgpr_std_real, fgpr_upper_real, fgpr_lower_real,
      fgpr_std_predictive_real, fgpr_upper_predictive_real, fgpr_lower_predictive_real,
      fgpr_mean_norm, fgpr_std_norm, fgpr_upper_norm, fgpr_lower_norm,
      fgpr_std_predictive_norm, fgpr_upper_predictive_norm, fgpr_lower_predictive_norm,
      sigma_btw_squared, nll
    """
    from scipy.stats import norm as sp_norm

    fgpr_mean = getattr(result, "fgpr_mean", None)
    fgpr_std = getattr(result, "fgpr_std", None)
    if fgpr_mean is None or fgpr_std is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    z = sp_norm.ppf(0.5 + confidence_level / 2)

    # --- Real (observation) scale ---
    mean_real = result.fgpr_mean
    std_real = result.fgpr_std
    upper_real = mean_real + z * std_real
    lower_real = mean_real - z * std_real

    std_pred_real = getattr(result, "fgpr_std_predictive", std_real)
    if std_pred_real is None:
        std_pred_real = std_real
    upper_pred_real = mean_real + z * std_pred_real
    lower_pred_real = mean_real - z * std_pred_real

    # --- Normalized scale ---
    mean_norm = getattr(result, "fgpr_mean_norm", None)
    std_norm = getattr(result, "fgpr_std_norm", None)
    std_pred_norm = getattr(result, "fgpr_std_predictive_norm", None)

    has_norm = mean_norm is not None and std_norm is not None

    data = {
        'x_transformed': result.x_pred_transformed,
        'x_real': result.x_pred_original,
        # Observation (real) scale
        'fgpr_mean_real': mean_real,
        'fgpr_std_real': std_real,
        'fgpr_upper_real': upper_real,
        'fgpr_lower_real': lower_real,
        'fgpr_std_predictive_real': std_pred_real,
        'fgpr_upper_predictive_real': upper_pred_real,
        'fgpr_lower_predictive_real': lower_pred_real,
    }

    if has_norm:
        upper_norm = mean_norm + z * std_norm
        lower_norm = mean_norm - z * std_norm
        if std_pred_norm is None:
            std_pred_norm = std_norm
        upper_pred_norm = mean_norm + z * std_pred_norm
        lower_pred_norm = mean_norm - z * std_pred_norm

        data.update({
            'fgpr_mean_norm': mean_norm,
            'fgpr_std_norm': std_norm,
            'fgpr_upper_norm': upper_norm,
            'fgpr_lower_norm': lower_norm,
            'fgpr_std_predictive_norm': std_pred_norm,
            'fgpr_upper_predictive_norm': upper_pred_norm,
            'fgpr_lower_predictive_norm': lower_pred_norm,
        })

    # Scalar diagnostics (broadcast as constant columns)
    data['sigma_btw_squared'] = result.fgpr_sigma_btw_squared
    data['nll'] = result.fgpr_nll

    df = pd.DataFrame(data)
    filename = output_dir / f'FGPR_Curve_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved FGPR curve CSV to {filename}")


def export_fgpr_covariance_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export FGPR covariance matrices (C_agg and C_pred) in normalized scale.

    Produces two CSVs per group:
      - ``FGPR_Covariance_Agg_{key}.csv``  — C_agg (posterior)
      - ``FGPR_Covariance_Pred_{key}.csv`` — C_pred = C_agg + σ_btw² I (predictive)

    These CSVs can fully redraw ``FGPR_Covariance_{key}.png``.
    """
    C_agg = getattr(result, "fgpr_cov_norm", None)
    if C_agg is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # Save C_agg
    agg_path = output_dir / f'FGPR_Covariance_Agg_{safe_key}.csv'
    pd.DataFrame(C_agg).to_csv(agg_path, index=False, header=False)
    if log_fn:
        log_fn(f"  Saved FGPR C_agg covariance CSV to {agg_path}")

    # Save C_pred if available
    C_pred = getattr(result, "fgpr_cov_predictive_norm", None)
    if C_pred is not None:
        pred_path = output_dir / f'FGPR_Covariance_Pred_{safe_key}.csv'
        pd.DataFrame(C_pred).to_csv(pred_path, index=False, header=False)
        if log_fn:
            log_fn(f"  Saved FGPR C_pred covariance CSV to {pred_path}")


def export_fgpr_sigma_calibration_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[list] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export FGPR sigma calibration data as CSV.

    Columns: x_real, fgpr_mean_real, fgpr_1sig_upper/lower, fgpr_2sig_upper/lower,
    individual curve predictions (sample_0, sample_1, ...) in real scale,
    and corresponding normalized columns if available.

    This CSV fully describes the FGPR sigma calibration plots.
    """
    from scipy.interpolate import interp1d as _interp1d

    fgpr_mean = getattr(result, "fgpr_mean", None)
    fgpr_std_pred = getattr(result, "fgpr_std_predictive", None)
    if fgpr_mean is None or fgpr_std_pred is None:
        return
    if gprs is None or len(gprs) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original

    data: dict = {
        'x_real': x,
        'fgpr_mean_real': fgpr_mean,
        'fgpr_std_pred_real': fgpr_std_pred,
        'fgpr_1sig_upper_real': fgpr_mean + fgpr_std_pred,
        'fgpr_1sig_lower_real': fgpr_mean - fgpr_std_pred,
        'fgpr_2sig_upper_real': fgpr_mean + 2 * fgpr_std_pred,
        'fgpr_2sig_lower_real': fgpr_mean - 2 * fgpr_std_pred,
    }

    # Interpolate individual curves (real)
    for i, gpr in enumerate(gprs):
        f_y = _interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred, kind='linear', fill_value='extrapolate',
        )
        data[f'sample_{i}_real'] = f_y(x)

    # Normalized scale
    fgpr_mean_n = getattr(result, "fgpr_mean_norm", None)
    fgpr_std_pred_n = getattr(result, "fgpr_std_predictive_norm", None)
    if fgpr_mean_n is not None and fgpr_std_pred_n is not None:
        data['fgpr_mean_norm'] = fgpr_mean_n
        data['fgpr_std_pred_norm'] = fgpr_std_pred_n
        data['fgpr_1sig_upper_norm'] = fgpr_mean_n + fgpr_std_pred_n
        data['fgpr_1sig_lower_norm'] = fgpr_mean_n - fgpr_std_pred_n
        data['fgpr_2sig_upper_norm'] = fgpr_mean_n + 2 * fgpr_std_pred_n
        data['fgpr_2sig_lower_norm'] = fgpr_mean_n - 2 * fgpr_std_pred_n

        for i, gpr in enumerate(gprs):
            y_norm = getattr(gpr, 'y_pred_normalized', None)
            if y_norm is not None:
                f_y = _interp1d(
                    gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
                    y_norm, kind='linear', fill_value='extrapolate',
                )
                data[f'sample_{i}_norm'] = f_y(x)

    df = pd.DataFrame(data)
    filename = output_dir / f'FGPR_Sigma_Calibration_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved FGPR sigma calibration CSV to {filename}")


def export_fgpr_iteration_history_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Export FGPR structured weight convergence history CSVs.

    Produces two files (modelled after iterative GPR history export):

    - ``FGPR_Weight_History_{key}.csv``
      Columns: iteration, weight_sample0, weight_sample1, ...
    - ``FGPR_Curve_History_{key}.csv``
      Columns: x_transformed, iter_1, iter_2, ...
    """
    # Weight history
    wh = getattr(result, 'fgpr_weight_history', None)
    ch = getattr(result, 'fgpr_curve_history', None)
    if not wh and not ch:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # --- Weight history ---
    if wh and len(wh) > 0:
        n_curves = len(wh[0])
        # Use sample_ids if available, else generic names
        sample_ids = getattr(result, 'sample_ids', None)
        if sample_ids and len(sample_ids) == n_curves:
            col_names = [f'weight_{sid}' for sid in sample_ids]
        else:
            col_names = [f'weight_sample{i}' for i in range(n_curves)]

        weight_df = pd.DataFrame(wh, columns=col_names)
        weight_df.insert(0, 'iteration', range(1, len(weight_df) + 1))
        weight_filename = output_dir / f'FGPR_Weight_History_{safe_key}.csv'
        weight_df.to_csv(weight_filename, index=False)
        if log_fn:
            log_fn(f"  Saved FGPR weight history to {weight_filename}")

    # --- Curve history (mu_agg per iteration) ---
    if ch and len(ch) > 0:
        x_trans = getattr(result, 'x_pred_transformed', None)
        curve_dict: dict = {}
        if x_trans is not None:
            curve_dict['x_transformed'] = x_trans
        for i, curve in enumerate(ch):
            curve_dict[f'iter_{i+1}'] = curve
        curve_df = pd.DataFrame(curve_dict)
        curve_filename = output_dir / f'FGPR_Curve_History_{safe_key}.csv'
        curve_df.to_csv(curve_filename, index=False)
        if log_fn:
            log_fn(f"  Saved FGPR curve history to {curve_filename}")
