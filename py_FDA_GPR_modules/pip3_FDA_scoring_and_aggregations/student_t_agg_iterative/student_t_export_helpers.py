# student_t_agg_iterative/student_t_export_helpers.py
"""
Export helpers for Student-t robust curve aggregation.

Handles CSV/text export of:
- Student-t curvewise weights
- Weight / energy / curve iteration history
- Aggregated curve with CI bands
- Covariance matrices
- Sigma calibration data
- Diagnostics (σ²_btw, ν, convergence)

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


def export_student_t_weights_csv(
    result: "SummaryGPRResult",
    index_ids: List[int],
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
    sample_ids: Optional[List[str]] = None,
) -> None:
    """Export Student-t curvewise weights and energies to CSV."""
    weights = getattr(result, "student_t_weights", None)
    if weights is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    ids = sample_ids if sample_ids is not None else result.sample_ids
    energies = getattr(result, "student_t_energies", None)

    data = {
        'sample_id': ids,
        'index_id': index_ids,
        'student_t_weight': weights,
    }
    if energies is not None:
        data['mahalanobis_energy'] = energies
        # Also show raw (un-normalised) weights
        raw_w = getattr(result, "student_t_weights_raw", None)
        if raw_w is not None:
            data['weight_raw'] = raw_w

    df = pd.DataFrame(data)
    filename = output_dir / f'StudentT_Weights_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved Student-t weights to {filename}")


def export_student_t_diagnostics(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t diagnostics to text file."""
    sigma2 = getattr(result, "student_t_sigma_btw_squared", None)
    if sigma2 is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    nu = getattr(result, "student_t_nu", None)
    weights = getattr(result, "student_t_weights", None)
    energies = getattr(result, "student_t_energies", None)

    # Compute nuN = nu * N_grid (effective DOF for Gamma(nuN/2, nuN/2))
    st_mean = getattr(result, 'student_t_mean', None)
    N_grid = len(st_mean) if st_mean is not None else None
    nuN = nu * N_grid if (nu is not None and N_grid is not None) else None

    lines = [
        f"sigma_btw_squared: {sigma2}",
        f"nu (raw): {nu}",
        f"nuN (effective DOF = nu * N_grid): {nuN}  [N_grid={N_grid}]",
        f"iterations: {getattr(result, 'student_t_iterations', 'N/A')}",
        f"converged: {getattr(result, 'student_t_converged', 'N/A')}",
        f"max_weight_delta: {getattr(result, 'student_t_max_weight_delta', 'N/A')}",
    ]

    # Variance stats
    within_norm = getattr(result, "student_t_within_variance_norm", None)
    within_real = getattr(result, "student_t_within_variance_real", None)
    pred_norm = getattr(result, "student_t_predictive_variance_norm", None)
    pred_real = getattr(result, "student_t_predictive_variance_real", None)
    if within_norm is not None:
        lines.append(f"within_variance_norm: {within_norm}")
    if within_real is not None:
        lines.append(f"within_variance_real: {within_real}")
    if pred_norm is not None:
        lines.append(f"predictive_variance_norm (C_agg + sigma_btw^2 I): {pred_norm}")
    if pred_real is not None:
        lines.append(f"predictive_variance_real (C_agg + sigma_btw^2 I): {pred_real}")

    if weights is not None:
        lines.append(f"n_models: {len(weights)}")
        lines.append(f"weights (normalised): {weights.tolist()}")
    if energies is not None:
        lines.append(f"energies d_r: {energies.tolist()}")

    # sigma_btw history (joint EM trajectory)
    sigma_btw_hist = getattr(result, "student_t_sigma_btw_history", None)
    if sigma_btw_hist is not None and len(sigma_btw_hist) > 0:
        lines.append(f"sigma_btw^2 initial: {sigma_btw_hist[0]}")
        lines.append(f"sigma_btw^2 final:   {sigma_btw_hist[-1]}")
        lines.append(f"sigma_btw^2 history: {sigma_btw_hist}")

    # nu history (degrees-of-freedom trajectory)
    nu_hist = getattr(result, "student_t_nu_history", None)
    if nu_hist is not None and len(nu_hist) > 0:
        lines.append(f"nu initial: {nu_hist[0]}")
        lines.append(f"nu final:   {nu_hist[-1]}")
        lines.append(f"nu history: {nu_hist}")
        if N_grid is not None:
            lines.append(f"nuN initial: {nu_hist[0] * N_grid}")
            lines.append(f"nuN final:   {nu_hist[-1] * N_grid}")
            lines.append(f"nuN history: {[v * N_grid for v in nu_hist]}")

    filename = output_dir / f'StudentT_Diagnostics_{safe_key}.txt'
    filename.write_text("\n".join(lines) + "\n")
    if log_fn:
        log_fn(f"  Saved Student-t diagnostics to {filename}")


def export_student_t_curve_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    confidence_level: float = 0.95,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t aggregated curve with CI bands."""
    from scipy.stats import norm as sp_norm

    mean_r = getattr(result, "student_t_mean", None)
    std_r = getattr(result, "student_t_std", None)
    if mean_r is None or std_r is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    z = sp_norm.ppf(0.5 + confidence_level / 2)

    std_pred_r = getattr(result, "student_t_std_predictive", std_r)
    if std_pred_r is None:
        std_pred_r = std_r

    data = {
        'x_transformed': result.x_pred_transformed,
        'x_real': result.x_pred_original,
        'student_t_mean_real': mean_r,
        'student_t_std_real': std_r,
        'student_t_upper_real': mean_r + z * std_r,
        'student_t_lower_real': mean_r - z * std_r,
        'student_t_std_predictive_real': std_pred_r,
        'student_t_upper_predictive_real': mean_r + z * std_pred_r,
        'student_t_lower_predictive_real': mean_r - z * std_pred_r,
    }

    # Normalised scale
    mean_n = getattr(result, "student_t_mean_norm", None)
    std_n = getattr(result, "student_t_std_norm", None)
    std_pred_n = getattr(result, "student_t_std_predictive_norm", None)
    if mean_n is not None and std_n is not None:
        if std_pred_n is None:
            std_pred_n = std_n
        data.update({
            'student_t_mean_norm': mean_n,
            'student_t_std_norm': std_n,
            'student_t_upper_norm': mean_n + z * std_n,
            'student_t_lower_norm': mean_n - z * std_n,
            'student_t_std_predictive_norm': std_pred_n,
            'student_t_upper_predictive_norm': mean_n + z * std_pred_n,
            'student_t_lower_predictive_norm': mean_n - z * std_pred_n,
        })

    data['sigma_btw_squared'] = getattr(result, "student_t_sigma_btw_squared", np.nan)
    data['nu'] = getattr(result, "student_t_nu", np.nan)

    df = pd.DataFrame(data)
    filename = output_dir / f'StudentT_Curve_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved Student-t curve CSV to {filename}")


def export_student_t_covariance_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t covariance matrices (posterior + predictive) in normalised scale."""
    C_agg = getattr(result, "student_t_cov_norm", None)
    if C_agg is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    agg_path = output_dir / f'StudentT_Covariance_Agg_{safe_key}.csv'
    pd.DataFrame(C_agg).to_csv(agg_path, index=False, header=False)
    if log_fn:
        log_fn(f"  Saved Student-t C_agg covariance to {agg_path}")

    C_pred = getattr(result, "student_t_cov_predictive_norm", None)
    if C_pred is not None:
        pred_path = output_dir / f'StudentT_Covariance_Pred_{safe_key}.csv'
        pd.DataFrame(C_pred).to_csv(pred_path, index=False, header=False)
        if log_fn:
            log_fn(f"  Saved Student-t C_pred covariance to {pred_path}")


def export_student_t_iteration_history_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t IRLS weight, energy, and curve iteration history CSVs."""
    wh = getattr(result, 'student_t_weight_history', None)
    ch = getattr(result, 'student_t_curve_history', None)
    eh = getattr(result, 'student_t_energy_history', None)
    if not wh and not ch:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # --- Weight history ---
    if wh and len(wh) > 0:
        n_curves = len(wh[0])
        sample_ids = getattr(result, 'sample_ids', None)
        if sample_ids and len(sample_ids) == n_curves:
            col_names = [f'weight_{sid}' for sid in sample_ids]
        else:
            col_names = [f'weight_sample{i}' for i in range(n_curves)]

        weight_df = pd.DataFrame(wh, columns=col_names)
        weight_df.insert(0, 'iteration', range(1, len(weight_df) + 1))
        filename = output_dir / f'StudentT_Weight_History_{safe_key}.csv'
        weight_df.to_csv(filename, index=False)
        if log_fn:
            log_fn(f"  Saved Student-t weight history to {filename}")

    # --- Energy history ---
    if eh and len(eh) > 0:
        n_curves = len(eh[0])
        sample_ids = getattr(result, 'sample_ids', None)
        if sample_ids and len(sample_ids) == n_curves:
            col_names = [f'energy_{sid}' for sid in sample_ids]
        else:
            col_names = [f'energy_sample{i}' for i in range(n_curves)]

        energy_df = pd.DataFrame(eh, columns=col_names)
        energy_df.insert(0, 'iteration', range(1, len(energy_df) + 1))

        # Append nu history as extra column if available
        nu_hist = getattr(result, 'student_t_nu_history', None)
        if nu_hist is not None and len(nu_hist) > 0:
            # nu_history has len = iterations + 1 (includes initial);
            # energy_history has len = iterations.  Align by dropping initial.
            nu_iter = nu_hist[1:] if len(nu_hist) > len(eh) else nu_hist
            if len(nu_iter) == len(energy_df):
                energy_df['nu'] = nu_iter

        filename = output_dir / f'StudentT_Energy_History_{safe_key}.csv'
        energy_df.to_csv(filename, index=False)
        if log_fn:
            log_fn(f"  Saved Student-t energy history to {filename}")

    # --- Curve history ---
    if ch and len(ch) > 0:
        x_trans = getattr(result, 'x_pred_transformed', None)
        curve_dict: dict = {}
        if x_trans is not None:
            curve_dict['x_transformed'] = x_trans
        for i, curve in enumerate(ch):
            curve_dict[f'iter_{i+1}'] = curve
        curve_df = pd.DataFrame(curve_dict)
        filename = output_dir / f'StudentT_Curve_History_{safe_key}.csv'
        curve_df.to_csv(filename, index=False)
        if log_fn:
            log_fn(f"  Saved Student-t curve history to {filename}")


def export_student_t_hyperparameter_history_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t hyperparameter trajectory (sigma_btw^2 and nu) per iteration."""
    sigma_hist = getattr(result, 'student_t_sigma_btw_history', None)
    nu_hist = getattr(result, 'student_t_nu_history', None)

    has_sigma = sigma_hist is not None and len(sigma_hist) >= 1
    has_nu = nu_hist is not None and len(nu_hist) >= 1
    if not has_sigma and not has_nu:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    # Both histories include the initial value at index 0.
    # Build a single table: iteration 0 = initial, 1..n = IRLS iterations.
    max_len = max(len(sigma_hist) if has_sigma else 0,
                  len(nu_hist) if has_nu else 0)
    data: dict = {'iteration': list(range(max_len))}
    if has_sigma:
        padded = list(sigma_hist) + [float('nan')] * (max_len - len(sigma_hist))
        data['sigma_btw_squared'] = padded[:max_len]
    if has_nu:
        padded = list(nu_hist) + [float('nan')] * (max_len - len(nu_hist))
        data['nu'] = padded[:max_len]
        # Also export nuN = nu * N_grid (effective DOF for Gamma(nuN/2, nuN/2))
        st_mean = getattr(result, 'student_t_mean', None)
        N_grid = len(st_mean) if st_mean is not None else None
        if N_grid is not None:
            data['nuN'] = [v * N_grid for v in padded[:max_len]]

    df = pd.DataFrame(data)
    filename = output_dir / f'StudentT_Hyperparameter_History_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved Student-t hyperparameter history to {filename}")


def export_student_t_sigma_calibration_csv(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[list] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    """Export Student-t sigma calibration data as CSV."""
    from scipy.interpolate import interp1d as _interp1d

    mean_r = getattr(result, "student_t_mean", None)
    std_pred = getattr(result, "student_t_std_predictive", None)
    if mean_r is None or std_pred is None:
        return
    if gprs is None or len(gprs) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original

    data: dict = {
        'x_real': x,
        'student_t_mean_real': mean_r,
        'student_t_std_pred_real': std_pred,
        'student_t_1sig_upper_real': mean_r + std_pred,
        'student_t_1sig_lower_real': mean_r - std_pred,
        'student_t_2sig_upper_real': mean_r + 2 * std_pred,
        'student_t_2sig_lower_real': mean_r - 2 * std_pred,
    }

    for i, gpr in enumerate(gprs):
        f_y = _interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred, kind='linear', fill_value='extrapolate',
        )
        data[f'sample_{i}_real'] = f_y(x)

    # Normalised
    mean_n = getattr(result, "student_t_mean_norm", None)
    std_pred_n = getattr(result, "student_t_std_predictive_norm", None)
    if mean_n is not None and std_pred_n is not None:
        data['student_t_mean_norm'] = mean_n
        data['student_t_std_pred_norm'] = std_pred_n
        data['student_t_1sig_upper_norm'] = mean_n + std_pred_n
        data['student_t_1sig_lower_norm'] = mean_n - std_pred_n
        data['student_t_2sig_upper_norm'] = mean_n + 2 * std_pred_n
        data['student_t_2sig_lower_norm'] = mean_n - 2 * std_pred_n

        for i, gpr in enumerate(gprs):
            y_norm = getattr(gpr, 'y_pred_normalized', None)
            if y_norm is not None:
                f_y = _interp1d(
                    gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
                    y_norm, kind='linear', fill_value='extrapolate',
                )
                data[f'sample_{i}_norm'] = f_y(x)

    df = pd.DataFrame(data)
    filename = output_dir / f'StudentT_Sigma_Calibration_{safe_key}.csv'
    df.to_csv(filename, index=False)
    if log_fn:
        log_fn(f"  Saved Student-t sigma calibration CSV to {filename}")
