# student_t_agg_iterative/student_t_plot_helpers.py
"""
Plot helpers for Student-t robust curve aggregation.

Handles:
- Aggregated curve plot (mean + posterior/predictive CI, real + normalised)
- Curvewise weight distribution
- Weight convergence across IRLS iterations
- Mahalanobis energy convergence
- Aggregated curve iterations
- Sigma calibration (deviation scatter + 1σ/2σ bands)
- Covariance heatmap
- Diagonal std profile

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from scipy.interpolate import interp1d
from scipy.stats import norm as sp_norm

from ..summary_gpr_plotting import plot_weight_distribution, plot_weight_convergence
from ...pip0_dataloading.filename_parser import format_group_key_title

if TYPE_CHECKING:
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult
    from ..summary_gpr_loader import IndividualGPRData

_format_group_key_title = format_group_key_title


# =====================================================================
# Student-t Aggregated Curve Plot
# =====================================================================

def plot_student_t_curve(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[List["IndividualGPRData"]] = None,
    *,
    confidence_level: float = 0.95,
    individual_alpha: float = 0.20,
    min_time_cap: float = 0.01,
    max_time_cap: Optional[float] = None,
    x_axis_label: str = "Time (s)",
    y_axis_label: str = "Current (A/cm²)",
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot Student-t aggregated curve with posterior + predictive CI bands."""
    mean_r = getattr(result, "student_t_mean", None)
    std_pred_r = getattr(result, "student_t_std_predictive", None)
    if mean_r is None or std_pred_r is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    z = sp_norm.ppf(0.5 + confidence_level / 2)
    x = result.x_pred_original
    title_str = _format_group_key_title(result.group_key)
    nu_raw = getattr(result, "student_t_nu", None)
    N_grid = len(mean_r) if mean_r is not None else None
    nuN = nu_raw * N_grid if (nu_raw is not None and N_grid is not None) else "?"

    if max_time_cap is None:
        max_time_cap = float(x.max())
        if gprs:
            for gpr in gprs:
                x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
                max_time_cap = max(max_time_cap, float(x_ind.max()))
        max_time_cap *= 1.05

    # Layout: real + normalised
    has_norm = (getattr(result, "student_t_mean_norm", None) is not None
                and getattr(result, "student_t_std_predictive_norm", None) is not None)
    n_panels = 2 if has_norm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # --- Panel 1: Real scale ---
    ax = axes[0]
    std_r = getattr(result, "student_t_std", std_pred_r)

    if gprs:
        for gpr in gprs:
            x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
            ax.plot(x_ind, gpr.y_pred, color='gray', alpha=individual_alpha, linewidth=0.8)

    ax.fill_between(x, mean_r - z * std_pred_r, mean_r + z * std_pred_r,
                    color='C1', alpha=0.10,
                    label=f'Predictive {confidence_level*100:.0f}% CI')
    ax.fill_between(x, mean_r - z * std_r, mean_r + z * std_r,
                    color='C1', alpha=0.25,
                    label=f'Posterior {confidence_level*100:.0f}% CI')
    ax.plot(x, mean_r, 'C1-', linewidth=1.8, label='Student-t mean')

    ax.set_xscale('log')
    ax.set_xlim([min_time_cap, max_time_cap])
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    nuN_str = f"{nuN:.4g}" if isinstance(nuN, (int, float)) else str(nuN)
    ax.set_title(f"Student-t (obs. scale) — {title_str}  "
                 f"({result.n_curves} curves, νN={nuN_str})")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)

    # --- Panel 2: Normalised scale ---
    if has_norm:
        ax2 = axes[1]
        mean_n = result.student_t_mean_norm
        std_pred_n = result.student_t_std_predictive_norm
        std_n = getattr(result, "student_t_std_norm", std_pred_n)

        if gprs:
            for gpr in gprs:
                x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
                ax2.plot(x_ind, gpr.y_pred_normalized, color='gray',
                         alpha=individual_alpha, linewidth=0.8)

        ax2.fill_between(x, mean_n - z * std_pred_n, mean_n + z * std_pred_n,
                         color='C2', alpha=0.10,
                         label=f'Predictive {confidence_level*100:.0f}% CI')
        ax2.fill_between(x, mean_n - z * std_n, mean_n + z * std_n,
                         color='C2', alpha=0.25,
                         label=f'Posterior {confidence_level*100:.0f}% CI')
        ax2.plot(x, mean_n, 'C2-', linewidth=1.8, label='Student-t mean (norm)')

        ax2.set_xscale('log')
        ax2.set_xlim([min_time_cap, max_time_cap])
        ax2.set_xlabel(x_axis_label)
        ax2.set_ylabel(f"{y_axis_label} (normalized)")
        ax2.set_title(f"Student-t (norm. scale) — {title_str}")
        ax2.legend(loc='best', fontsize='small')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)

    output_path = output_dir / f'StudentT_Summary_{safe_key}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t curve plot to {output_path}")
    return output_path


# =====================================================================
# Sigma Calibration
# =====================================================================

def plot_student_t_sigma_calibration(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[List["IndividualGPRData"]] = None,
    *,
    x_axis_label: str = "Time (s)",
    y_axis_label: str = "Current (A/cm²)",
    verbose: bool = True,
    dpi: int = 200,
) -> Optional[Path]:
    """Student-t sigma calibration: deviation scatter with 1σ/2σ bands."""
    mean_r = getattr(result, "student_t_mean", None)
    std_pred_r = getattr(result, "student_t_std_predictive", None)
    if mean_r is None or std_pred_r is None:
        return None
    if gprs is None or len(gprs) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original

    # Interpolate individual curves
    y_real_stack = []
    for gpr in gprs:
        f_y = interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred, kind='linear', fill_value='extrapolate',
        )
        y_real_stack.append(f_y(x))
    y_real_arr = np.vstack(y_real_stack)

    mean_n = getattr(result, "student_t_mean_norm", None)
    std_pred_n = getattr(result, "student_t_std_predictive_norm", None)
    has_norm = mean_n is not None and std_pred_n is not None

    y_norm_stack = None
    if has_norm:
        y_norm_stack = []
        for gpr in gprs:
            y_norm = getattr(gpr, 'y_pred_normalized', None)
            if y_norm is None:
                has_norm = False
                break
            f_y = interp1d(
                gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
                y_norm, kind='linear', fill_value='extrapolate',
            )
            y_norm_stack.append(f_y(x))

    n_panels = 2 if has_norm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # --- Real panel ---
    dev_real = y_real_arr - mean_r
    cov1 = float(np.mean((dev_real >= -std_pred_r) & (dev_real <= std_pred_r)))
    cov2 = float(np.mean((dev_real >= -2 * std_pred_r) & (dev_real <= 2 * std_pred_r)))

    ax0 = axes[0]
    ax0.axhline(0, color='k', linewidth=1.0, linestyle='--')
    ax0.fill_between(x, -2 * std_pred_r, 2 * std_pred_r,
                     color='C1', alpha=0.10, label='Student-t 2σ (pred)')
    ax0.fill_between(x, -std_pred_r, std_pred_r,
                     color='C1', alpha=0.22, label='Student-t 1σ (pred)')
    ax0.scatter(np.tile(x, len(y_real_stack)), dev_real.flatten(),
                s=6, alpha=0.08, color='gray', label='Samples')
    ax0.set_xscale('log')
    ax0.set_xlabel(x_axis_label)
    ax0.set_ylabel(f"Deviation ({y_axis_label})")
    ax0.set_title(f"Student-t σ calibration (real): {_format_group_key_title(result.group_key)}")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
    txt0 = [f"Coverage (pred, real):",
            f"  1σ: {cov1*100:.1f}%",
            f"  2σ: {cov2*100:.1f}%"]
    ax0.text(0.02, 0.98, "\n".join(txt0), transform=ax0.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax0.legend(loc='best', fontsize='small')

    # --- Normalised panel ---
    if has_norm and y_norm_stack:
        y_norm_arr = np.vstack(y_norm_stack)
        dev_norm = y_norm_arr - mean_n
        cov1n = float(np.mean((dev_norm >= -std_pred_n) & (dev_norm <= std_pred_n)))
        cov2n = float(np.mean((dev_norm >= -2 * std_pred_n) & (dev_norm <= 2 * std_pred_n)))

        ax1 = axes[1]
        ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
        ax1.fill_between(x, -2 * std_pred_n, 2 * std_pred_n,
                         color='C2', alpha=0.10, label='Student-t 2σ (pred, norm)')
        ax1.fill_between(x, -std_pred_n, std_pred_n,
                         color='C2', alpha=0.22, label='Student-t 1σ (pred, norm)')
        ax1.scatter(np.tile(x, len(y_norm_stack)), dev_norm.flatten(),
                    s=6, alpha=0.08, color='gray', label='Samples (norm)')
        ax1.set_xscale('log')
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel(f"Deviation ({y_axis_label}, norm)")
        ax1.set_title(f"Student-t σ calibration (norm): {_format_group_key_title(result.group_key)}")
        ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
        txt1 = [f"Coverage (pred, norm):",
                f"  1σ: {cov1n*100:.1f}%",
                f"  2σ: {cov2n*100:.1f}%"]
        ax1.text(0.02, 0.98, "\n".join(txt1), transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax1.legend(loc='best', fontsize='small')

    output_path = output_dir / f'StudentT_Sigma_Calibration_{safe_key}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t sigma calibration to {output_path}")
    return output_path


# =====================================================================
# Weight Distribution
# =====================================================================

def plot_student_t_weight_distribution(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
) -> Optional[Path]:
    """Plot Student-t curvewise weight distribution."""
    weights = getattr(result, "student_t_weights", None)
    if weights is None or len(weights) <= 1:
        return None

    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    path = output_dir / f'StudentT_Weight_Distribution_{safe_key}.png'
    return plot_weight_distribution(
        weights=weights,
        output_path=path,
        group_key=result.group_key + " (Student-t)",
        verbose=verbose,
    )


# =====================================================================
# Weight Convergence
# =====================================================================

def plot_student_t_weight_convergence(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot Student-t IRLS weight convergence across iterations."""
    wh = getattr(result, 'student_t_weight_history', None)
    if not wh or len(wh) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = Path(output_dir) / f'StudentT_Weight_Convergence_{safe_key}.png'

    weight_arr = np.array(wh)
    n_iters, n_curves = weight_arr.shape
    iterations = np.arange(1, n_iters + 1)
    sample_ids = getattr(result, 'sample_ids', None)

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in range(n_curves):
        label = sample_ids[r] if (sample_ids and r < len(sample_ids)) else f'curve {r}'
        ax.plot(iterations, weight_arr[:, r], 'o-', markersize=4, label=label)

    ax.set_xlabel("IRLS Iteration", fontsize=11)
    ax.set_ylabel("Weight (normalised)", fontsize=11)
    nu_raw = getattr(result, "student_t_nu", None)
    st_mean = getattr(result, 'student_t_mean', None)
    N_grid = len(st_mean) if st_mean is not None else None
    nuN = nu_raw * N_grid if (nu_raw is not None and N_grid is not None) else "?"
    nuN_str = f"{nuN:.4g}" if isinstance(nuN, (int, float)) else str(nuN)
    ax.set_title(
        f"Student-t Weight Convergence — "
        f"{_format_group_key_title(result.group_key)}"
        f"  ({n_curves} curves, νN={nuN_str})",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=max(1, n_curves // 4))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_iters + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t weight convergence plot to {output_path}")
    return output_path


# =====================================================================
# Energy Convergence
# =====================================================================

def plot_student_t_energy_convergence(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot Mahalanobis energy d_r convergence across IRLS iterations."""
    eh = getattr(result, 'student_t_energy_history', None)
    if not eh or len(eh) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = Path(output_dir) / f'StudentT_Energy_Convergence_{safe_key}.png'

    energy_arr = np.array(eh)
    n_iters, n_curves = energy_arr.shape
    iterations = np.arange(1, n_iters + 1)
    sample_ids = getattr(result, 'sample_ids', None)

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in range(n_curves):
        label = sample_ids[r] if (sample_ids and r < len(sample_ids)) else f'curve {r}'
        ax.plot(iterations, energy_arr[:, r], 'o-', markersize=4, label=label)

    nu_raw = getattr(result, "student_t_nu", None)
    st_mean = getattr(result, 'student_t_mean', None)
    N_grid = len(st_mean) if st_mean is not None else None
    nuN = nu_raw * N_grid if (nu_raw is not None and N_grid is not None) else "?"
    nuN_str = f"{nuN:.4g}" if isinstance(nuN, (int, float)) else str(nuN)
    ax.axhline(float(result.x_pred_transformed.shape[0]) if hasattr(result, 'x_pred_transformed') else 0,
               color='k', ls='--', lw=0.8, alpha=0.5, label='N (grid pts)')
    ax.set_xlabel("IRLS Iteration", fontsize=11)
    ax.set_ylabel("Mahalanobis Energy $d_r$", fontsize=11)
    ax.set_title(
        f"Student-t Energy Convergence — "
        f"{_format_group_key_title(result.group_key)}"
        f"  (νN={nuN_str})",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=max(1, n_curves // 4))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_iters + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t energy convergence plot to {output_path}")
    return output_path


# =====================================================================
# Curve Iterations
# =====================================================================

def plot_student_t_curve_iterations(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot Student-t aggregated mean at each IRLS iteration."""
    ch = getattr(result, 'student_t_curve_history', None)
    if not ch or len(ch) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = Path(output_dir) / f'StudentT_Curve_Iterations_{safe_key}.png'

    x = getattr(result, 'x_pred_transformed', None)
    if x is None:
        x = np.arange(len(ch[0]))

    fig, ax = plt.subplots(figsize=(10, 6))
    n_iters = len(ch)
    cmap = plt.cm.viridis
    for i, curve in enumerate(ch):
        alpha = 0.3 + 0.7 * (i / max(1, n_iters - 1))
        color = cmap(i / max(1, n_iters - 1))
        ax.plot(x, curve, color=color, alpha=alpha,
                label=f'iter {i+1}', linewidth=1.5)

    ax.set_xlabel("x (transformed)", fontsize=11)
    ax.set_ylabel("Aggregated mean (normalised)", fontsize=11)
    ax.set_title(
        f"Student-t Curve Iterations — "
        f"{_format_group_key_title(result.group_key)}"
        f"  ({n_iters} iters)",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=max(1, n_iters // 5))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t curve iterations plot to {output_path}")
    return output_path


# =====================================================================
# Covariance Heatmap
# =====================================================================

def plot_student_t_covariance_heatmap(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot Student-t aggregated and predictive covariance heatmaps."""
    C_agg = getattr(result, "student_t_cov_norm", None)
    if C_agg is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = output_dir / f'StudentT_Covariance_{safe_key}.png'

    C_pred = getattr(result, "student_t_cov_predictive_norm", None)
    panels = [(C_agg, r"$C_{\mathrm{agg}}$ (normalised)")]
    if C_pred is not None:
        panels.append((C_pred, r"$C_{\mathrm{pred}}$ (predictive, norm)"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    all_diag_max = max(np.max(np.abs(np.diag(C))) for C, _ in panels)
    vmax = all_diag_max
    linthresh = vmax * 1e-6

    for ax, (C, title) in zip(axes, panels):
        C_clipped = np.clip(C, -vmax, vmax)
        im = ax.imshow(
            C_clipped, aspect="auto", cmap="RdBu_r",
            norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
            origin="upper",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("grid index $j$", fontsize=9)
        ax.set_ylabel("grid index $k$", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"Student-t Covariance — {_format_group_key_title(result.group_key)}"
        f"  ({result.n_curves} curves)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t covariance heatmap to {output_path}")
    return output_path


# =====================================================================
# Diagonal Std Profile
# =====================================================================

def plot_student_t_diagonal_std(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot sqrt(diag(C)) profiles for Student-t aggregated/predictive covariances."""
    C_agg = getattr(result, "student_t_cov_norm", None)
    if C_agg is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = output_dir / f'StudentT_DiagStd_{safe_key}.png'

    fig, ax = plt.subplots(figsize=(8, 4.5))

    diag_agg = np.sqrt(np.clip(np.diag(C_agg), 0, None))
    ax.plot(diag_agg, label=r"$\sqrt{\mathrm{diag}(C_{\mathrm{agg}})}$",
            linewidth=1.5, color="tab:orange")

    C_pred = getattr(result, "student_t_cov_predictive_norm", None)
    if C_pred is not None:
        diag_pred = np.sqrt(np.clip(np.diag(C_pred), 0, None))
        ax.plot(diag_pred, label=r"$\sqrt{\mathrm{diag}(C_{\mathrm{pred}})}$",
                linewidth=1.5, color="tab:red", linestyle="--")

    sigma2 = getattr(result, "student_t_sigma_btw_squared", None)
    if sigma2 is not None:
        ax.axhline(np.sqrt(sigma2), color="tab:purple", linestyle=":",
                   linewidth=1, label=r"$\sigma_{\mathrm{btw}}$")

    ax.set_xlabel("Grid index $j$", fontsize=11)
    ax.set_ylabel("Std (normalised)", fontsize=11)
    nu_raw = getattr(result, "student_t_nu", None)
    N_grid_diag = len(getattr(result, 'student_t_mean', [])) or None
    nuN = nu_raw * N_grid_diag if (nu_raw is not None and N_grid_diag) else "?"
    nuN_str = f"{nuN:.4g}" if isinstance(nuN, (int, float)) else str(nuN)
    ax.set_title(
        f"Student-t Posterior Std Profile — "
        f"{_format_group_key_title(result.group_key)}"
        f"  ({result.n_curves} curves, νN={nuN_str})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t diagonal std profile to {output_path}")
    return output_path


# =====================================================================
# Hyperparameter History (σ²_btw and ν vs iteration)
# =====================================================================

def plot_student_t_hyperparameter_history(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """Plot σ²_btw and ν trajectories across IRLS iterations.

    Uses a dual-axis layout: left axis for σ²_btw, right axis for ν.
    Both histories include the initial value at iteration 0.
    """
    sigma_hist = getattr(result, 'student_t_sigma_btw_history', None)
    nu_hist = getattr(result, 'student_t_nu_history', None)

    has_sigma = sigma_hist is not None and len(sigma_hist) >= 2
    has_nu = nu_hist is not None and len(nu_hist) >= 2
    if not has_sigma and not has_nu:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = (
        Path(output_dir)
        / f'StudentT_Hyperparameter_History_{safe_key}.png'
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_sigma = "tab:blue"
    color_nu = "tab:red"

    # --- σ²_btw on left axis ---
    if has_sigma:
        iters_sigma = np.arange(0, len(sigma_hist))
        ax1.plot(iters_sigma, sigma_hist, 'o-', color=color_sigma,
                 markersize=5, linewidth=1.5, label=r"$\sigma^2_{\mathrm{btw}}$")
        ax1.set_ylabel(r"$\sigma^2_{\mathrm{btw}}$", fontsize=12, color=color_sigma)
        ax1.tick_params(axis='y', labelcolor=color_sigma)
        # If values span several orders of magnitude, use log scale
        vals = np.array(sigma_hist)
        vals_pos = vals[vals > 0]
        if len(vals_pos) >= 2 and vals_pos.max() / vals_pos.min() > 100:
            ax1.set_yscale('log')
    else:
        ax1.set_ylabel(r"$\sigma^2_{\mathrm{btw}}$", fontsize=12)

    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # --- νN on right axis (effective DOF = ν × N_grid) ---
    # Retrieve N_grid from result to compute νN
    st_mean = getattr(result, 'student_t_mean', None)
    N_grid = len(st_mean) if st_mean is not None else None

    if has_nu:
        ax2 = ax1.twinx()
        iters_nu = np.arange(0, len(nu_hist))
        if N_grid is not None:
            nuN_hist = [v * N_grid for v in nu_hist]
            ax2.plot(iters_nu, nuN_hist, 's--', color=color_nu,
                     markersize=5, linewidth=1.5, label=r"$\nu N$")
            ax2.set_ylabel(r"$\nu N$ (effective DOF)", fontsize=12, color=color_nu)
            # Also plot raw ν as thin dotted line on a secondary annotation
            vals_nuN = np.array(nuN_hist)
            vals_nuN_pos = vals_nuN[vals_nuN > 0]
            if len(vals_nuN_pos) >= 2 and vals_nuN_pos.max() / vals_nuN_pos.min() > 100:
                ax2.set_yscale('log')
        else:
            ax2.plot(iters_nu, nu_hist, 's--', color=color_nu,
                     markersize=5, linewidth=1.5, label=r"$\nu$")
            ax2.set_ylabel(r"$\nu$ (degrees of freedom)", fontsize=12, color=color_nu)
            vals_nu = np.array(nu_hist)
            vals_nu_pos = vals_nu[vals_nu > 0]
            if len(vals_nu_pos) >= 2 and vals_nu_pos.max() / vals_nu_pos.min() > 100:
                ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor=color_nu)
    else:
        ax2 = None

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
    else:
        ax1.legend(fontsize=10)

    n_curves = getattr(result, 'n_curves', '?')
    final_nu = nu_hist[-1] if has_nu else '?'
    final_sigma = sigma_hist[-1] if has_sigma else '?'
    # Compute final nuN for the title
    final_nuN = final_nu * N_grid if (isinstance(final_nu, float) and N_grid is not None) else None
    if isinstance(final_nu, float):
        if final_nuN is not None:
            title_str = (
                f"Student-t Hyperparameter Convergence — "
                f"{_format_group_key_title(result.group_key)}"
                f"  ({n_curves} curves)\n"
                f"Final: $\\sigma^2_{{\\mathrm{{btw}}}}$={final_sigma:.4g}, "
                f"$\\nu$={final_nu:.4f}, $\\nu N$={final_nuN:.1f}"
            )
        else:
            title_str = (
                f"Student-t Hyperparameter Convergence — "
                f"{_format_group_key_title(result.group_key)}"
                f"  ({n_curves} curves)\n"
                f"Final: $\\sigma^2_{{\\mathrm{{btw}}}}$={final_sigma:.4g}, "
                f"$\\nu$={final_nu:.4f}"
            )
    else:
        title_str = (
            f"Student-t Hyperparameter Convergence — "
            f"{_format_group_key_title(result.group_key)}"
            f"  ({n_curves} curves)"
        )
    ax1.set_title(title_str, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved Student-t hyperparameter history plot to {output_path}")
    return output_path
