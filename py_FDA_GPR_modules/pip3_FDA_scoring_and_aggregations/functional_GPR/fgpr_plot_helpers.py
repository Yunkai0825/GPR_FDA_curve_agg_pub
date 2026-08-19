# functional_GPR/fgpr_plot_helpers.py
"""
Plot helpers for Functional GPR (FGPR) aggregation.

Handles:
- FGPR aggregated curve plot (mean + CI bands, real + normalized)
- FGPR weight distribution plot
- FGPR covariance heatmap (aggregated + predictive)
- FGPR diagonal std profile
- FGPR sigma calibration (deviation scatter + 1σ/2σ bands)

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

from ..summary_gpr_plotting import plot_weight_distribution
from ...pip0_dataloading.filename_parser import format_group_key_title

if TYPE_CHECKING:
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult
    from ..summary_gpr_loader import IndividualGPRData

# Alias for internal use
_format_group_key_title = format_group_key_title


# =====================================================================
# FGPR Aggregated Curve Plot
# =====================================================================

def plot_fgpr_curve(
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
    """
    Plot FGPR aggregated curve with CI bands (real + normalized panels).

    Overlays individual GPR curves if *gprs* is provided.
    Uses predictive std (C_agg + σ_btw² I) for the CI bands.

    The companion CSV is ``FGPR_Curve_{safe_key}.csv`` produced by
    ``export_fgpr_curve_csv``.
    """
    fgpr_mean = getattr(result, "fgpr_mean", None)
    fgpr_std_pred = getattr(result, "fgpr_std_predictive", None)
    if fgpr_mean is None or fgpr_std_pred is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')

    z = sp_norm.ppf(0.5 + confidence_level / 2)
    x = result.x_pred_original
    title_str = _format_group_key_title(result.group_key)

    # Auto-derive max_time_cap from data when not explicitly set
    if max_time_cap is None:
        max_time_cap = float(x.max())
        if gprs:
            for gpr in gprs:
                x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
                max_time_cap = max(max_time_cap, float(x_ind.max()))
        max_time_cap *= 1.05  # 5% padding

    # --- Determine layout (1 or 2 panels) ---
    has_norm = (getattr(result, "fgpr_mean_norm", None) is not None
                and getattr(result, "fgpr_std_predictive_norm", None) is not None)
    n_panels = 2 if has_norm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ----- Panel 1: Real scale -----
    ax = axes[0]
    mean_r = result.fgpr_mean
    std_pred_r = result.fgpr_std_predictive
    fgpr_std_r = result.fgpr_std

    # Individual curves
    if gprs:
        for gpr in gprs:
            x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
            ax.plot(x_ind, gpr.y_pred, color='gray', alpha=individual_alpha, linewidth=0.8)

    # Predictive CI band (wider)
    ax.fill_between(x, mean_r - z * std_pred_r, mean_r + z * std_pred_r,
                    color='C3', alpha=0.10, label=f'Predictive {confidence_level*100:.0f}% CI')
    # Posterior CI band (tighter)
    ax.fill_between(x, mean_r - z * fgpr_std_r, mean_r + z * fgpr_std_r,
                    color='C3', alpha=0.25, label=f'Posterior {confidence_level*100:.0f}% CI')
    ax.plot(x, mean_r, 'C3-', linewidth=1.8, label='FGPR mean')

    ax.set_xscale('log')
    ax.set_xlim([min_time_cap, max_time_cap])
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title(f"FGPR (obs. scale) — {title_str}  ({result.n_curves} curves)")
    ax.legend(loc='best', fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)

    # ----- Panel 2: Normalized scale -----
    if has_norm:
        ax2 = axes[1]
        mean_n = result.fgpr_mean_norm
        std_pred_n = result.fgpr_std_predictive_norm
        fgpr_std_n = result.fgpr_std_norm

        if gprs:
            for gpr in gprs:
                x_ind = gpr.x_scaling.inverse_transform(gpr.x_pred_transformed)
                ax2.plot(x_ind, gpr.y_pred_normalized, color='gray',
                         alpha=individual_alpha, linewidth=0.8)

        ax2.fill_between(x, mean_n - z * std_pred_n, mean_n + z * std_pred_n,
                         color='C4', alpha=0.10, label=f'Predictive {confidence_level*100:.0f}% CI')
        ax2.fill_between(x, mean_n - z * fgpr_std_n, mean_n + z * fgpr_std_n,
                         color='C4', alpha=0.25, label=f'Posterior {confidence_level*100:.0f}% CI')
        ax2.plot(x, mean_n, 'C4-', linewidth=1.8, label='FGPR mean (norm)')

        ax2.set_xscale('log')
        ax2.set_xlim([min_time_cap, max_time_cap])
        ax2.set_xlabel(x_axis_label)
        ax2.set_ylabel(f"{y_axis_label} (normalized)")
        ax2.set_title(f"FGPR (norm. scale) — {title_str}")
        ax2.legend(loc='best', fontsize='small')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)

    output_path = output_dir / f'FGPR_Summary_{safe_key}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved FGPR curve plot to {output_path}")
    return output_path


# =====================================================================
# FGPR Sigma Calibration
# =====================================================================

def plot_fgpr_sigma_calibration(
    result: "SummaryGPRResult",
    output_dir: Path,
    gprs: Optional[List["IndividualGPRData"]] = None,
    *,
    x_axis_label: str = "Time (s)",
    y_axis_label: str = "Current (A/cm²)",
    verbose: bool = True,
    dpi: int = 200,
) -> Optional[Path]:
    """
    FGPR sigma calibration: deviation scatter with 1σ/2σ bands.

    Produces a two-panel figure:
      Left:  real scale (uses predictive std)
      Right: normalized scale (uses predictive std)

    Also exports a companion CSV ``FGPR_Sigma_Calibration_{key}.csv``
    via ``export_fgpr_sigma_calibration_csv`` (called separately by
    the orchestrator).
    """
    fgpr_mean = getattr(result, "fgpr_mean", None)
    fgpr_std_pred = getattr(result, "fgpr_std_predictive", None)
    if fgpr_mean is None or fgpr_std_pred is None:
        return None
    if gprs is None or len(gprs) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    x = result.x_pred_original

    # Interpolate individual curves onto the common grid
    y_real_stack = _interpolate_individual_real(gprs, x)
    y_norm_stack = _interpolate_individual_norm(gprs, x)
    if y_real_stack is None:
        return None
    y_real_arr = np.vstack(y_real_stack)

    # --- Determine layout ---
    fgpr_mean_n = getattr(result, "fgpr_mean_norm", None)
    fgpr_std_pred_n = getattr(result, "fgpr_std_predictive_norm", None)
    has_norm = (fgpr_mean_n is not None and fgpr_std_pred_n is not None
                and y_norm_stack is not None)
    n_panels = 2 if has_norm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    # ----- Panel 1: Real scale -----
    dev_real = y_real_arr - fgpr_mean
    cov1_r = float(np.mean((dev_real >= -fgpr_std_pred) & (dev_real <= fgpr_std_pred)))
    cov2_r = float(np.mean((dev_real >= -2 * fgpr_std_pred) & (dev_real <= 2 * fgpr_std_pred)))

    ax0 = axes[0]
    ax0.axhline(0, color='k', linewidth=1.0, linestyle='--')
    ax0.fill_between(x, -2 * fgpr_std_pred, 2 * fgpr_std_pred,
                     color='C3', alpha=0.10, label='FGPR 2σ (pred)')
    ax0.fill_between(x, -fgpr_std_pred, fgpr_std_pred,
                     color='C3', alpha=0.22, label='FGPR 1σ (pred)')
    ax0.scatter(np.tile(x, len(y_real_stack)), dev_real.flatten(),
                s=6, alpha=0.08, color='gray', label='Samples')
    ax0.set_xscale('log')
    ax0.set_xlabel(x_axis_label)
    ax0.set_ylabel(f"Deviation ({y_axis_label})")
    ax0.set_title(f"FGPR σ calibration (real): {_format_group_key_title(result.group_key)}")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
    txt0 = [f"Coverage (pred, real):",
            f"  1σ: {cov1_r*100:.1f}%",
            f"  2σ: {cov2_r*100:.1f}%"]
    ax0.text(0.02, 0.98, "\n".join(txt0), transform=ax0.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax0.legend(loc='best', fontsize='small')

    # ----- Panel 2: Normalized scale -----
    cov1_n = cov2_n = None
    if has_norm:
        y_norm_arr = np.vstack(y_norm_stack)
        dev_norm = y_norm_arr - fgpr_mean_n
        cov1_n = float(np.mean((dev_norm >= -fgpr_std_pred_n) & (dev_norm <= fgpr_std_pred_n)))
        cov2_n = float(np.mean((dev_norm >= -2 * fgpr_std_pred_n) & (dev_norm <= 2 * fgpr_std_pred_n)))

        ax1 = axes[1]
        ax1.axhline(0, color='k', linewidth=1.0, linestyle='--')
        ax1.fill_between(x, -2 * fgpr_std_pred_n, 2 * fgpr_std_pred_n,
                         color='C4', alpha=0.10, label='FGPR 2σ (pred, norm)')
        ax1.fill_between(x, -fgpr_std_pred_n, fgpr_std_pred_n,
                         color='C4', alpha=0.22, label='FGPR 1σ (pred, norm)')
        ax1.scatter(np.tile(x, len(y_norm_stack)), dev_norm.flatten(),
                    s=6, alpha=0.08, color='gray', label='Samples (norm)')
        ax1.set_xscale('log')
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylabel(f"Deviation ({y_axis_label}, norm)")
        ax1.set_title(f"FGPR σ calibration (norm): {_format_group_key_title(result.group_key)}")
        ax1.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
        txt1 = [f"Coverage (pred, norm):",
                f"  1σ: {cov1_n*100:.1f}%",
                f"  2σ: {cov2_n*100:.1f}%"]
        ax1.text(0.02, 0.98, "\n".join(txt1), transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax1.legend(loc='best', fontsize='small')

    output_path = output_dir / f'FGPR_Sigma_Calibration_{safe_key}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    if verbose:
        print(f"  Saved FGPR sigma calibration to {output_path}")
    return output_path


# =====================================================================
# Interpolation helpers (for sigma calibration)
# =====================================================================

def _interpolate_individual_real(
    gprs: List["IndividualGPRData"], x: np.ndarray
) -> Optional[List[np.ndarray]]:
    """Interpolate individual GPR real-scale predictions onto common x grid."""
    y_stack = []
    for gpr in gprs:
        f_y = interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            gpr.y_pred,
            kind='linear', fill_value='extrapolate',
        )
        y_stack.append(f_y(x))
    return y_stack if y_stack else None


def _interpolate_individual_norm(
    gprs: List["IndividualGPRData"], x: np.ndarray
) -> Optional[List[np.ndarray]]:
    """Interpolate individual GPR normalized predictions onto common x grid."""
    y_stack = []
    for gpr in gprs:
        y_norm = getattr(gpr, 'y_pred_normalized', None)
        if y_norm is None:
            return None
        f_y = interp1d(
            gpr.x_scaling.inverse_transform(gpr.x_pred_transformed),
            y_norm,
            kind='linear', fill_value='extrapolate',
        )
        y_stack.append(f_y(x))
    return y_stack if y_stack else None


# =====================================================================
# FGPR Weight Distribution
# =====================================================================

def plot_fgpr_weight_distribution(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
) -> Optional[Path]:
    """Plot FGPR weight distribution if weights are available."""
    if result.fgpr_weights is None or len(result.fgpr_weights) <= 1:
        return None

    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    fgpr_weight_dist_path = output_dir / f'FGPR_Weight_Distribution_{safe_key}.png'

    return plot_weight_distribution(
        weights=result.fgpr_weights,
        output_path=fgpr_weight_dist_path,
        group_key=result.group_key + " (FGPR)",
        verbose=verbose,
    )


def plot_fgpr_covariance_heatmap(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot FGPR aggregated and predictive covariance heatmaps side-by-side.

    Produces a two-panel figure:
      Left:  C_agg  (posterior covariance, normalized scale)
      Right: C_pred = C_agg + sigma_btw^2 I  (predictive covariance)

    Uses SymLogNorm colour scaling capped at 6 decades below the
    maximum diagonal value to avoid jitter from very small entries.

    Parameters
    ----------
    result : SummaryGPRResult
        Must have ``fgpr_cov_norm`` populated.
    output_dir : Path
        Destination directory (plot saved as PNG).
    verbose : bool
        Print save path.
    dpi : int
        Resolution for the output figure.

    Returns
    -------
    Path or None
        Saved figure path, or None if covariance is not available.
    """
    C_agg = getattr(result, "fgpr_cov_norm", None)
    if C_agg is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = output_dir / f'FGPR_Covariance_{safe_key}.png'

    # Build panel list: always C_agg; add C_pred if available
    C_pred = getattr(result, "fgpr_cov_predictive_norm", None)
    panels = [(C_agg, r"$C_{\mathrm{agg}}$ (normalized)")]
    if C_pred is not None:
        panels.append((C_pred, r"$C_{\mathrm{pred}}$ (predictive, norm)"))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    # Shared colour scale: 6 decades below max diagonal
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
        f"FGPR Covariance — {_format_group_key_title(result.group_key)}"
        f"  ({result.n_curves} curves)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved FGPR covariance heatmap to {output_path}")
    return output_path


def plot_fgpr_diagonal_std(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot sqrt(diag(C)) profiles for aggregated and predictive covariances.

    Overlays:
      - sqrt(diag(C_agg))   — posterior std
      - sqrt(diag(C_pred))  — predictive std (if available)

    Parameters
    ----------
    result : SummaryGPRResult
        Must have ``fgpr_cov_norm`` populated.
    output_dir : Path
    verbose : bool
    dpi : int

    Returns
    -------
    Path or None
    """
    C_agg = getattr(result, "fgpr_cov_norm", None)
    if C_agg is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = output_dir / f'FGPR_DiagStd_{safe_key}.png'

    fig, ax = plt.subplots(figsize=(8, 4.5))

    diag_agg = np.sqrt(np.clip(np.diag(C_agg), 0, None))
    ax.plot(diag_agg, label=r"$\sqrt{\mathrm{diag}(C_{\mathrm{agg}})}$",
            linewidth=1.5, color="tab:blue")

    C_pred = getattr(result, "fgpr_cov_predictive_norm", None)
    if C_pred is not None:
        diag_pred = np.sqrt(np.clip(np.diag(C_pred), 0, None))
        ax.plot(diag_pred, label=r"$\sqrt{\mathrm{diag}(C_{\mathrm{pred}})}$",
                linewidth=1.5, color="tab:orange", linestyle="--")

    sigma_btw2 = getattr(result, "fgpr_sigma_btw_squared", None)
    if sigma_btw2 is not None:
        ax.axhline(np.sqrt(sigma_btw2), color="tab:red", linestyle=":",
                   linewidth=1, label=r"$\sigma_{\mathrm{btw}}$")

    ax.set_xlabel("Grid index $j$", fontsize=11)
    ax.set_ylabel("Std (normalized)", fontsize=11)
    ax.set_title(
        f"FGPR Posterior Std Profile — "
        f"{_format_group_key_title(result.group_key)}"
        f"  ({result.n_curves} curves)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved FGPR diagonal std profile to {output_path}")
    return output_path


# =====================================================================
# FGPR Weight Convergence Plot
# =====================================================================

def plot_fgpr_weight_convergence(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot FGPR structured weight convergence across outer iterations.

    Shows per-curve weights vs iteration number, analogous to
    ``plot_weight_convergence_from_csv`` in iterative GPR.

    Returns path to saved PNG or None if no history available.
    """
    wh = getattr(result, 'fgpr_weight_history', None)
    if not wh or len(wh) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = Path(output_dir) / f'FGPR_Weight_Convergence_{safe_key}.png'

    weight_arr = np.array(wh)  # (n_iters, R)
    n_iters, n_curves = weight_arr.shape
    iterations = np.arange(1, n_iters + 1)

    sample_ids = getattr(result, 'sample_ids', None)

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in range(n_curves):
        label = sample_ids[r] if (sample_ids and r < len(sample_ids)) else f'curve {r}'
        ax.plot(iterations, weight_arr[:, r], 'o-', markersize=4, label=label)

    ax.set_xlabel("Outer Iteration", fontsize=11)
    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title(
        f"FGPR Structured Weight Convergence — "
        f"{_format_group_key_title(result.group_key)}"
        f"  ({n_curves} curves)",
        fontsize=12,
    )
    ax.legend(fontsize=8, ncol=max(1, n_curves // 4))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n_iters + 0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved FGPR weight convergence plot to {output_path}")
    return output_path


def plot_fgpr_curve_iterations(
    result: "SummaryGPRResult",
    output_dir: Path,
    *,
    verbose: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot FGPR aggregated mean curve at each outer iteration.

    Useful to see how the aggregated curve evolves across
    weight-convergence iterations.
    """
    ch = getattr(result, 'fgpr_curve_history', None)
    if not ch or len(ch) < 2:
        return None

    os.makedirs(output_dir, exist_ok=True)
    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    output_path = Path(output_dir) / f'FGPR_Curve_Iterations_{safe_key}.png'

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
    ax.set_ylabel("Aggregated mean (normalized)", fontsize=11)
    ax.set_title(
        f"FGPR Aggregated Curve Iterations — "
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
        print(f"  Saved FGPR curve iterations plot to {output_path}")
    return output_path
