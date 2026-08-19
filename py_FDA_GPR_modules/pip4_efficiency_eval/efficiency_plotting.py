# pip4_efficiency_eval/efficiency_plotting.py
"""
Plotting utilities for Efficiency Evaluation.

Design pattern:
- Core functions: Take DataFrames directly for in-pipeline use
- CSV wrappers: Load from CSV and call core functions for post-hoc plotting
- Multi-method comparison: Overlay learning curves across methods
- Diagnostic plots: sigma_btw and covariance matrix visualisation

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import SymLogNorm

from .efficiency_config import ScaleParams, PlotParams
from ..pip0_dataloading.filename_parser import format_group_key_title

# Alias for internal use (keeps existing code working)
_format_group_key_title = format_group_key_title


# =============================================================================
# Aggregation Utilities
# =============================================================================

def aggregate_detailed_to_summary(df_detailed: pd.DataFrame, q_low: float = 0.25, q_high: float = 0.75) -> pd.DataFrame:
    """
    Aggregate detailed MC results to summary statistics per subset_size.
    
    Parameters
    ----------
    df_detailed : pd.DataFrame
        Detailed DataFrame with columns: subset_size, mc_index, error, time_s, n_iterations.
    q_low : float
        Lower quantile (default 0.25).
    q_high : float
        Upper quantile (default 0.75).
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per subset_size.
    """
    if df_detailed.empty:
        return pd.DataFrame()
    
    records: list[dict] = []
    for subset_size, g in df_detailed.groupby("subset_size"):
        n = len(g)
        ss = g["subset_size"].iloc[0]
        error_var = g["error"].var(ddof=1) if n > 1 else 0.0
        time_var = g["time_s"].var(ddof=1) if n > 1 else 0.0
        iter_std = g["n_iterations"].std(ddof=1) if n > 1 else 0.0
        rec = {
            "subset_size": ss,
            # Error statistics
            "mean_error": g["error"].mean(),
            "median_error": g["error"].median(),
            "variance_error": error_var,
            "q_error_low": g["error"].quantile(q_low),
            "q_error_high": g["error"].quantile(q_high),
            # Time statistics
            "avg_subset_time[s]": g["time_s"].mean(),
            "avg_time_per_elem[s]": g["time_s"].mean() / ss,
            "median_subset_time[s]": g["time_s"].median(),
            "variance_subset_time[s]": time_var,
            "q_subset_time_low[s]": g["time_s"].quantile(q_low),
            "q_subset_time_high[s]": g["time_s"].quantile(q_high),
            # Iteration statistics
            "mean_iterations": g["n_iterations"].mean(),
            "median_iterations": g["n_iterations"].median(),
            "min_iterations": g["n_iterations"].min(),
            "max_iterations": g["n_iterations"].max(),
            "std_iterations": iter_std,
            # Count
            "n_mc_runs": n,
        }
        # Between-model variance statistics (sigma_btw)
        if "sigma_btw" in g.columns and g["sigma_btw"].notna().any():
            sbtw = g["sigma_btw"].dropna()
            sbtw_std = sbtw.std(ddof=1) if len(sbtw) > 1 else 0.0
            rec.update({
                "mean_sigma_btw": sbtw.mean(),
                "median_sigma_btw": sbtw.median(),
                "std_sigma_btw": sbtw_std,
                "q_sigma_btw_low": sbtw.quantile(q_low),
                "q_sigma_btw_high": sbtw.quantile(q_high),
                "min_sigma_btw": sbtw.min(),
                "max_sigma_btw": sbtw.max(),
            })
        records.append(rec)
    
    summary = pd.DataFrame(records)
    return summary


def load_detailed_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load detailed CSV file."""
    return pd.read_csv(csv_path)


# =============================================================================
# Core Plotting Functions (take DataFrames directly for in-pipeline use)
# =============================================================================

def plot_learning_curve(
    df_summary: pd.DataFrame,
    group_key: str,
    output_path: Path,
    *,
    scapara: Optional[ScaleParams] = None,
    plotpara: Optional[PlotParams] = None,
    summary_config_info: Optional[dict] = None,
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for learning curve from summary DataFrame.
    
    Parameters
    ----------
    df_summary : pd.DataFrame
        Aggregated summary DataFrame with columns: subset_size, median_error, mean_error, etc.
    group_key : str
        Group key string for title (e.g., "potential=-1.95").
    output_path : Path
        Full path for output file (including filename).
    scapara : ScaleParams, optional
        Scaling parameters.
    plotpara : PlotParams, optional
        Plot parameters.
    summary_config_info : dict, optional
        Summary GPR config info for subtitle.
    verbose : bool
        Print status.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    # Use defaults if not provided
    scapara = scapara or ScaleParams()
    plotpara = plotpara or PlotParams()
    
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    metric = "RMSE"
    normalize_error = scapara.normalize_w_rbar
    use_log_error = scapara.use_log_error
    q_low = 0.25
    q_high = 0.75
    
    # Build y-axis label with unit info
    if use_log_error:
        error_unit = f"log₁₀({metric})"
    else:
        error_unit = metric
    if normalize_error:
        error_unit += " (normalized)"
    else:
        error_unit += " (obs. scale)"
    
    fig, axes = plt.subplots(4, 1, figsize=(8, 13), sharex=False)
    ax1, ax2, ax3, ax4 = axes
    
    gA = df_summary.set_index("subset_size") if not df_summary.empty else pd.DataFrame()
    
    # === Panel 1: Error vs Number of Curves ===
    if not df_summary.empty and "median_error" in gA.columns:
        ax1.plot(gA.index, gA["median_error"], "-o", color="#1f77b4", label="MC median")
        ax1.plot(gA.index, gA["mean_error"], "--o", color="#1f77b4", alpha=0.6, label="MC mean")
        
        if "q_error_low" in gA.columns and "q_error_high" in gA.columns:
            ax1.fill_between(
                gA.index, gA["q_error_low"], gA["q_error_high"],
                color="#1f77b4", alpha=0.15, label="MC IQR"
            )
        
        if "variance_error" in gA.columns:
            sigma = np.sqrt(gA["variance_error"])
            ax1.fill_between(
                gA.index, gA["mean_error"] - sigma, gA["mean_error"] + sigma,
                color="#1f77b4", alpha=0.08, label="MC ±1σ"
            )
    
    ax1.set_xlabel("# curves kept")
    ax1.set_ylabel(f"{error_unit} vs reference")
    
    title = f"{_format_group_key_title(group_key)} – Data Efficiency"
    if summary_config_info:
        title += f"\n{summary_config_info.get('subtitle', '')}"
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    
    # === Panel 2: Error vs CPU Time ===
    if not df_summary.empty and "median_subset_time[s]" in gA.columns:
        x_cent = gA["median_subset_time[s]"]
        y_cent = gA["median_error"]
        
        xerr = np.vstack([
            x_cent - gA["q_subset_time_low[s]"],
            gA["q_subset_time_high[s]"] - x_cent
        ])
        yerr = np.vstack([
            y_cent - gA["q_error_low"],
            gA["q_error_high"] - y_cent
        ])
        
        ax2.errorbar(
            x_cent, y_cent, xerr=xerr, yerr=yerr,
            fmt="o", color="#1f77b4", ecolor="#1f77b4",
            alpha=0.8, capsize=3,
            label=f"MC median [{q_low:.2f},{q_high:.2f}]"
        )
    
    ax2.set_xlabel("Average CPU-time per subset [s]")
    ax2.set_ylabel(f"{error_unit} vs reference")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    # === Panel 3: Iterations vs Number of Curves ===
    if not df_summary.empty and "mean_iterations" in gA.columns:
        ax3.plot(gA.index, gA["median_iterations"], "-s", color="#2ca02c", label="Median iterations")
        ax3.plot(gA.index, gA["mean_iterations"], "--s", color="#2ca02c", alpha=0.6, label="Mean iterations")
        
        if "min_iterations" in gA.columns and "max_iterations" in gA.columns:
            ax3.fill_between(
                gA.index, gA["min_iterations"], gA["max_iterations"],
                color="#2ca02c", alpha=0.15, label="Min-Max range"
            )
        
        if "std_iterations" in gA.columns:
            ax3.fill_between(
                gA.index,
                gA["mean_iterations"] - gA["std_iterations"],
                gA["mean_iterations"] + gA["std_iterations"],
                color="#2ca02c", alpha=0.08, label="±1σ"
            )
    
    ax3.set_xlabel("# curves kept")
    ax3.set_ylabel("Number of iterations")
    ax3.set_title("Convergence Iterations vs Subset Size")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")
    
    # === Panel 4: Between-Model Variance (sigma_btw) vs Number of Curves ===
    if not df_summary.empty and "median_sigma_btw" in gA.columns:
        ax4.plot(gA.index, gA["median_sigma_btw"], "-D", color="#d62728", label="MC median")
        ax4.plot(gA.index, gA["mean_sigma_btw"], "--D", color="#d62728", alpha=0.6, label="MC mean")
        
        if "q_sigma_btw_low" in gA.columns and "q_sigma_btw_high" in gA.columns:
            ax4.fill_between(
                gA.index, gA["q_sigma_btw_low"], gA["q_sigma_btw_high"],
                color="#d62728", alpha=0.15, label="MC IQR"
            )
        
        if "std_sigma_btw" in gA.columns:
            sigma = gA["std_sigma_btw"]
            ax4.fill_between(
                gA.index,
                gA["mean_sigma_btw"] - sigma,
                gA["mean_sigma_btw"] + sigma,
                color="#d62728", alpha=0.08, label=r"MC $\pm 1\sigma$"
            )
        
        if "min_sigma_btw" in gA.columns and "max_sigma_btw" in gA.columns:
            ax4.scatter(gA.index, gA["min_sigma_btw"], marker="v", color="#d62728",
                       alpha=0.3, s=15, zorder=3)
            ax4.scatter(gA.index, gA["max_sigma_btw"], marker="^", color="#d62728",
                       alpha=0.3, s=15, zorder=3)
    else:
        ax4.text(0.5, 0.5, r"$\sigma_{\mathrm{btw}}^2$ not available",
                 transform=ax4.transAxes, ha="center", va="center",
                 fontsize=12, color="gray")
    
    ax4.set_xlabel("# curves kept")
    ax4.set_ylabel(r"$\sigma_{\mathrm{btw}}^2$ (between-model variance, obs. scale)")
    ax4.set_title("Between-Model Variance vs Subset Size")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")
    
    fig.tight_layout()
    
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"    -> Saved learning curve plot: {output_path}")
    
    return output_path


def plot_iteration_statistics(
    df_summary: pd.DataFrame,
    group_key: str,
    output_path: Path,
    *,
    verbose: bool = True,
) -> Path:
    """
    Core plotting function for iteration statistics from summary DataFrame.
    
    Parameters
    ----------
    df_summary : pd.DataFrame
        Aggregated summary DataFrame with iteration columns.
    group_key : str
        Group key string for title (e.g., "potential=-1.95").
    output_path : Path
        Full path for output file (including filename).
    verbose : bool
        Print status.
        
    Returns
    -------
    Path
        Path to saved plot.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gA = df_summary.set_index("subset_size")
    
    if "mean_iterations" in gA.columns:
        ax.errorbar(
            gA.index,
            gA["mean_iterations"],
            yerr=gA["std_iterations"] if "std_iterations" in gA.columns else None,
            fmt="-o",
            color="#2ca02c",
            capsize=4,
            label="Mean ± Std"
        )
        
        if "min_iterations" in gA.columns:
            ax.scatter(gA.index, gA["min_iterations"], marker="v", color="#d62728", 
                      alpha=0.7, label="Min", zorder=5)
        if "max_iterations" in gA.columns:
            ax.scatter(gA.index, gA["max_iterations"], marker="^", color="#9467bd", 
                      alpha=0.7, label="Max", zorder=5)
    
    ax.set_xlabel("Subset Size (# curves)")
    ax.set_ylabel("Number of Iterations to Converge")
    ax.set_title(f"Iteration Statistics – {_format_group_key_title(group_key)}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if verbose:
        print(f"    -> Saved iteration statistics plot: {output_path}")
    
    return output_path


# =============================================================================
# CSV Wrapper Functions (load from CSV and call core functions)
# =============================================================================

def plot_learning_curve_from_detailed(
    detailed_csv: Union[str, Path, pd.DataFrame],
    group_key: str,
    output_dir: Path,
    *,
    scapara: Optional[ScaleParams] = None,
    plotpara: Optional[PlotParams] = None,
    summary_config_info: Optional[dict] = None,
    verbose: bool = True,
) -> Path:
    """
    CSV wrapper: Plot learning curve from detailed CSV.
    
    Loads data from CSV, aggregates to summary, and calls the core `plot_learning_curve` function.
    """
    # Load data
    if isinstance(detailed_csv, (str, Path)):
        df_detailed = load_detailed_csv(detailed_csv)
    else:
        df_detailed = detailed_csv
    
    # Aggregate to summary
    df_summary = aggregate_detailed_to_summary(df_detailed)
    
    # Build output path
    output_path = Path(output_dir) / f"Efficiency_LearningCurve_{group_key}.png"
    
    # Call core function
    return plot_learning_curve(
        df_summary=df_summary,
        group_key=group_key,
        output_path=output_path,
        scapara=scapara,
        plotpara=plotpara,
        summary_config_info=summary_config_info,
        verbose=verbose,
    )


def plot_iteration_statistics_from_detailed(
    detailed_csv: Union[str, Path, pd.DataFrame],
    group_key: str,
    output_dir: Path,
    *,
    verbose: bool = True,
) -> Path:
    """
    CSV wrapper: Plot iteration statistics from detailed CSV.
    
    Loads data from CSV, aggregates to summary, and calls the core `plot_iteration_statistics` function.
    """
    # Load data
    if isinstance(detailed_csv, (str, Path)):
        df_detailed = load_detailed_csv(detailed_csv)
    else:
        df_detailed = detailed_csv
    
    # Aggregate to summary
    df_summary = aggregate_detailed_to_summary(df_detailed)
    
    # Build output path
    output_path = Path(output_dir) / f"Iteration_Statistics_{group_key}.png"
    
    # Call core function
    return plot_iteration_statistics(
        df_summary=df_summary,
        group_key=group_key,
        output_path=output_path,
        verbose=verbose,
    )


# =============================================================================
# Multi-Method Comparison Plotting
# =============================================================================

# Default palette for up to 5 methods
_PALETTE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f"]
_PALETTE_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "h"]


def _aggregate_detailed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-MC-run detailed results to per-subset-size stats."""
    records: list[dict] = []
    for ss, g in df.groupby("subset_size"):
        records.append({
            "subset_size": int(ss),
            "mean_error": g["error"].mean(),
            "median_error": g["error"].median(),
            "q25_error": g["error"].quantile(0.25),
            "q75_error": g["error"].quantile(0.75),
            "mean_time": g["time_s"].mean(),
            "median_time": g["time_s"].median(),
            "q25_time": g["time_s"].quantile(0.25),
            "q75_time": g["time_s"].quantile(0.75),
            "n_mc": len(g),
        })
    return pd.DataFrame(records).sort_values("subset_size")


def plot_multimethod_comparison(
    method_detailed: Dict[str, pd.DataFrame],
    group_key: str,
    output_path: Path,
    *,
    scapara: Optional[ScaleParams] = None,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Create a 3-panel comparison figure (error, cost-accuracy, time) for one
    group, overlaying multiple methods.

    Parameters
    ----------
    method_detailed : dict
        ``{method_label: detailed_dataframe}`` – one detailed DataFrame per
        method.
    group_key : str
        Group key for the plot title.
    output_path : Path
        Full path (including filename) for the saved figure.
    scapara : ScaleParams, optional
        Scaling parameters (for y-axis label).
    plotpara : PlotParams, optional
        Plot parameters.
    verbose : bool
        Print save path.

    Returns
    -------
    Path
    """
    scapara = scapara or ScaleParams()
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_err, ax_cost, ax_time = axes

    for idx, (label, df_det) in enumerate(method_detailed.items()):
        agg = _aggregate_detailed(df_det)
        c = _PALETTE_COLORS[idx % len(_PALETTE_COLORS)]
        m = _PALETTE_MARKERS[idx % len(_PALETTE_MARKERS)]

        # Panel 1 — error vs # curves
        ax_err.plot(agg["subset_size"], agg["median_error"],
                    marker=m, color=c, label=label, linewidth=1.5, markersize=6)
        ax_err.fill_between(agg["subset_size"], agg["q25_error"],
                            agg["q75_error"], color=c, alpha=0.12)

        # Panel 2 — error vs CPU time
        ax_cost.plot(agg["median_time"], agg["median_error"],
                     marker=m, color=c, label=label, linewidth=1.5, markersize=6)
        xerr = np.vstack([agg["median_time"] - agg["q25_time"],
                          agg["q75_time"] - agg["median_time"]])
        yerr = np.vstack([agg["median_error"] - agg["q25_error"],
                          agg["q75_error"] - agg["median_error"]])
        ax_cost.errorbar(agg["median_time"], agg["median_error"],
                         xerr=xerr, yerr=yerr, fmt="none",
                         ecolor=c, alpha=0.35, capsize=2)

        # Panel 3 — time vs # curves
        ax_time.plot(agg["subset_size"], agg["median_time"] * 1000,
                     marker=m, color=c, label=label, linewidth=1.5, markersize=6)
        ax_time.fill_between(agg["subset_size"],
                             agg["q25_time"] * 1000, agg["q75_time"] * 1000,
                             color=c, alpha=0.12)

    # Formatting
    y_label = ("log₁₀(RMSE, obs. scale)" if not scapara.normalize_w_rbar
               else "log₁₀(normalised RMSE)") if scapara.use_log_error else "RMSE (obs. scale)"
    ax_err.set(xlabel="# curves in subset", ylabel=y_label,
               title="Learning Curve: Error vs Subset Size")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=8.5, loc="upper right")
    ax_err.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_cost.set(xlabel="Median CPU time (s)", ylabel=y_label,
                title="Cost–Accuracy Trade-off")
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=8.5, loc="upper right")

    ax_time.set(xlabel="# curves in subset", ylabel="Median CPU time (ms)",
                title="Computational Cost")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8.5, loc="upper left")
    ax_time.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle(f"Multi-Method Efficiency Comparison — "
                 f"{_format_group_key_title(group_key)}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved comparison plot: {output_path}")
    return output_path


def plot_multimethod_bar_summary(
    comparison_df: pd.DataFrame,
    output_path: Path,
    *,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Bar chart of min-mean-error per method per group from a comparison
    DataFrame with columns ``method, group_key, min_mean_error``.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison table (as produced by the unified DEBUG script).
    output_path : Path
        Destination path for the saved figure.
    plotpara : PlotParams, optional
    verbose : bool

    Returns
    -------
    Path
    """
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    groups = comparison_df["group_key"].unique()
    methods = comparison_df["method"].unique()
    n_groups, n_methods = len(groups), len(methods)

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.14
    x = np.arange(n_groups)

    for i, method in enumerate(methods):
        vals = []
        for g in groups:
            row = comparison_df[
                (comparison_df["method"] == method) &
                (comparison_df["group_key"] == g)
            ]
            vals.append(row["min_mean_error"].values[0] if len(row) else 0)
        label = method.split("_", 1)[-1].replace("_", " ").title()
        ax.bar(x + i * width, vals, width, label=label,
               color=_PALETTE_COLORS[i % len(_PALETTE_COLORS)])

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels([g.replace("|", "\n") for g in groups], fontsize=9)
    ax.set_ylabel("Min mean error (log₁₀ RMSE, obs. scale)", fontsize=11)
    ax.set_title("Best Achievable Error by Method (at smallest subset)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved bar summary: {output_path}")
    return output_path


# =============================================================================
# sigma_btw and Covariance Diagnostic Plots
# =============================================================================

def plot_sigma_btw_comparison(
    method_detailed: Dict[str, pd.DataFrame],
    group_key: str,
    output_path: Path,
    *,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Overlay sigma_btw vs subset_size for multiple methods.

    Parameters
    ----------
    method_detailed : dict
        ``{method_label: detailed_df}`` — each DataFrame must contain a
        ``sigma_btw`` column.
    group_key : str
        Group key for the plot title.
    output_path : Path
        Destination file path.
    plotpara : PlotParams, optional
    verbose : bool

    Returns
    -------
    Path
    """
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, df) in enumerate(method_detailed.items()):
        if "sigma_btw" not in df.columns:
            continue
        df_mc = df[df["error"] != 0.0].copy()
        if df_mc.empty:
            df_mc = df

        agg = df_mc.groupby("subset_size")["sigma_btw"].agg(
            ["median", "mean",
             lambda x: x.quantile(0.25),
             lambda x: x.quantile(0.75)]
        ).rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
        agg = agg.sort_index()

        c = _PALETTE_COLORS[idx % len(_PALETTE_COLORS)]
        m = _PALETTE_MARKERS[idx % len(_PALETTE_MARKERS)]
        ax.plot(agg.index, agg["median"], marker=m, color=c,
                label=label, linewidth=1.5, markersize=7)
        ax.fill_between(agg.index, agg["q25"], agg["q75"],
                        color=c, alpha=0.15)

    ax.set_xlabel("# curves in subset", fontsize=11)
    ax.set_ylabel(r"$\sigma_{\mathrm{btw}}^2$  (between-curve variance, obs. scale)",
                  fontsize=11)
    ax.set_title(f"Between-Curve Variance vs Subset Size\n"
                 f"{_format_group_key_title(group_key)}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved sigma_btw comparison: {output_path}")
    return output_path


def plot_covariance_heatmaps(
    cov_dict: Dict[int, np.ndarray],
    group_key: str,
    output_path: Path,
    *,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Heatmap of aggregated covariance C_agg at each subset size.

    Parameters
    ----------
    cov_dict : dict
        ``{subset_size: cov_matrix}`` — (N, N) arrays.
    group_key : str
        Group key for the plot title.
    output_path : Path
        Destination file path.
    plotpara : PlotParams, optional
    verbose : bool

    Returns
    -------
    Path
    """
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    ss_cov = sorted(cov_dict.items())
    n = len(ss_cov)
    if n == 0:
        return output_path

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    # Shared color range: 6 decades below max diagonal
    all_diag_max = max(np.max(np.abs(np.diag(C))) for _, C in ss_cov)
    vmax = all_diag_max
    linthresh = vmax * 1e-6

    for ax, (ss, C) in zip(axes, ss_cov):
        C_clipped = np.clip(C, -vmax, vmax)
        im = ax.imshow(C_clipped, aspect="auto", cmap="RdBu_r",
                       norm=SymLogNorm(linthresh=linthresh,
                                       vmin=-vmax, vmax=vmax),
                       origin="upper")
        ax.set_title(f"subset = {ss} curves", fontsize=10)
        ax.set_xlabel("grid index $j$", fontsize=9)
        ax.set_ylabel("grid index $k$", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8,
                     label=r"$C_{\mathrm{agg},jk}$")

    fig.suptitle(
        f"FGPR Aggregated Covariance $C_{{\\mathrm{{agg}}}}$ — "
        f"{_format_group_key_title(group_key)}",
        fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved covariance heatmaps: {output_path}")
    return output_path


def plot_covariance_diagonal(
    cov_dict: Dict[int, np.ndarray],
    group_key: str,
    output_path: Path,
    *,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Plot sqrt(diag(C_agg)) profiles overlaid for all subset sizes.

    Parameters
    ----------
    cov_dict : dict
        ``{subset_size: cov_matrix}``
    group_key : str
    output_path : Path
    plotpara : PlotParams, optional
    verbose : bool

    Returns
    -------
    Path
    """
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ss in sorted(cov_dict):
        C = cov_dict[ss]
        diag_sqrt = np.sqrt(np.clip(np.diag(C), 0, None))
        ax.plot(diag_sqrt, label=f"subset = {ss}", linewidth=1.2)
    ax.set_xlabel("Grid index $j$", fontsize=11)
    ax.set_ylabel(r"$\sqrt{\mathrm{diag}(C_{\mathrm{agg}})}$ (posterior std)",
                  fontsize=11)
    ax.set_title(f"FGPR Posterior Std Profile — "
                 f"{_format_group_key_title(group_key)}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved diagonal std profile: {output_path}")
    return output_path


def plot_pointwise_sigma_btw(
    pw_dict: Dict[int, np.ndarray],
    method_label: str,
    group_key: str,
    output_path: Path,
    *,
    plotpara: Optional[PlotParams] = None,
    verbose: bool = True,
) -> Path:
    """
    Plot sigma_btw_pointwise(x) curves at each subset size.

    Parameters
    ----------
    pw_dict : dict
        ``{subset_size: sigma_btw_pointwise_array}``
    method_label : str
        Method name for the plot title.
    group_key : str
    output_path : Path
    plotpara : PlotParams, optional
    verbose : bool

    Returns
    -------
    Path
    """
    plotpara = plotpara or PlotParams()
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ss_sorted = sorted(pw_dict.keys())
    cmap = plt.cm.viridis
    for i, ss in enumerate(ss_sorted):
        frac = i / max(len(ss_sorted) - 1, 1)
        ax.plot(pw_dict[ss], label=f"subset = {ss}",
                linewidth=1.2, color=cmap(frac))
    ax.set_xlabel("Grid index $j$", fontsize=11)
    ax.set_ylabel(r"$\sigma_{\mathrm{btw},j}^2$ (pointwise between-var, obs. scale)",
                  fontsize=11)
    ax.set_title(f"{method_label}: Pointwise Between-Variance — "
                 f"{_format_group_key_title(group_key)}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plotpara.dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"    -> Saved pointwise sigma_btw: {output_path}")
    return output_path


def export_sigma_btw_csv(
    method_detailed: Dict[str, pd.DataFrame],
    output_path: Path,
    *,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Export sigma_btw statistics per method, group, subset_size.

    Parameters
    ----------
    method_detailed : dict
        ``{method_label: detailed_df}`` — each DataFrame should contain a
        ``sigma_btw`` column and a ``subset_size`` column.
    output_path : Path
        Destination CSV path.
    verbose : bool

    Returns
    -------
    Path or None
    """
    rows: list[dict] = []
    for label, df in method_detailed.items():
        if "sigma_btw" not in df.columns:
            continue
        for ss, g in df.groupby("subset_size"):
            rows.append({
                "method": label,
                "subset_size": int(ss),
                "n_mc_runs": len(g),
                "sigma_btw_mean": g["sigma_btw"].mean(),
                "sigma_btw_median": g["sigma_btw"].median(),
                "sigma_btw_std": (g["sigma_btw"].std(ddof=1)
                                  if len(g) > 1 else 0.0),
                "sigma_btw_q25": g["sigma_btw"].quantile(0.25),
                "sigma_btw_q75": g["sigma_btw"].quantile(0.75),
                "sigma_btw_min": g["sigma_btw"].min(),
                "sigma_btw_max": g["sigma_btw"].max(),
            })
    if not rows:
        return None
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    if verbose:
        print(f"    -> Saved sigma_btw CSV: {output_path}")
    return output_path