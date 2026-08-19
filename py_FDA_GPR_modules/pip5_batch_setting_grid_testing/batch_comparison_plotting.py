# pip5_batch_setting_grid_testing/batch_comparison_plotting.py
"""
=============================================================================
Batch Comparison Plotting for the GPR-FDA Pipeline
=============================================================================

Creates overlay plots comparing Summary GPR and Efficiency results across
different batch testing permutations.

Features:
- Summary GPR overlay: Compare summary curves with CI bands from different settings
- Efficiency overlay: Compare learning curves from different settings
- Condensed legends with all setting permutations
- Automatic color/linestyle assignment per permutation

Author: Yunkai Sun (C-STEEL, CSE, ANL)
=============================================================================
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from .batch_config import BatchTestingOptions


# =============================================================================
# Configuration
# =============================================================================

# Color palette for different permutations
PERMUTATION_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Linestyles for additional distinction
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class BatchComparisonConfig:
    """Configuration for batch comparison plots."""
    
    # Summary GPR plot settings
    summary_figsize: Tuple[int, int] = (12, 8)
    summary_individual_alpha: float = 0.1
    summary_ci_alpha: float = 0.15
    
    # Efficiency plot settings
    efficiency_figsize: Tuple[int, int] = (14, 10)
    efficiency_marker_size: int = 6
    efficiency_fill_alpha: float = 0.1
    
    # Common settings
    dpi: int = 300
    min_time_cap: float = 1e-4
    max_time_cap: float = 1e4
    
    # Legend settings
    legend_fontsize: int = 8
    legend_ncol: int = 2
    
    # Axis labels (from settings column_names)
    x_axis_label: str = "X_label"
    y_axis_label: str = "Y_label"
    

# =============================================================================
# Data Loading Utilities
# =============================================================================

def discover_permutations(batch_output_dir: Path) -> List[str]:
    """
    Discover all permutation subdirectories in batch output.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
        
    Returns
    -------
    List[str]
        List of permutation tags (folder names).
    """
    batch_output_dir = Path(batch_output_dir)
    if not batch_output_dir.exists():
        return []
    
    # Filter for directories that look like permutation folders (contain "WM_" or "NS_")
    permutations = []
    for d in sorted(batch_output_dir.iterdir()):
        if d.is_dir() and ("WM_" in d.name or "NS_" in d.name):
            permutations.append(d.name)
    
    return permutations


def discover_groupkeys(batch_output_dir: Path, permutation_tag: str) -> List[str]:
    """
    Discover group_keys (e.g., "pH_1.48_potential_-1.95") from a permutation output folder.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    permutation_tag : str
        Permutation folder name.
        
    Returns
    -------
    List[str]
        List of group_key strings (e.g., "pH_1.48_potential_-1.95").
    """
    perm_dir = Path(batch_output_dir) / permutation_tag
    if not perm_dir.exists():
        return []
    
    group_keys = set()
    
    # Look for Summary_GPR files in summary_gpr/iterative subfolder, then summary_gpr, then root
    summary_dir = perm_dir / "summary_gpr" / "iterative"
    if not summary_dir.exists():
        summary_dir = perm_dir / "summary_gpr"
    if not summary_dir.exists():
        summary_dir = perm_dir  # Fallback to root for backward compatibility
    
    # New pattern: Summary_GPR_pH_X.XX_potential_Y.YY.csv (with group_key)
    for f in summary_dir.glob("Summary_GPR_*.csv"):
        # Extract everything after "Summary_GPR_" and before ".csv"
        name = f.stem  # e.g., "Summary_GPR_pH_1.48_potential_-1.95"
        if name.startswith("Summary_GPR_"):
            group_key = name[len("Summary_GPR_"):]  # e.g., "pH_1.48_potential_-1.95"
            if group_key:  # non-empty
                group_keys.add(group_key)
    
    # Also check efficiency files in learning_curve subfolder or root
    lc_dir = perm_dir / "learning_curve"
    if not lc_dir.exists():
        lc_dir = perm_dir  # Fallback to root for backward compatibility
    
    # New pattern: LearningCurve_pH_X.XX_potential_Y.YY_summary.csv
    for f in lc_dir.glob("LearningCurve_*_summary.csv"):
        name = f.stem  # e.g., "LearningCurve_pH_1.48_potential_-1.95_summary"
        if name.startswith("LearningCurve_") and name.endswith("_summary"):
            # Extract group_key between "LearningCurve_" and "_summary"
            group_key = name[len("LearningCurve_"):-len("_summary")]
            if group_key:
                group_keys.add(group_key)
    
    return sorted(group_keys)


def load_summary_gpr_data(
    batch_output_dir: Path,
    permutation_tag: str,
    group_key: str,
) -> Optional[pd.DataFrame]:
    """
    Load Summary GPR CSV for a given permutation and group_key.
    
    For FGPR-only permutations (no iterative output), loads the
    FGPR_Curve CSV and maps its columns to the expected schema
    (x_real, y_real, Lower_CI_real, Upper_CI_real).
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    permutation_tag : str
        Permutation folder name.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
        
    Returns
    -------
    pd.DataFrame or None
        Summary GPR data, or None if not found.
    """
    perm_dir = Path(batch_output_dir) / permutation_tag
    
    # Try summary_gpr/iterative subfolder first (current layout)
    summary_dir = perm_dir / "summary_gpr" / "iterative"
    csv_path = summary_dir / f"Summary_GPR_{group_key}.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Fallback to summary_gpr root for backward compatibility
    csv_path = perm_dir / "summary_gpr" / f"Summary_GPR_{group_key}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Fallback to root folder for backward compatibility
    csv_path = perm_dir / f"Summary_GPR_{group_key}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # FGPR-only permutations: load FGPR_Curve CSV and remap columns
    fgpr_path = perm_dir / "summary_gpr" / "fgpr" / f"FGPR_Curve_{group_key}.csv"
    if fgpr_path.exists():
        df = pd.read_csv(fgpr_path)
        # Map FGPR columns to the standard schema expected by comparison code
        remap = {}
        if "fgpr_mean_real" in df.columns:
            remap["fgpr_mean_real"] = "y_real"
        if "fgpr_std_real" in df.columns:
            remap["fgpr_std_real"] = "y_std_real"
        if "fgpr_lower_real" in df.columns:
            remap["fgpr_lower_real"] = "Lower_CI_real"
        if "fgpr_upper_real" in df.columns:
            remap["fgpr_upper_real"] = "Upper_CI_real"
        if remap:
            df = df.rename(columns=remap)
        return df
    
    return None


def load_efficiency_data(
    batch_output_dir: Path,
    permutation_tag: str,
    group_key: str,
) -> Optional[pd.DataFrame]:
    """
    Load Efficiency summary CSV for a given permutation and group_key.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    permutation_tag : str
        Permutation folder name.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
        
    Returns
    -------
    pd.DataFrame or None
        Efficiency summary data, or None if not found.
    """
    perm_dir = Path(batch_output_dir) / permutation_tag
    
    # Try learning_curve subfolder first
    lc_dir = perm_dir / "learning_curve"
    csv_path = lc_dir / f"LearningCurve_{group_key}_summary.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    # Fallback to root folder for backward compatibility
    csv_path = perm_dir / f"LearningCurve_{group_key}_summary.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    
    return None


def aggregate_efficiency_to_summary(
    df_detailed: pd.DataFrame,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> pd.DataFrame:
    """
    Aggregate detailed efficiency results to summary statistics.
    
    Parameters
    ----------
    df_detailed : pd.DataFrame
        Detailed DataFrame with columns: subset_size, mc_index, error, time_s.
    q_low : float
        Lower quantile.
    q_high : float
        Upper quantile.
        
    Returns
    -------
    pd.DataFrame
        Summary statistics per subset_size.
    """
    if df_detailed.empty:
        return pd.DataFrame()
    
    records = []
    for subset_size, g in df_detailed.groupby("subset_size"):
        n = len(g)
        records.append({
            "subset_size": subset_size,
            "mean_error": g["error"].mean(),
            "median_error": g["error"].median(),
            "q_error_low": g["error"].quantile(q_low),
            "q_error_high": g["error"].quantile(q_high),
            "mean_time": g["time_s"].mean() if "time_s" in g else 0,
            "median_time": g["time_s"].median() if "time_s" in g else 0,
            "q_time_low": g["time_s"].quantile(q_low) if "time_s" in g else 0,
            "q_time_high": g["time_s"].quantile(q_high) if "time_s" in g else 0,
            "n_mc_runs": n,
        })
    
    return pd.DataFrame(records)


def parse_permutation_tag(tag: str) -> Dict[str, str]:
    """
    Parse permutation tag into components.
    
    Example: "NS_norm__WM_equal__WS_curve" -> 
             {"NS": "norm", "WM": "equal", "WS": "curve"}
    
    Parameters
    ----------
    tag : str
        Permutation tag string.
        
    Returns
    -------
    Dict[str, str]
        Parsed components.
    """
    components = {}
    parts = tag.split("__")
    for part in parts:
        if "_" in part:
            key, value = part.split("_", 1)
            components[key] = value
    # Ensure WM/WS keys exist for FGPR tags (they only have NS + AM)
    if "AM" in components and "WM" not in components:
        components["WM"] = "-"
        components["WS"] = "-"
    return components


def get_short_label(tag: str) -> str:
    """
    Get a short label for legend from permutation tag.
    
    Example: "NS_norm__WM_equal__WS_curve" -> "N.eq.c"
    
    Parameters
    ----------
    tag : str
        Permutation tag string.
        
    Returns
    -------
    str
        Short label.
    """
    components = parse_permutation_tag(tag)
    
    # Abbreviations
    ns_abbr = {"norm": "N", "real": "R"}
    wm_abbr = {"equal": "eq", "iterative": "it", "-": "-"}
    ws_abbr = {"curve": "c", "point": "p", "-": "-"}
    
    ns = ns_abbr.get(components.get("NS", ""), components.get("NS", "?"))
    
    # FGPR tag: NS_norm__AM_fgpr  -> "N.fgpr"
    am = components.get("AM")
    if am == "fgpr":
        return f"{ns}.fgpr"
    
    wm = wm_abbr.get(components.get("WM", ""), components.get("WM", "?"))
    ws = ws_abbr.get(components.get("WS", ""), components.get("WS", "?"))
    
    return f"{ns}.{wm}.{ws}"


# =============================================================================
# Summary GPR Comparison Plot
# =============================================================================

def plot_summary_gpr_comparison(
    batch_output_dir: Path,
    group_key: str,
    output_path: Path,
    *,
    permutation_tags: Optional[List[str]] = None,
    config: Optional[BatchComparisonConfig] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Create overlay plot comparing Summary GPR results from multiple permutations.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
    output_path : Path
        Full path for output PNG file.
    permutation_tags : List[str], optional
        List of permutation tags to include. If None, auto-discovers.
    config : BatchComparisonConfig, optional
        Plot configuration.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Path or None
        Path to saved plot, or None if no data found.
    """
    batch_output_dir = Path(batch_output_dir)
    output_path = Path(output_path)
    config = config or BatchComparisonConfig()
    
    # Discover permutations if not provided
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print(f"No permutation folders found in {batch_output_dir}")
        return None
    
    # Load data for each permutation
    data_dict = {}
    for tag in permutation_tags:
        df = load_summary_gpr_data(batch_output_dir, tag, group_key)
        if df is not None and not df.empty:
            data_dict[tag] = df
    
    if not data_dict:
        if verbose:
            print(f"No Summary GPR data found for group_key {group_key}")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=config.summary_figsize)
    
    # Legend elements
    legend_elements = []
    
    for idx, (tag, df) in enumerate(data_dict.items()):
        color = PERMUTATION_COLORS[idx % len(PERMUTATION_COLORS)]
        linestyle = LINESTYLES[idx // len(PERMUTATION_COLORS) % len(LINESTYLES)]
        short_label = get_short_label(tag)
        
        # Extract data columns
        x_col = "x_real" if "x_real" in df.columns else None
        y_mean_col = "y_real" if "y_real" in df.columns else None
        
        if x_col is None or y_mean_col is None:
            if verbose:
                print(f"  Warning: Missing columns in {tag} (x={x_col}, y={y_mean_col})")
            continue
        
        x = df[x_col].values
        y_mean = df[y_mean_col].values
        
        # Plot mean curve
        line, = ax.plot(
            x, y_mean,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.9,
            label=short_label,
        )
        
        # Plot confidence interval if available
        y_lower_col = None
        y_upper_col = None
        for col_lo, col_hi in [
            ("Lower_CI_real", "Upper_CI_real"),
            ("Lower_CI_normalised", "Upper_CI_normalised"),
        ]:
            if col_lo in df.columns and col_hi in df.columns:
                y_lower_col, y_upper_col = col_lo, col_hi
                break
        
        if y_lower_col and y_upper_col:
            y_lower = df[y_lower_col].values
            y_upper = df[y_upper_col].values
            ax.fill_between(
                x, y_lower, y_upper,
                color=color,
                alpha=config.summary_ci_alpha,
            )
        
        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=tag)
        )
    
    # Format plot - use dynamic labels from config (which come from settings)
    ax.set_xlabel(config.x_axis_label)
    ax.set_ylabel(config.y_axis_label)
    ax.set_title(f"Summary GPR Comparison: {group_key}\n({len(data_dict)} permutations)")
    ax.set_xscale("log")
    ax.set_xlim((config.min_time_cap, config.max_time_cap))
    ax.grid(True, alpha=0.3)
    
    # Condensed legend
    ax.legend(
        handles=legend_elements,
        fontsize=config.legend_fontsize,
        ncol=config.legend_ncol,
        loc="upper right",
        title="Permutations",
        title_fontsize=config.legend_fontsize + 1,
    )
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    
    if verbose:
        print(f"Saved Summary GPR comparison plot: {output_path}")
    
    return output_path


# =============================================================================
# Efficiency Comparison Plot
# =============================================================================

def plot_efficiency_comparison(
    batch_output_dir: Path,
    group_key: str,
    output_path: Path,
    *,
    permutation_tags: Optional[List[str]] = None,
    config: Optional[BatchComparisonConfig] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Create overlay plot comparing Efficiency results from multiple permutations.
    
    Creates a multi-panel figure:
    - Panel 1: Error vs Number of Curves (learning curve)
    - Panel 2: Error vs CPU Time
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
    output_path : Path
        Full path for output PNG file.
    permutation_tags : List[str], optional
        List of permutation tags to include. If None, auto-discovers.
    config : BatchComparisonConfig, optional
        Plot configuration.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Path or None
        Path to saved plot, or None if no data found.
    """
    batch_output_dir = Path(batch_output_dir)
    output_path = Path(output_path)
    config = config or BatchComparisonConfig()
    
    # Discover permutations if not provided
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print(f"No permutation folders found in {batch_output_dir}")
        return None
    
    # Load and aggregate data for each permutation
    data_dict = {}
    for tag in permutation_tags:
        df_detailed = load_efficiency_data(batch_output_dir, tag, group_key)
        if df_detailed is not None and not df_detailed.empty:
            df_summary = aggregate_efficiency_to_summary(df_detailed)
            if not df_summary.empty:
                data_dict[tag] = df_summary
    
    if not data_dict:
        if verbose:
            print(f"No Efficiency data found for group_key {group_key}")
        return None
    
    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.efficiency_figsize)
    
    # Legend elements
    legend_elements = []
    
    for idx, (tag, df_summary) in enumerate(data_dict.items()):
        color = PERMUTATION_COLORS[idx % len(PERMUTATION_COLORS)]
        linestyle = LINESTYLES[idx // len(PERMUTATION_COLORS) % len(LINESTYLES)]
        short_label = get_short_label(tag)
        marker = ["o", "s", "^", "D", "v", "<", ">", "p"][idx % 8]
        
        gA = df_summary.set_index("subset_size")
        
        # === Panel 1: Error vs Number of Curves ===
        ax1.plot(
            gA.index, gA["median_error"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=config.efficiency_marker_size,
            linewidth=1.5,
            alpha=0.9,
            label=short_label,
        )
        
        # IQR band
        if "q_error_low" in gA.columns and "q_error_high" in gA.columns:
            ax1.fill_between(
                gA.index, gA["q_error_low"], gA["q_error_high"],
                color=color,
                alpha=config.efficiency_fill_alpha,
            )
        
        # === Panel 2: Error vs CPU Time ===
        if "median_time" in gA.columns:
            ax2.plot(
                gA["median_time"], gA["median_error"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=config.efficiency_marker_size,
                linewidth=1.5,
                alpha=0.9,
                label=short_label,
            )
            
            # Optional: Error bars for time/error IQR
            if "q_time_low" in gA.columns and "q_time_high" in gA.columns:
                for i, (time_med, err_med) in enumerate(zip(gA["median_time"], gA["median_error"])):
                    time_lo = gA["q_time_low"].iloc[i]
                    time_hi = gA["q_time_high"].iloc[i]
                    err_lo = gA["q_error_low"].iloc[i] if "q_error_low" in gA.columns else err_med
                    err_hi = gA["q_error_high"].iloc[i] if "q_error_high" in gA.columns else err_med
                    
                    # Horizontal error bar (time)
                    ax2.hlines(err_med, time_lo, time_hi, color=color, alpha=0.3, linewidth=1)
                    # Vertical error bar (error)
                    ax2.vlines(time_med, err_lo, err_hi, color=color, alpha=0.3, linewidth=1)
        
        # Add to legend
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle=linestyle, marker=marker,
                   markersize=6, linewidth=1.5, label=tag)
        )
    
    # Format Panel 1
    ax1.set_xlabel("Number of Curves")
    ax1.set_ylabel("Error (RMSE)")
    ax1.set_title(f"Learning Curve Comparison: {group_key}\n({len(data_dict)} permutations)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(
        handles=legend_elements,
        fontsize=config.legend_fontsize,
        ncol=config.legend_ncol,
        loc="upper right",
        title="Permutations",
        title_fontsize=config.legend_fontsize + 1,
    )
    
    # Format Panel 2
    ax2.set_xlabel("CPU Time (seconds)")
    ax2.set_ylabel("Error (RMSE)")
    ax2.set_title("Error vs CPU Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend(
        handles=legend_elements,
        fontsize=config.legend_fontsize,
        ncol=config.legend_ncol,
        loc="upper right",
        title="Permutations",
        title_fontsize=config.legend_fontsize + 1,
    )
    
    fig.tight_layout()
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    
    if verbose:
        print(f"Saved Efficiency comparison plot: {output_path}")
    
    return output_path


# =============================================================================
# Combined Comparison Plot (Efficiency Learning Curves - mirroring pip4 style)
# =============================================================================

def load_efficiency_summary_csv(
    batch_output_dir: Path,
    permutation_tag: str,
    group_key: str,
) -> Optional[pd.DataFrame]:
    """
    Load Efficiency summary CSV (wide format) for a given permutation and group_key.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    permutation_tag : str
        Permutation folder name.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
        
    Returns
    -------
    pd.DataFrame or None
        Efficiency summary data (wide format), or None if not found.
    """
    perm_dir = Path(batch_output_dir) / permutation_tag
    
    # Try learning_curve subfolder first
    lc_dir = perm_dir / "learning_curve"
    search_dirs = [lc_dir, perm_dir] if lc_dir.exists() else [perm_dir]
    
    # Try different naming patterns
    patterns = [
        f"LearningCurve_{group_key}_summary.csv",  # pip4 output format
        f"A_LearningCurve_{group_key}_summary.csv",
        f"Efficiency_{group_key}_summary.csv",
    ]
    
    for search_dir in search_dirs:
        for pattern in patterns:
            csv_path = search_dir / pattern
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0)
                    # Wide format has subset sizes as columns, metrics as rows
                    return df.T  # Transpose to have subset_size as rows
                except Exception:
                    continue
    
    return None


def plot_combined_comparison(
    batch_output_dir: Path,
    group_key: str,
    output_path: Path,
    *,
    permutation_tags: Optional[List[str]] = None,
    config: Optional[BatchComparisonConfig] = None,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Create combined overlay plot comparing Efficiency results across permutations.
    
    Mirrors the pip4 efficiency learning curve layout with 3 panels:
    - Panel 1: Error vs Number of Curves (learning curve)
    - Panel 2: Error vs CPU Time
    - Panel 3: Convergence Iterations vs Subset Size
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    group_key : str
        Group key string (e.g., "pH_1.48_potential_-1.95").
    output_path : Path
        Full path for output PNG file.
    permutation_tags : List[str], optional
        List of permutation tags to include. If None, auto-discovers.
    config : BatchComparisonConfig, optional
        Plot configuration.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Path or None
        Path to saved plot, or None if no data found.
    """
    batch_output_dir = Path(batch_output_dir)
    output_path = Path(output_path)
    config = config or BatchComparisonConfig()
    
    # Discover permutations if not provided
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print(f"No permutation folders found in {batch_output_dir}")
        return None
    
    # Load efficiency data for all permutations
    efficiency_data = {}
    
    for tag in permutation_tags:
        # Try loading summary CSV (wide format from pip4)
        df_summary = load_efficiency_summary_csv(batch_output_dir, tag, group_key)
        if df_summary is not None and not df_summary.empty:
            efficiency_data[tag] = df_summary
            continue
        
        # Fall back to detailed CSV and aggregate
        df_detailed = load_efficiency_data(batch_output_dir, tag, group_key)
        if df_detailed is not None and not df_detailed.empty:
            df_summary = aggregate_efficiency_to_summary(df_detailed)
            if not df_summary.empty:
                efficiency_data[tag] = df_summary
    
    if not efficiency_data:
        if verbose:
            print(f"No Efficiency data found for group_key {group_key}")
        return None
    
    # Create figure with 3 panels (matching pip4 efficiency layout)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    ax1, ax2, ax3 = axes
    
    # Shared legend elements
    legend_elements = []
    
    # Sort tags for consistent ordering
    all_tags = sorted(efficiency_data.keys())
    
    for idx, tag in enumerate(all_tags):
        color = PERMUTATION_COLORS[idx % len(PERMUTATION_COLORS)]
        linestyle = LINESTYLES[idx // len(PERMUTATION_COLORS) % len(LINESTYLES)]
        marker = ["o", "s", "^", "D", "v", "<", ">", "p"][idx % 8]
        
        df = efficiency_data[tag]
        
        # Handle different column naming conventions
        # The summary CSV from pip4 has different column names after transpose
        
        # Determine x-axis (subset sizes)
        if "subset_size" in df.columns:
            x = df["subset_size"].values
            df_indexed = df.set_index("subset_size")
        else:
            # Transposed wide format - index is subset size
            x = df.index.astype(float).values
            df_indexed = df
        
        # === Panel 1: Error vs Number of Curves ===
        # Try different column names for median error
        median_err_col = None
        for col in ["median_error", "median", "q_error_50"]:
            if col in df_indexed.columns:
                median_err_col = col
                break
        
        mean_err_col = None
        for col in ["mean_error", "mean"]:
            if col in df_indexed.columns:
                mean_err_col = col
                break
        
        if median_err_col:
            y_median = df_indexed[median_err_col].values
            ax1.plot(
                x, y_median,
                color=color, linestyle=linestyle, marker=marker,
                markersize=config.efficiency_marker_size,
                linewidth=1.5, alpha=0.9,
            )
            
            # IQR band
            q_low_col = None
            q_high_col = None
            for lo, hi in [("q_error_low", "q_error_high"), ("q25", "q75"), ("q_error_25", "q_error_75")]:
                if lo in df_indexed.columns and hi in df_indexed.columns:
                    q_low_col, q_high_col = lo, hi
                    break
            
            if q_low_col and q_high_col:
                ax1.fill_between(
                    x, df_indexed[q_low_col].values, df_indexed[q_high_col].values,
                    color=color, alpha=config.efficiency_fill_alpha,
                )
        
        # === Panel 2: Error vs CPU Time ===
        time_col = None
        for col in ["median_time", "median_subset_time[s]", "avg_subset_time[s]", "mean_time"]:
            if col in df_indexed.columns:
                time_col = col
                break
        
        if time_col and median_err_col:
            x_time = df_indexed[time_col].values
            y_err = df_indexed[median_err_col].values
            
            ax2.errorbar(
                x_time, y_err,
                fmt=marker, color=color, markersize=config.efficiency_marker_size,
                linestyle=linestyle, linewidth=1.5, alpha=0.8,
                capsize=3,
            )
        
        # === Panel 3: Iterations vs Subset Size ===
        iter_col = None
        for col in ["median_iterations", "mean_iterations"]:
            if col in df_indexed.columns:
                iter_col = col
                break
        
        if iter_col:
            y_iter = df_indexed[iter_col].values
            ax3.plot(
                x, y_iter,
                color=color, linestyle=linestyle, marker=marker,
                markersize=config.efficiency_marker_size,
                linewidth=1.5, alpha=0.9,
            )
            
            # Min-max range for iterations
            if "min_iterations" in df_indexed.columns and "max_iterations" in df_indexed.columns:
                ax3.fill_between(
                    x,
                    df_indexed["min_iterations"].values,
                    df_indexed["max_iterations"].values,
                    color=color, alpha=config.efficiency_fill_alpha,
                )
        
        # Legend element for this permutation
        legend_elements.append(
            Line2D([0], [0], color=color, linestyle=linestyle, marker=marker,
                   markersize=6, linewidth=2, label=tag)
        )
    
    # Format Panel 1: Error vs Number of Curves
    ax1.set_xlabel("# curves kept")
    ax1.set_ylabel("RMSE vs reference\nnormalize:True, log:True")
    ax1.set_title(f"{group_key} - Data Efficiency\n({len(efficiency_data)} permutations)")
    ax1.grid(True, alpha=0.3)
    
    # Format Panel 2: Error vs CPU Time
    ax2.set_xlabel("Average CPU-time per subset [s]")
    ax2.set_ylabel("RMSE vs reference\nnormalize:True, log:True")
    ax2.grid(True, alpha=0.3)
    
    # Format Panel 3: Iterations
    ax3.set_xlabel("# curves kept")
    ax3.set_ylabel("Number of Iterations")
    ax3.set_title("Convergence Iterations vs Subset Size")
    ax3.grid(True, alpha=0.3)
    
    # Single legend at bottom for all panels
    fig.legend(
        handles=legend_elements,
        fontsize=config.legend_fontsize,
        ncol=min(len(all_tags), 4),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        title="Permutations (NS.WM.WS: N=norm, R=real, eq=equal, it=iterative, c=curve, p=point)",
        title_fontsize=config.legend_fontsize,
    )
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)  # Make room for legend
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    
    if verbose:
        print(f"Saved combined comparison plot: {output_path}")
    
    return output_path


# =============================================================================
# Batch Plotting for All Group Keys
# =============================================================================

def plot_all_comparisons(
    batch_output_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    permutation_tags: Optional[List[str]] = None,
    config: Optional[BatchComparisonConfig] = None,
    plot_types: List[str] = ["combined", "summary", "efficiency"],
    verbose: bool = True,
) -> Dict[str, List[Path]]:
    """
    Generate comparison plots for all group keys found in batch output.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    output_dir : Path, optional
        Directory for output plots. Defaults to batch_output_dir/comparisons.
    permutation_tags : List[str], optional
        List of permutation tags to include. If None, auto-discovers.
    config : BatchComparisonConfig, optional
        Plot configuration.
    plot_types : List[str]
        Types of plots to generate: "combined", "summary", "efficiency".
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping plot type to list of output paths.
    """
    batch_output_dir = Path(batch_output_dir)
    output_dir = Path(output_dir) if output_dir else batch_output_dir / "comparisons"
    config = config or BatchComparisonConfig()
    
    # Discover permutations if not provided
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print(f"No permutation folders found in {batch_output_dir}")
        return {}
    
    if verbose:
        print(f"Found {len(permutation_tags)} permutations:")
        for tag in permutation_tags:
            print(f"  - {tag}")
    
    # Discover all group keys across all permutations
    all_group_keys = set()
    for tag in permutation_tags:
        group_keys = discover_groupkeys(batch_output_dir, tag)
        all_group_keys.update(group_keys)
    
    all_group_keys = sorted(all_group_keys)
    
    if verbose:
        print(f"Found {len(all_group_keys)} group_keys: {all_group_keys}")
    
    # Generate plots
    results = {"combined": [], "summary": [], "efficiency": []}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for group_key in all_group_keys:
        if verbose:
            print(f"\nProcessing group_key {group_key}...")
        
        if "combined" in plot_types:
            output_path = output_dir / f"Combined_Comparison_{group_key}.png"
            path = plot_combined_comparison(
                batch_output_dir, group_key, output_path,
                permutation_tags=permutation_tags,
                config=config,
                verbose=verbose,
            )
            if path:
                results["combined"].append(path)
        
        if "summary" in plot_types:
            output_path = output_dir / f"SummaryGPR_Comparison_{group_key}.png"
            path = plot_summary_gpr_comparison(
                batch_output_dir, group_key, output_path,
                permutation_tags=permutation_tags,
                config=config,
                verbose=verbose,
            )
            if path:
                results["summary"].append(path)
        
        if "efficiency" in plot_types:
            output_path = output_dir / f"Efficiency_Comparison_{group_key}.png"
            path = plot_efficiency_comparison(
                batch_output_dir, group_key, output_path,
                permutation_tags=permutation_tags,
                config=config,
                verbose=verbose,
            )
            if path:
                results["efficiency"].append(path)
    
    if verbose:
        total_plots = sum(len(v) for v in results.values())
        print(f"\n{'='*60}")
        print(f"Generated {total_plots} comparison plots:")
        for plot_type, paths in results.items():
            print(f"  - {plot_type}: {len(paths)} plots")
        print(f"Output directory: {output_dir}")
    
    return results


# =============================================================================
# Summary CSV Export Functions
# =============================================================================

def export_summary_gpr_aggregate_csv(
    batch_output_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    permutation_tags: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[float, Path]:
    """
    Export aggregated Summary GPR data across all permutations to CSVs.
    
    Creates one CSV per potential with columns for each permutation's summary curve.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    output_dir : Path, optional
        Directory for output CSVs. Defaults to batch_output_dir/comparisons.
    permutation_tags : List[str], optional
        List of permutation tags to include.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping group_key -> output CSV path.
    """
    batch_output_dir = Path(batch_output_dir)
    output_dir = Path(output_dir) if output_dir else batch_output_dir / "comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print("No permutation folders found")
        return {}
    
    # Discover all group keys
    all_group_keys = set()
    for tag in permutation_tags:
        group_keys = discover_groupkeys(batch_output_dir, tag)
        all_group_keys.update(group_keys)
    
    results = {}
    
    for group_key in sorted(all_group_keys):
        # Collect data from all permutations
        all_data = {}
        
        for tag in permutation_tags:
            df = load_summary_gpr_data(batch_output_dir, tag, group_key)
            if df is None or df.empty:
                continue
            
            # Find x and y columns
            x_col = "x_real" if "x_real" in df.columns else None
            y_mean_col = "y_real" if "y_real" in df.columns else None
            
            if x_col and y_mean_col:
                x_vals = df[x_col].values
                y_vals = df[y_mean_col].values
                
                all_data[tag] = {"x": x_vals, "y_mean": y_vals}
                
                # Also get CI bounds if available
                for lo_col, hi_col in [
                    ("Lower_CI_real", "Upper_CI_real"),
                    ("Lower_CI_normalised", "Upper_CI_normalised"),
                ]:
                    if lo_col in df.columns and hi_col in df.columns:
                        all_data[tag]["y_lower"] = df[lo_col].values
                        all_data[tag]["y_upper"] = df[hi_col].values
                        break
        
        if not all_data:
            continue
        
        # Use the finest (longest) x grid as the common grid
        common_x = max(
            (data["x"] for data in all_data.values()),
            key=len,
        )
        
        # Build aggregated DataFrame, interpolating when grid sizes differ
        agg_df = pd.DataFrame({"x": common_x})
        
        for tag, data in all_data.items():
            short_label = get_short_label(tag)
            x_src = data["x"]
            
            if len(x_src) == len(common_x) and np.allclose(x_src, common_x):
                # Same grid — direct assignment
                agg_df[f"{short_label}_y_mean"] = data["y_mean"]
                if "y_lower" in data:
                    agg_df[f"{short_label}_y_lower"] = data["y_lower"]
                if "y_upper" in data:
                    agg_df[f"{short_label}_y_upper"] = data["y_upper"]
            else:
                # Different grid — interpolate onto common_x
                agg_df[f"{short_label}_y_mean"] = np.interp(
                    common_x, x_src, data["y_mean"]
                )
                if "y_lower" in data:
                    agg_df[f"{short_label}_y_lower"] = np.interp(
                        common_x, x_src, data["y_lower"]
                    )
                if "y_upper" in data:
                    agg_df[f"{short_label}_y_upper"] = np.interp(
                        common_x, x_src, data["y_upper"]
                    )
        
        # Save
        csv_path = output_dir / f"Aggregate_SummaryGPR_{group_key}.csv"
        agg_df.to_csv(csv_path, index=False)
        results[group_key] = csv_path
        
        if verbose:
            print(f"Exported: {csv_path.name} ({len(all_data)} permutations)")
    
    return results


def export_efficiency_aggregate_csv(
    batch_output_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    permutation_tags: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[float, Path]:
    """
    Export aggregated Efficiency data across all permutations to CSVs.
    
    Creates one CSV per potential with learning curve data for all permutations.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    output_dir : Path, optional
        Directory for output CSVs. Defaults to batch_output_dir/comparisons.
    permutation_tags : List[str], optional
        List of permutation tags to include.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping group_key -> output CSV path.
    """
    batch_output_dir = Path(batch_output_dir)
    output_dir = Path(output_dir) if output_dir else batch_output_dir / "comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    if permutation_tags is None:
        permutation_tags = discover_permutations(batch_output_dir)
    
    if not permutation_tags:
        if verbose:
            print("No permutation folders found")
        return {}
    
    # Discover group keys with efficiency data
    all_group_keys = set()
    for tag in permutation_tags:
        perm_dir = batch_output_dir / tag
        
        # Try learning_curve subfolder first
        lc_dir = perm_dir / "learning_curve"
        search_dir = lc_dir if lc_dir.exists() else perm_dir
        
        for f in search_dir.glob("LearningCurve_*_summary.csv"):
            # Extract group_key: LearningCurve_{group_key}_summary.csv
            name = f.stem  # e.g., "LearningCurve_pH_1.48_potential_-1.95_summary"
            if name.startswith("LearningCurve_") and name.endswith("_summary"):
                group_key = name[len("LearningCurve_"):-len("_summary")]
                all_group_keys.add(group_key)
    
    results = {}
    
    for group_key in sorted(all_group_keys):
        # Collect data from all permutations
        all_data = []
        
        for tag in permutation_tags:
            df = load_efficiency_summary_csv(batch_output_dir, tag, group_key)
            if df is None or df.empty:
                continue
            
            short_label = get_short_label(tag)
            
            # Reset index to get subset_size as a column
            df_copy = df.copy()
            df_copy = df_copy.reset_index()
            df_copy.columns = ["subset_size"] + list(df_copy.columns[1:])
            
            # Add permutation identifier
            df_copy["permutation"] = tag
            df_copy["permutation_short"] = short_label
            
            all_data.append(df_copy)
        
        if not all_data:
            continue
        
        # Combine all permutations
        agg_df = pd.concat(all_data, ignore_index=True)
        
        # Save
        csv_path = output_dir / f"Aggregate_Efficiency_{group_key}.csv"
        agg_df.to_csv(csv_path, index=False)
        results[group_key] = csv_path
        
        if verbose:
            print(f"Exported: {csv_path.name} ({len(all_data)} permutations)")
    
    return results


def export_all_aggregate_csvs(
    batch_output_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    permutation_tags: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """
    Export all aggregated CSVs for Summary GPR and Efficiency data.
    
    Parameters
    ----------
    batch_output_dir : Path
        Base directory containing permutation subfolders.
    output_dir : Path, optional
        Directory for output CSVs.
    permutation_tags : List[str], optional
        List of permutation tags to include.
    verbose : bool
        Print status messages.
        
    Returns
    -------
    Dict[str, Dict[str, Path]]
        Dictionary with keys "summary_gpr" and "efficiency", each mapping
        group_key -> CSV path.
    """
    if verbose:
        print("\n" + "="*60)
        print("Exporting Aggregate CSVs")
        print("="*60)
    
    results = {}
    
    results["summary_gpr"] = export_summary_gpr_aggregate_csv(
        batch_output_dir,
        output_dir,
        permutation_tags=permutation_tags,
        verbose=verbose,
    )
    
    results["efficiency"] = export_efficiency_aggregate_csv(
        batch_output_dir,
        output_dir,
        permutation_tags=permutation_tags,
        verbose=verbose,
    )
    
    if verbose:
        n_summary = len(results["summary_gpr"])
        n_eff = len(results["efficiency"])
        print(f"\nExported {n_summary} Summary GPR CSVs, {n_eff} Efficiency CSVs")
    
    return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "BatchComparisonConfig",
    "PERMUTATION_COLORS",
    "LINESTYLES",
    # Data utilities
    "discover_permutations",
    "discover_groupkeys",
    "load_summary_gpr_data",
    "load_efficiency_data",
    "load_efficiency_summary_csv",
    "aggregate_efficiency_to_summary",
    "parse_permutation_tag",
    "get_short_label",
    # CSV export functions
    "export_summary_gpr_aggregate_csv",
    "export_efficiency_aggregate_csv",
    "export_all_aggregate_csvs",
    # Individual plot functions
    "plot_summary_gpr_comparison",
    "plot_efficiency_comparison",
    "plot_combined_comparison",
    # Batch function
    "plot_all_comparisons",
]
