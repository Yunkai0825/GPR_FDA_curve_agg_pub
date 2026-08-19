# pip3_FDA_scoring_and_aggregations/DEBUG_summary_gpr.py
"""
DEBUG Script: Summary GPR Aggregation Pipeline

This script tests the Summary GPR aggregation using output from pip2 DEBUG.

Uses SettingsManager to load all configuration from DEBUG_settings.json.
Settings are automatically saved to output for reproducibility.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Add parent directories to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Settings file location
SETTINGS_FILE = PROJECT_ROOT / "py_FDA_GPR_modules" / "pip0_dataloading" / "DEBUG_pip0_input" / "DEBUG_settings.json"


def print_results_summary(results: Dict, output_dir: Path):
    """Print summary of results."""
    print(f"\n  Processed {len(results)} groups:")
    for group_key, result in sorted(results.items()):
        weights = result.weights
        if weights.ndim == 2:
            weights = weights.mean(axis=1)
        
        print(f"\n    {group_key}:")
        print(f"      Curves aggregated: {result.n_curves}")
        print(f"      Y scaling: {result.y_scaling.method} (factor={result.y_scaling.params.get('factor', 1.0):.4f})")
        print(f"      Weights: min={weights.min():.4f}, max={weights.max():.4f}")
    
    print(f"\n  Output saved to: {output_dir}")
    all_files = [p for p in output_dir.rglob('*') if p.is_file()]
    print(f"  Files generated: {len(all_files)}")
    # List subfolders
    subdirs = sorted(set(p.parent.relative_to(output_dir) for p in all_files if p.parent != output_dir))
    if subdirs:
        print(f"  Subfolders: {[str(s) for s in subdirs]}")


def compare_fgpr_vs_direct_average(
    results: Dict,
    all_gprs: List,
    output_dir: Path,
):
    """Compare FGPR aggregated mean vs direct (unweighted) average in normalised space.

    For every multi-curve group that has an FGPR result, this function:
    1. Reconstructs the per-curve normalised means on the covariance grid.
    2. Computes the direct (unweighted) point-wise mean and std.
    3. Prints numeric comparison stats (max |diff|, RMSE, R²).
    4. Generates an overlay plot saved to ``output_dir/fgpr/``.
    """
    from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import group_gprs_by_key

    grouped = group_gprs_by_key(all_gprs)  # same grouping as orchestrator
    compare_dir = output_dir / "fgpr"
    os.makedirs(compare_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("[Step 4] FGPR vs Direct Average Comparison (normalised space)")
    print("=" * 60)

    for group_key, result in sorted(results.items()):
        if result.fgpr_mean_norm is None or result.n_curves < 2:
            continue

        gprs = grouped.get(group_key, [])
        if not gprs:
            continue

        # --- Build normalised means on the covariance (operator) grid ---
        # Use the same grid that FGPR operated on.
        x_grid = result.x_pred_transformed
        y_norm_curves = []
        for gpr in gprs:
            sort_idx = np.argsort(gpr.x_pred_transformed)
            x_sorted = gpr.x_pred_transformed[sort_idx]
            y_norm_sorted = gpr.y_scaling.transform(gpr.y_pred)[sort_idx]
            f = interp1d(x_sorted, y_norm_sorted, kind="linear",
                         fill_value="extrapolate")
            y_norm_curves.append(f(x_grid))

        y_norm_stack = np.vstack(y_norm_curves)     # (R, N)
        direct_mean = y_norm_stack.mean(axis=0)     # simple average
        direct_std = y_norm_stack.std(axis=0, ddof=1)

        fgpr_mean = result.fgpr_mean_norm
        fgpr_std = result.fgpr_std_norm

        diff = fgpr_mean - direct_mean
        max_abs_diff = float(np.max(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        # R² of FGPR mean relative to direct mean
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((direct_mean - direct_mean.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        # FGPR weights
        weights = result.fgpr_weights
        w_min, w_max = float(weights.min()), float(weights.max())
        w_ratio = w_max / w_min if w_min > 0 else float("inf")

        print(f"\n  {group_key} ({result.n_curves} curves):")
        print(f"    max|FGPR - avg|  = {max_abs_diff:.6g}")
        print(f"    RMSE(FGPR, avg)  = {rmse:.6g}")
        print(f"    R²               = {r2:.8f}")
        print(f"    σ²_btw           = {result.fgpr_sigma_btw_squared:.6g}")
        print(f"    weight range     = [{w_min:.4f}, {w_max:.4f}]  (ratio {w_ratio:.2f})")

        # --- Plot ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})

        ax_main, ax_diff, ax_wt = axes

        # Top panel: curves + means
        for i, y_n in enumerate(y_norm_curves):
            ax_main.plot(x_grid, y_n, color="0.75", lw=0.6,
                         label="individual" if i == 0 else None)
        ax_main.plot(x_grid, direct_mean, "k--", lw=1.5,
                     label="direct average")
        ax_main.fill_between(x_grid,
                             direct_mean - direct_std,
                             direct_mean + direct_std,
                             color="k", alpha=0.08, label="±1 SD (sample)")
        ax_main.plot(x_grid, fgpr_mean, "C0-", lw=1.5,
                     label="FGPR mean")
        ax_main.fill_between(x_grid,
                             fgpr_mean - fgpr_std,
                             fgpr_mean + fgpr_std,
                             color="C0", alpha=0.15, label="±1 SD (FGPR)")
        ax_main.set_ylabel("normalised y")
        ax_main.legend(fontsize=8, loc="best")
        ax_main.set_title(f"FGPR vs Direct Average — {group_key}\n"
                          f"RMSE={rmse:.4g}   σ²_btw={result.fgpr_sigma_btw_squared:.4g}   "
                          f"R²={r2:.6f}")

        # Middle panel: difference
        ax_diff.axhline(0, color="k", lw=0.5)
        ax_diff.plot(x_grid, diff, "C3-", lw=1.0)
        ax_diff.set_ylabel("FGPR − avg")
        ax_diff.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

        # Bottom panel: weights
        w_sorted = np.sort(weights)[::-1]
        ax_wt.bar(range(len(w_sorted)), w_sorted, color="C2", alpha=0.7)
        ax_wt.axhline(1.0 / len(weights), color="k", ls="--", lw=0.8,
                      label="uniform")
        ax_wt.set_ylabel("weight")
        ax_wt.set_xlabel("curve rank")
        ax_wt.legend(fontsize=8)

        fig.tight_layout()
        safe_key = group_key.replace("|", "_").replace("=", "_")
        fig_path = compare_dir / f"FGPR_vs_DirectAvg_{safe_key}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"    plot → {fig_path.name}")


def main():
    """Main entry point for DEBUG script."""
    
    print("=" * 60)
    print("DEBUG: Summary GPR Aggregation Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 0: Import modules and load settings via SettingsManager
    # -------------------------------------------------------------------------
    print("\n[Step 0] Importing modules and loading settings...")
    
    try:
        from py_FDA_GPR_modules.pip0_dataloading import SettingsManager
        print("  + SettingsManager imported")
    except ImportError as e:
        print(f"  X Failed to import SettingsManager: {e}")
        return
    
    try:
        from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import (
            SummaryGPROrchestrator,
            load_all_individual_gprs,
            group_gprs_by_key,
        )
        print("  + pip3_FDA_scoring_and_aggregations imported")
    except ImportError as e:
        print(f"  X Failed to import modules: {e}")
        return
    
    # Load settings via SettingsManager
    if not SETTINGS_FILE.exists():
        print(f"  X Settings file not found: {SETTINGS_FILE}")
        return
    
    settings_manager = SettingsManager(
        settings_path=SETTINGS_FILE,
        input_folder=SETTINGS_FILE.parent,
    )
    print(f"  + Settings loaded from: {SETTINGS_FILE}")
    
    # Get paths from settings
    input_folder = settings_manager.get_pip2_output_dir()  # pip2 output is pip3 input
    output_folder = settings_manager.get_pip3_output_dir()
    print(f"  + Input folder: {input_folder}")
    print(f"  + Output folder: {output_folder}")
    
    # -------------------------------------------------------------------------
    # Step 1: Check input data
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] Checking input: {input_folder}")
    
    if not input_folder.exists():
        print(f"  X Input folder does not exist: {input_folder}")
        print("  Please run DEBUG_individual_GPR.py first.")
        return
    
    # Load individual GPRs
    all_gprs = load_all_individual_gprs(
        directory=str(input_folder),
        pattern="Individual_GPR_*.csv",
        verbose=True,
    )
    
    if not all_gprs:
        print("  X No individual GPR curves found!")
        return
    
    gprs_by_pot = group_gprs_by_key(all_gprs, key_attr="group_flags.potential")
    print(f"  Found {len(gprs_by_pot)} potentials:")
    for pot in sorted(gprs_by_pot.keys()):
        print(f"    {pot}: {len(gprs_by_pot[pot])} curves")
    
    # -------------------------------------------------------------------------
    # Step 2: Run Summary GPR - config from SettingsManager
    # -------------------------------------------------------------------------
    print(f"\n[Step 2] Running Summary GPR aggregation...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    summary_cfg, summary_hp = settings_manager.get_summary_gpr_configs(
        input_directory=input_folder,
        output_directory=output_folder,
    )
    
    print(f"  Weight mode: {summary_cfg.weight_mode}")
    print(f"  Weight scope: {summary_cfg.weight_scope}")
    print(f"  FGPR enabled: {summary_cfg.enable_fgpr}")
    print(f"  Operator fusion enabled: {summary_cfg.enable_operator_fusion}")
    
    orchestrator = SummaryGPROrchestrator(summary_cfg, summary_hp, verbose=True)
    
    results = orchestrator.process_all(
        export_results=True,
        plot_results=True,
    )

    # -------------------------------------------------------------------------
    # Step 3: Results Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 3] Results Summary")
    print("=" * 60)
    
    print_results_summary(results, output_folder)
    
    # -------------------------------------------------------------------------
    # Step 4: FGPR vs Direct Average comparison
    # -------------------------------------------------------------------------
    compare_fgpr_vs_direct_average(results, all_gprs, output_folder)
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(output_folder, "settings_used.json")
    print(f"\nSettings saved to: {output_folder / 'settings_used.json'}")


if __name__ == "__main__":
    main()
