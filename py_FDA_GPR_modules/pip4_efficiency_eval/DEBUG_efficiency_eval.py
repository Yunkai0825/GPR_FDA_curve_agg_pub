# pip4_efficiency_eval/DEBUG_efficiency_eval.py
"""
DEBUG Script: Unified Efficiency Evaluation Pipeline

Tests all six aggregation methods through the efficiency evaluation pipeline,
generates comparison plots, and produces sigma_btw / covariance diagnostics.

Methods tested:
  1. Plain Average             (equal weights, no normalization)
  2. Plain Average (normalised)(equal weights, with normalization)
  3. Pointwise Aggregation     (iterative weights, point scope)
  4. Curvewise Aggregation     (iterative weights, curve scope)
  5. FGPR (normalised scale)   (functional GPR, normalized inputs)
  6. FGPR (observation scale)  (functional GPR, raw inputs, scale=1)

Uses SettingsManager to load all configuration from settings.json.
Settings are automatically saved to output for reproducibility.

Output: DEBUG_pip4_Efficiency_Output/
  <method_name>/             per-method CSVs, plots, .npy diagnostics
  AllMethod_Comparison_*.png
  AllMethod_BarSummary.png
  SigmaBtw_comparison_*.png
  FGPR_CovMatrix_*.png
  FGPR_DiagStd_*.png
  SigmaBtw_pointwise_*.png
  sigma_btw_summary_*.csv
  all_method_comparison.csv

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---- Path setup ------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SETTINGS_FILE = (
    PROJECT_ROOT
    / "py_FDA_GPR_modules"
    / "pip0_dataloading"
    / "DEBUG_pip0_input"
    / "DEBUG_settings.json"
)


# ---- Method specification ---------------------------------------------------

@dataclass
class MethodSpec:
    """Specification for one aggregation method."""
    name: str               # Short name for display & folder
    weight_mode: str         # "equal" or "iterative"
    weight_scope: str        # "curve" or "point"
    normalization: bool      # normalization_summary
    aggregation_method: str  # "iterative" or "fgpr"


METHODS = [
    MethodSpec("1_plain_average",
               weight_mode="equal", weight_scope="curve",
               normalization=False, aggregation_method="iterative"),
    MethodSpec("2_plain_average_normalised",
               weight_mode="equal", weight_scope="curve",
               normalization=True, aggregation_method="iterative"),
    MethodSpec("3_pointwise_iterative",
               weight_mode="iterative", weight_scope="point",
               normalization=True, aggregation_method="iterative"),
    MethodSpec("4_curvewise_iterative",
               weight_mode="iterative", weight_scope="curve",
               normalization=True, aggregation_method="iterative"),
    MethodSpec("5_fgpr",
               weight_mode="equal", weight_scope="curve",
               normalization=True, aggregation_method="fgpr"),
    MethodSpec("6_fgpr_observation_scale",
               weight_mode="equal", weight_scope="curve",
               normalization=False, aggregation_method="fgpr"),
]


# ---- Helpers ----------------------------------------------------------------

def run_single_method(
    method: MethodSpec,
    input_folder: Path,
    base_output_folder: Path,
    settings_manager,
) -> Optional[Dict]:
    """Run efficiency evaluation for one method and return summary dict."""
    from py_FDA_GPR_modules.pip4_efficiency_eval import EfficiencyOrchestrator

    output_folder = base_output_folder / method.name
    os.makedirs(output_folder, exist_ok=True)

    dir_params, glob_params, scale_params, plot_params = (
        settings_manager.get_efficiency_configs(
            indiv_dir=input_folder,
            output_dir=output_folder,
        )
    )
    summary_cfg, summary_hp = settings_manager.get_summary_gpr_configs(
        input_directory=input_folder,
        output_directory=output_folder,
    )
    summary_cfg.weight_mode = method.weight_mode
    summary_cfg.weight_scope = method.weight_scope
    summary_cfg.normalization_summary = method.normalization

    print(f"  Config: weight_mode={method.weight_mode}, "
          f"weight_scope={method.weight_scope}, "
          f"normalization={method.normalization}, "
          f"aggregation_method={method.aggregation_method}")

    t_start = time.perf_counter()

    orchestrator = EfficiencyOrchestrator(
        dirpara=dir_params,
        summary_gpr_config=summary_cfg,
        summary_gpr_hyperparams=summary_hp,
        globpara=glob_params,
        scapara=scale_params,
        plotpara=plot_params,
        verbose=True,
        aggregation_method=method.aggregation_method,
    )

    results = orchestrator.process_all()
    elapsed = time.perf_counter() - t_start

    group_summaries = {}
    for gk in sorted(results):
        lc = results[gk]
        df = lc.summary
        group_summaries[gk] = {
            "n_subsets": len(df),
            "min_error": (float(df["mean_error"].min())
                          if "mean_error" in df.columns else None),
            "max_error": (float(df["mean_error"].max())
                          if "mean_error" in df.columns else None),
            "n_mc_runs": len(lc.detailed),
        }

    return {
        "method": method.name,
        "groups": group_summaries,
        "elapsed_s": elapsed,
        "n_groups": len(results),
    }


def _discover_groups(base_output: Path) -> List[str]:
    """Discover group suffixes from method-1 folder."""
    m1 = base_output / "1_plain_average"
    if not m1.exists():
        return []
    groups = []
    for f in sorted(m1.glob("LearningCurve_*_detailed.csv")):
        stem = f.stem.replace("LearningCurve_", "").replace("_detailed", "")
        groups.append(stem)
    return groups


def _load_detailed(base_output: Path, folder: str,
                   group_suffix: str) -> Optional[pd.DataFrame]:
    """Load a detailed CSV for one method + group."""
    p = base_output / folder / f"LearningCurve_{group_suffix}_detailed.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_cov_dict(base_output: Path, group_suffix: str
                   ) -> Dict[int, np.ndarray]:
    """Load FGPR covariance .npy files into {subset_size: cov} (legacy, uses 5_fgpr)."""
    return _load_cov_dict_from(base_output, "5_fgpr", group_suffix)


def _load_cov_dict_from(base_output: Path, folder: str,
                        group_suffix: str) -> Dict[int, np.ndarray]:
    """Load FGPR covariance .npy files for a given method folder."""
    cov_dir = base_output / folder / "cov_matrices"
    out: Dict[int, np.ndarray] = {}
    if not cov_dir.exists():
        return out
    for f in sorted(cov_dir.glob(f"Cagg_{group_suffix}_ss*.npy")):
        ss = int(f.stem.split("_ss")[-1])
        out[ss] = np.load(f)
    return out


def _load_pw_dict(base_output: Path, folder: str,
                  group_suffix: str) -> Dict[int, np.ndarray]:
    """Load sigma_btw_pointwise .npy files into {subset_size: array}."""
    pw_dir = base_output / folder / "sigma_btw_pointwise"
    out: Dict[int, np.ndarray] = {}
    if not pw_dir.exists():
        return out
    for f in sorted(pw_dir.glob(f"sigma_btw_pw_{group_suffix}_ss*.npy")):
        ss = int(f.stem.split("_ss")[-1])
        out[ss] = np.load(f)
    return out


# ---- Main -------------------------------------------------------------------

def main():
    """Main entry point for unified DEBUG script."""
    print("=" * 70)
    print("DEBUG: Unified Efficiency Evaluation Pipeline")
    print("=" * 70)

    # ================================================================
    # Step 0: Settings
    # ================================================================
    print("\n[Step 0] Loading settings...")
    try:
        from py_FDA_GPR_modules.pip0_dataloading import SettingsManager
    except ImportError as e:
        print(f"  X Failed to import SettingsManager: {e}")
        return 1

    if not SETTINGS_FILE.exists():
        print(f"  X Settings file not found: {SETTINGS_FILE}")
        return 1

    settings_manager = SettingsManager(
        settings_path=SETTINGS_FILE,
        input_folder=SETTINGS_FILE.parent,
    )
    print(f"  + Settings loaded from: {SETTINGS_FILE}")

    input_folder = settings_manager.get_pip2_output_dir()
    base_output = settings_manager.get_pip4_output_dir()
    print(f"  + Input folder:  {input_folder}")
    print(f"  + Output folder: {base_output}")

    # Verify input data
    from py_FDA_GPR_modules.pip4_efficiency_eval import (
        load_all_individual_gprs, group_gprs_by_key,
    )
    all_gprs = load_all_individual_gprs(
        directory=str(input_folder),
        pattern="Individual_GPR_*.csv",
        verbose=False,
    )
    if not all_gprs:
        print("  X No individual GPR curves found!")
        return 1

    gprs_by_group = group_gprs_by_key(all_gprs, key_attr="group_key")
    n_total, n_groups = len(all_gprs), len(gprs_by_group)
    viable = sum(1 for v in gprs_by_group.values() if len(v) >= 3)
    print(f"  + {n_total} curves in {n_groups} groups "
          f"({viable} viable, >= 3 curves)")

    has_cov = sum(1 for g in all_gprs if g.covariance_matrix is not None)
    print(f"  + Covariance matrices: {has_cov}/{n_total} curves")

    # ================================================================
    # Step 1: Run all methods
    # ================================================================
    all_summaries: List[Dict] = []
    n_methods = len(METHODS)

    for i, method in enumerate(METHODS, 1):
        print(f"\n{'='*70}")
        print(f"[Method {i}/{n_methods}] {method.name}")
        print(f"{'='*70}")

        try:
            info = run_single_method(
                method, input_folder, base_output, settings_manager,
            )
            if info is not None:
                all_summaries.append(info)
                print(f"\n  >> Completed in {info['elapsed_s']:.2f}s "
                      f"({info['n_groups']} groups)")
        except Exception as e:
            print(f"\n  XX Method {method.name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ================================================================
    # Step 2: Print comparison summary
    # ================================================================
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")

    if not all_summaries:
        print("No methods completed successfully.")
        return 1

    header = f"{'Method':<35} {'Groups':>6} {'Time(s)':>10}"
    print(header)
    print("-" * len(header))
    for s in all_summaries:
        print(f"{s['method']:<35} {s['n_groups']:>6} {s['elapsed_s']:>10.2f}")

    group_keys = sorted({gk for s in all_summaries for gk in s["groups"]})
    for gk in group_keys:
        print(f"\n  Group: {gk}")
        row_hdr = (f"    {'Method':<35} {'MC runs':>8} "
                   f"{'min err':>10} {'max err':>10}")
        print(row_hdr)
        print("    " + "-" * (len(row_hdr) - 4))
        for s in all_summaries:
            g = s["groups"].get(gk)
            if g is None:
                print(f"    {s['method']:<35} {'N/A':>8} "
                      f"{'N/A':>10} {'N/A':>10}")
            else:
                me = (f"{g['min_error']:.4f}"
                      if g['min_error'] is not None else "N/A")
                mx = (f"{g['max_error']:.4f}"
                      if g['max_error'] is not None else "N/A")
                print(f"    {s['method']:<35} {g['n_mc_runs']:>8} "
                      f"{me:>10} {mx:>10}")

    # Save comparison CSV
    comp_rows = []
    for s in all_summaries:
        for gk, g in s["groups"].items():
            comp_rows.append({
                "method": s["method"],
                "group_key": gk,
                "n_mc_runs": g["n_mc_runs"],
                "min_mean_error": g["min_error"],
                "max_mean_error": g["max_error"],
                "total_time_s": s["elapsed_s"],
            })
    comp_df = pd.DataFrame(comp_rows) if comp_rows else pd.DataFrame()
    if not comp_df.empty:
        comp_csv = base_output / "all_method_comparison.csv"
        comp_df.to_csv(comp_csv, index=False)
        print(f"\n  Comparison CSV: {comp_csv}")

    # Save settings for reproducibility
    settings_manager.save_to_output(base_output, "settings_used.json")

    # ================================================================
    # Step 3: Generate comparison & diagnostic plots
    # ================================================================
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}\n")

    from py_FDA_GPR_modules.pip4_efficiency_eval import (
        plot_multimethod_comparison,
        plot_multimethod_bar_summary,
        plot_sigma_btw_comparison,
        plot_covariance_heatmaps,
        plot_covariance_diagonal,
        plot_pointwise_sigma_btw,
        export_sigma_btw_csv,
    )

    # Get scale params for axis labels
    _, _, scale_params, _ = settings_manager.get_efficiency_configs(
        indiv_dir=input_folder, output_dir=base_output,
    )

    groups = _discover_groups(base_output)
    if not groups:
        print("  No group data found for plotting.")
        return 0

    print(f"  Groups: {groups}\n")

    # Map: method folder -> display label
    METHOD_LABELS = {m.name: m.name.split("_", 1)[1].replace("_", " ").title()
                     for m in METHODS}
    # Which methods have sigma_btw
    SIGMA_BTW_FOLDERS = ["3_pointwise_iterative",
                         "4_curvewise_iterative",
                         "5_fgpr",
                         "6_fgpr_observation_scale"]
    # Which methods are FGPR (covariance heatmaps)
    FGPR_FOLDERS = ["5_fgpr", "6_fgpr_observation_scale"]

    for group_suffix in groups:
        filename_key = group_suffix
        print(f"--- {group_suffix} ---")

        # 3a. Multi-method learning curve comparison
        method_det: Dict[str, pd.DataFrame] = {}
        for m in METHODS:
            df = _load_detailed(base_output, m.name, group_suffix)
            if df is not None:
                method_det[METHOD_LABELS[m.name]] = df

        if method_det:
            plot_multimethod_comparison(
                method_det, group_suffix,
                base_output / f"AllMethod_Comparison_{filename_key}.png",
                scapara=scale_params,
            )

        # 3b. sigma_btw comparison (methods that have it)
        sbtw_det: Dict[str, pd.DataFrame] = {}
        for folder in SIGMA_BTW_FOLDERS:
            df = _load_detailed(base_output, folder, group_suffix)
            if df is not None and "sigma_btw" in df.columns:
                sbtw_det[METHOD_LABELS[folder]] = df

        if sbtw_det:
            plot_sigma_btw_comparison(
                sbtw_det, group_suffix,
                base_output / f"SigmaBtw_comparison_{filename_key}.png",
            )

        # 3c. FGPR covariance heatmaps + diagonal (for each FGPR method)
        for fgpr_folder in FGPR_FOLDERS:
            cov_dict = _load_cov_dict_from(base_output, fgpr_folder, group_suffix)
            if cov_dict:
                fgpr_dir = base_output / fgpr_folder
                plot_covariance_heatmaps(
                    cov_dict, group_suffix,
                    fgpr_dir / f"FGPR_CovMatrix_{filename_key}.png",
                )
                plot_covariance_diagonal(
                    cov_dict, group_suffix,
                    fgpr_dir / f"FGPR_DiagStd_{filename_key}.png",
                )

        # 3d. Pointwise sigma_btw profiles (iterative methods only)
        for folder in ("3_pointwise_iterative", "4_curvewise_iterative"):
            pw_dict = _load_pw_dict(base_output, folder, group_suffix)
            if pw_dict:
                safe = folder.split("_", 1)[1]
                plot_pointwise_sigma_btw(
                    pw_dict, METHOD_LABELS[folder], group_suffix,
                    base_output / folder / f"SigmaBtw_pointwise_{safe}_{filename_key}.png",
                )

        # 3e. sigma_btw summary CSV (per group)
        if sbtw_det:
            export_sigma_btw_csv(
                sbtw_det,
                base_output / f"sigma_btw_summary_{filename_key}.csv",
            )

    # 3f. Bar chart summary
    if not comp_df.empty:
        plot_multimethod_bar_summary(
            comp_df,
            base_output / "AllMethod_BarSummary.png",
        )

    # ================================================================
    # Done
    # ================================================================
    print(f"\n{'='*70}")
    print("All output saved to:")
    print(f"  {base_output}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
