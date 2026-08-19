#!/usr/bin/env python
# main_GPR_FDA_pipeline.py
"""
=============================================================================
GPR-FDA: Gaussian Process Regression for Functional Data Analysis
=============================================================================

This script provides a complete pipeline for:

    Step 1: Generate Individual GPRs (pip0 → pip1 → pip2)
        - Load .cor files from input folder
        - Preprocess with configurable y-scaling
        - Fit GPR models to each curve
        - Output: {folder_name}_output/individual_GPR/

    Step 2: Batch Grid Testing (pip5)
        - Test 6 baseline combinations plus optional FGPR and Student-t methods
        - Run Summary GPR and Efficiency Evaluation for each combination
        - Generate comparison plots
        - Output: {folder_name}_output/batch_testing/

All settings are loaded from settings.json via SettingsManager.
Settings used are automatically saved to output for reproducibility.

Usage:
------
    # Run full pipeline (Step 1 + Step 2)
    python main_GPR_FDA_pipeline.py --path "./_input/CCNF_CTAB_PT" --batch

    # Run only individual GPR (Step 1)
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 1

    # Run only batch testing (Step 2)
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 2

    # Run only Summary GPR (Step 2a) or only Learning Curve (Step 2b)
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 2a
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 2b

    # Run only specific Step 2 combinations
    python main_GPR_FDA_pipeline.py --path /path -s 2 --combos NS_norm__AM_fgpr NS_real__AM_fgpr
    python main_GPR_FDA_pipeline.py --path /path -s 2 --method fgpr
    python main_GPR_FDA_pipeline.py --path /path -s 2 --norm real --method iterative

Author: Yunkai Sun (C-STEEL, CSE, ANL)
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Step 1: Individual GPR Processing (pip0 → pip1 → pip2)
# =============================================================================

def run_step1_individual_gpr_batch(
    settings_manager,
    subdirectories: list,
    verbose: bool = True,
) -> bool:
    """
    Step 1 (Batch): Load .cor files from multiple subdirectories, combine, 
    and fit Individual GPR models.
    
    Args:
        settings_manager: SettingsManager instance with loaded settings
        subdirectories: List of Path objects to subdirectories containing .cor files
        verbose: Whether to print progress
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("STEP 1: Individual GPR Processing (BATCH - Combined Data)")
    print("=" * 60)
    
    output_dir = settings_manager.get_pip2_output_dir()
    print(f"  Input:  {len(subdirectories)} subdirectories")
    print(f"  Output: {output_dir}")
    
    # -------------------------------------------------------------------------
    # Import modules
    # -------------------------------------------------------------------------
    try:
        from py_FDA_GPR_modules.pip0_dataloading import DataLoader
        from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
        from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor
    except ImportError as e:
        print(f"  ERROR: Failed to import modules: {e}")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.1: Load data from ALL subdirectories using DataLoader class method
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.1] Loading data from {len(subdirectories)} subdirectories...")
    
    filename_parsing_config = settings_manager.get_filename_parsing_config()
    
    loading_result = DataLoader.load_from_subdirectories(
        parent_folder=subdirectories[0].parent,
        subdirectories=subdirectories,
        verbose=verbose,
        filename_parsing_config=filename_parsing_config,
    )
    
    print(f"\n  Total loaded: {loading_result.num_curves} curves")
    print(f"  Groups: {loading_result.primary_key_values}")
    
    if loading_result.num_curves == 0:
        print("  ERROR: No curves loaded from any subdirectory.")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.2: Preprocess (pip1)
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.2] Preprocessing...")
    
    preproc_cfg = settings_manager.get_preproc_config()
    print(f"  Y-Scaling: {preproc_cfg.y_scaling_method}")
    
    preprocessor = DataPreprocessor(config=preproc_cfg, verbose=verbose)
    preproc_result = preprocessor.preprocess_all(loading_result.curves)
    
    print(f"  Preprocessed {preproc_result.num_curves} curves")
    print(f"  Skipped {preproc_result.num_skipped} curves")
    
    if preproc_result.num_curves == 0:
        print("  ERROR: No valid curves after preprocessing.")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.3: Fit Individual GPRs (pip2)
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.3] Fitting Individual GPR models...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gpr_cfg, export_cfg = settings_manager.get_gpr_configs()
    
    processor = IndividualGPRProcessor(
        gpr_config=gpr_cfg,
        export_config=export_cfg,
        output_directory=output_dir,
        verbose=verbose,
    )
    
    gpr_result = processor.fit_all(
        preproc_result.curves,
        export_results=True,
        plot_results=True,
    )
    
    print(f"\n  Successfully fitted: {gpr_result.num_results} curves")
    print(f"  Skipped: {gpr_result.num_skipped} curves")
    print(f"  Output directory: {output_dir}")
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(output_dir, "settings_used.json")
    print(f"  Settings saved to: {output_dir / 'settings_used.json'}")
    
    return gpr_result.num_results > 0


def run_step1_individual_gpr(
    settings_manager,
    input_folder: Path,
    verbose: bool = True,
) -> bool:
    """
    Step 1: Process raw .cor files and fit Individual GPR models.
    
    All configuration is loaded from SettingsManager.
    
    Args:
        settings_manager: SettingsManager instance with loaded settings
        input_folder: Path to folder containing .cor files
        verbose: Whether to print progress
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("STEP 1: Individual GPR Processing (pip0 -> pip1 -> pip2)")
    print("=" * 60)
    
    output_dir = settings_manager.get_pip2_output_dir()
    print(f"  Input:  {input_folder}")
    print(f"  Output: {output_dir}")
    
    # -------------------------------------------------------------------------
    # Import modules
    # -------------------------------------------------------------------------
    try:
        from py_FDA_GPR_modules.pip0_dataloading import DataLoader
        from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
        from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor
    except ImportError as e:
        print(f"  ERROR: Failed to import modules: {e}")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.1: Load data (pip0)
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.1] Loading data...")
    
    if not input_folder.exists():
        print(f"  ERROR: Input folder does not exist: {input_folder}")
        return False
    
    # Get filename parsing config for extracting grouping keys from filenames
    filename_parsing_config = settings_manager.get_filename_parsing_config()
    
    loader = DataLoader(
        path_to_folder=input_folder,
        verbose=verbose,
        filename_parsing_config=filename_parsing_config,
    )
    loading_result = loader.load_all()
    
    print(f"  Loaded {loading_result.num_curves} curves")
    print(f"  Group keys: {loading_result.groups}")
    
    if loading_result.num_curves == 0:
        print("  ERROR: No curves loaded. Check input folder.")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.2: Preprocess (pip1) - config from SettingsManager
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.2] Preprocessing...")
    
    preproc_cfg = settings_manager.get_preproc_config()
    print(f"  Y-Scaling: {preproc_cfg.y_scaling_method}")
    
    preprocessor = DataPreprocessor(config=preproc_cfg, verbose=verbose)
    preproc_result = preprocessor.preprocess_all(loading_result.curves)
    
    print(f"  Preprocessed {preproc_result.num_curves} curves")
    print(f"  Skipped {preproc_result.num_skipped} curves")
    
    if preproc_result.num_curves == 0:
        print("  ERROR: No valid curves after preprocessing.")
        return False
    
    # -------------------------------------------------------------------------
    # Step 1.3: Fit Individual GPRs (pip2) - config from SettingsManager
    # -------------------------------------------------------------------------
    print(f"\n[Step 1.3] Fitting Individual GPR models...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    gpr_cfg, export_cfg = settings_manager.get_gpr_configs()
    
    processor = IndividualGPRProcessor(
        gpr_config=gpr_cfg,
        export_config=export_cfg,
        output_directory=output_dir,
        verbose=verbose,
    )
    
    gpr_result = processor.fit_all(
        preproc_result.curves,
        export_results=True,
        plot_results=True,
    )
    
    print(f"\n  Successfully fitted: {gpr_result.num_results} curves")
    print(f"  Skipped: {gpr_result.num_skipped} curves")
    print(f"  Output directory: {output_dir}")
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(output_dir, "settings_used.json")
    print(f"  Settings saved to: {output_dir / 'settings_used.json'}")
    
    return gpr_result.num_results > 0


# =============================================================================
# Step 2: Batch Grid Testing (pip5)
# =============================================================================

def _filter_testing_grid(
    grid: list,
    combo_tags: list | None = None,
    norm_filter: str | None = None,
    method_filter: str | None = None,
) -> list:
    """
    Filter a testing grid by explicit tags, normalization, and/or method.

    Parameters
    ----------
    grid : list[BatchTestingOptions]
        Full testing grid.
    combo_tags : list[str] or None
        If given, keep only combos whose tag matches one of these strings.
    norm_filter : {'norm', 'real'} or None
        If given, keep only combos with matching normalization.
    method_filter : {'iterative', 'fgpr', 'student_t'} or None
        If given, keep only combos with matching aggregation_method.

    Returns
    -------
    list[BatchTestingOptions]
        Filtered grid (may be empty).
    """
    filtered = list(grid)
    if combo_tags:
        filtered = [opt for opt in filtered if opt.tag in combo_tags]
    if norm_filter:
        want_norm = norm_filter == "norm"
        filtered = [opt for opt in filtered if opt.normalization_summary == want_norm]
    if method_filter:
        filtered = [opt for opt in filtered if opt.aggregation_method == method_filter]
    return filtered


def run_step2_batch_testing(
    settings_manager,
    verbose: bool = True,
    combo_tags: list | None = None,
    norm_filter: str | None = None,
    method_filter: str | None = None,
    run_summary_gpr: bool = True,
    run_efficiency_eval: bool = True,
) -> bool:
    """
    Step 2: Run Batch Grid Testing (pip5).
    
    Tests combinations of weight_mode/weight_scope/normalization.
    Runs Summary GPR and/or Efficiency Evaluation for each combination.
    
    Args:
        settings_manager: SettingsManager instance with loaded settings
        verbose: Whether to print progress
        combo_tags: If given, run only these exact combo tags
        norm_filter: 'norm' or 'real' to restrict normalization mode
        method_filter: 'iterative', 'fgpr', or 'student_t' to restrict aggregation method
        run_summary_gpr: Whether to run Summary GPR (step 2a)
        run_efficiency_eval: Whether to run Learning Curve (step 2b)
    
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("STEP 2: Batch Grid Testing (pip5)")
    print("=" * 60)
    
    input_dir = settings_manager.get_pip2_output_dir()
    output_base = settings_manager.get_base_output_dir()
    output_dir = output_base / "batch_testing"
    
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    
    # -------------------------------------------------------------------------
    # Import modules
    # -------------------------------------------------------------------------
    try:
        from py_FDA_GPR_modules.pip5_batch_setting_grid_testing import (
            BatchDirParams,
            BatchRunConfig,
            BatchTestingOrchestrator,
            build_testing_grid,
            discover_groupkeys,
        )
    except ImportError as e:
        print(f"  ERROR: Failed to import pip5 modules: {e}")
        return False
    
    # -------------------------------------------------------------------------
    # Check input
    # -------------------------------------------------------------------------
    if not input_dir.exists():
        print(f"  ERROR: Individual GPR directory does not exist: {input_dir}")
        print("  Please run Step 1 first.")
        return False
    
    # Discover group keys
    group_keys = discover_groupkeys(input_dir)
    print(f"  Found {len(group_keys)} group keys: {group_keys}")
    
    if len(group_keys) == 0:
        print("  ERROR: No group keys found. Check input folder.")
        return False
    
    # -------------------------------------------------------------------------
    # Build testing grid (read include_fgpr from pip5 settings)
    # -------------------------------------------------------------------------
    print(f"\n[Step 2.1] Building testing grid...")
    
    pip5_settings = settings_manager.get_raw("pip5_batch_testing")
    include_fgpr = pip5_settings.get("include_fgpr", False)
    include_student_t = pip5_settings.get("include_student_t", False)

    parts = ["6 iterative"]
    if include_fgpr:
        parts.append("2 FGPR")
    if include_student_t:
        parts.append("2 Student-t")
    total = 6 + (2 if include_fgpr else 0) + (2 if include_student_t else 0)
    grid_desc = " + ".join(parts) + f" = {total}"
    print(f"  include_fgpr: {include_fgpr} (from settings)")
    print(f"  include_student_t: {include_student_t} (from settings)")

    testing_grid = build_testing_grid(
        include_fgpr=include_fgpr,
        include_student_t=include_student_t,
    )

    # Apply combo filters (--combos / --norm / --method)
    if combo_tags or norm_filter or method_filter:
        testing_grid = _filter_testing_grid(
            testing_grid,
            combo_tags=combo_tags,
            norm_filter=norm_filter,
            method_filter=method_filter,
        )
        if not testing_grid:
            print("  ERROR: No combinations remain after filtering.")
            print("  Check --combos / --norm / --method flags.")
            return False
        grid_desc = f"{len(testing_grid)} (filtered)"

    print(f"  {len(testing_grid)} combinations to test ({grid_desc}):")
    for opt in testing_grid:
        print(f"    - {opt.tag}")
    
    # -------------------------------------------------------------------------
    # Configure batch testing
    # -------------------------------------------------------------------------
    print(f"\n[Step 2.2] Configuring batch testing...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get configs from SettingsManager
    summary_gpr_config, summary_gpr_hyperparams = settings_manager.get_summary_gpr_configs(
        input_directory=input_dir,
        output_directory=output_dir,
    )
    
    _, _, efficiency_scapara, _ = settings_manager.get_efficiency_configs(
        indiv_dir=input_dir,
        output_dir=output_dir,
    )
    
    dir_params = BatchDirParams(
        input_dir=input_dir,
        base_output_dir=output_dir,
    )
    
    # Configure what to run (Summary GPR + Efficiency Evaluation)
    run_config = BatchRunConfig(
        run_summary_gpr=run_summary_gpr,
        run_efficiency_eval=run_efficiency_eval,
        run_combine_gprs=False,
        run_comparison_plots=True,
        export_summary_csvs=True,
        copy_artifacts=True,
    )
    
    print(f"  run_summary_gpr: {run_config.run_summary_gpr}")
    print(f"  run_efficiency_eval: {run_config.run_efficiency_eval}")
    print(f"  run_comparison_plots: {run_config.run_comparison_plots}")
    
    # -------------------------------------------------------------------------
    # Run batch testing
    # -------------------------------------------------------------------------
    print(f"\n[Step 2.3] Running batch testing...")
    
    orchestrator = BatchTestingOrchestrator(
        dir_params=dir_params,
        summary_gpr_config=summary_gpr_config,
        summary_gpr_hyperparams=summary_gpr_hyperparams,
        efficiency_scapara=efficiency_scapara,
        run_config=run_config,
        testing_grid=testing_grid,
        settings_manager=settings_manager,
        verbose=verbose,
    )
    
    results = orchestrator.run_all()
    
    # -------------------------------------------------------------------------
    # Results summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch Testing Results Summary")
    print("=" * 60)
    
    for tag, combo_result in results.items():
        if tag.startswith("_"):
            continue
        
        print(f"\n{tag}:")
        if "summary_gpr" in combo_result:
            n_groups = len(combo_result["summary_gpr"])
            print(f"  Summary GPR: {n_groups} groups processed")
        if "efficiency" in combo_result:
            n_groups = len(combo_result["efficiency"])
            print(f"  Efficiency: {n_groups} groups processed")
    
    # Show aggregate outputs
    if "_aggregate_csvs" in results:
        agg = results["_aggregate_csvs"]
        n_summary = len(agg.get("summary_gpr", {}))
        n_eff = len(agg.get("efficiency", {}))
        print(f"\nAggregate CSVs: {n_summary} Summary GPR, {n_eff} Efficiency")
    
    if "_comparison_plots" in results:
        plots = results["_comparison_plots"]
        for ptype, paths in plots.items():
            print(f"Comparison Plots ({ptype}): {len(paths)} plots")
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(output_dir, "settings_used.json")
    print(f"\nSettings saved to: {output_dir / 'settings_used.json'}")
    print(f"Output directory: {output_dir}")
    
    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def run_single_folder(input_folder: Path, settings_file: Path, step: str, verbose: bool) -> bool:
    """Run the pipeline on a single folder."""
    from py_FDA_GPR_modules.pip0_dataloading import SettingsManager
    
    settings_manager = SettingsManager(
        settings_path=settings_file,
        input_folder=input_folder,
    )
    
    print("\n" + "=" * 70)
    print(f"Processing: {input_folder.name}")
    print("=" * 70)
    print(f"Input Folder:     {input_folder}")
    print(f"Output Base:      {settings_manager.get_base_output_dir()}")
    print(f"Settings File:    {settings_file}")
    
    success = True
    
    if step in ["all", "1"]:
        success = run_step1_individual_gpr(
            settings_manager=settings_manager,
            input_folder=input_folder,
            verbose=verbose,
        )
    
    if success and step in ["all", "2"]:
        success = run_step2_batch_testing(
            settings_manager=settings_manager,
            verbose=verbose,
        )
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description=(
            "GPR-FDA: Gaussian Process Regression for Functional Data Analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on single folder
    python main_GPR_FDA_pipeline.py --path "./_input/CCNF_CTAB_PT/20230616 FeNiCoCu acidB-BWR6"

  # Process ALL subfolders in a parent directory (batch mode)
    python main_GPR_FDA_pipeline.py --path "./_input/CCNF_CTAB_PT" --batch

  # Individual GPR only (Step 1)
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 1

  # Batch testing only (Step 2)
    python main_GPR_FDA_pipeline.py --path /path/to/folder --step 2

  # Summary GPR only (Step 2a) or Learning Curve only (Step 2b)
    python main_GPR_FDA_pipeline.py --path /path -s 2a --method fgpr
    python main_GPR_FDA_pipeline.py --path /path -s 2b --combos NS_norm__AM_fgpr

  # Step 2 — only FGPR combos
    python main_GPR_FDA_pipeline.py --path /path -s 2 --method fgpr

  # Step 2 — only observation-scale iterative combos
    python main_GPR_FDA_pipeline.py --path /path -s 2 --norm real --method iterative

  # Step 2 — specific combos by tag
    python main_GPR_FDA_pipeline.py --path /path -s 2 --combos NS_norm__AM_fgpr NS_real__AM_fgpr

Output Structure:
  {folder_name}_output/
    individual_GPR/           <- Step 1 output
    batch_testing/            <- Step 2 output (6 baseline, up to 10 total)
      NS_norm__WM_equal__WS_curve/
      NS_norm__WM_iterative__WS_curve/
      NS_norm__WM_iterative__WS_point/
      NS_real__WM_equal__WS_curve/
      NS_real__WM_iterative__WS_curve/
      NS_real__WM_iterative__WS_point/
      comparisons/
"""
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        required=True,
        help="Path to the folder containing .cor files (or parent folder with --batch)"
    )
    
    parser.add_argument(
        "--step", "-s",
        type=str,
        choices=["all", "1", "2", "2a", "2b"],
        default="all",
        help=(
            "Which step(s) to run: all, 1 (individual GPR), "
            "2 (summary GPR + learning curve), "
            "2a (summary GPR only), 2b (learning curve only)"
        ),
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all subdirectories in the given path (batch mode)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    # --- Step 2 combo filters ---
    parser.add_argument(
        "--combos", "-c",
        nargs="+",
        type=str,
        default=None,
        metavar="TAG",
        help=(
            "Run only these Step 2 combo tags (e.g. NS_norm__AM_fgpr "
            "NS_real__WM_iterative__WS_point). "
            "Ignored when running Step 1 only."
        ),
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=["norm", "real"],
        default=None,
        help="Filter Step 2 combos by normalization mode: norm or real",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["iterative", "fgpr", "student_t"],
        default=None,
        help="Filter Step 2 combos by aggregation method: iterative, fgpr, or student_t",
    )

    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # Setup paths and load settings via SettingsManager
    # -------------------------------------------------------------------------
    input_folder = Path(args.path).resolve()
    
    if not input_folder.exists():
        print(f"ERROR: Input path does not exist: {input_folder}")
        sys.exit(1)
    
    verbose = not args.quiet
    step = args.step.lower()
    combo_tags = args.combos
    norm_filter = args.norm
    method_filter = args.method
    
    # -------------------------------------------------------------------------
    # Print header
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GPR-FDA: Gaussian Process Regression - Functional Data Analysis")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Find settings file
    # -------------------------------------------------------------------------
    settings_file = input_folder / "settings.json"
    if not settings_file.exists():
        settings_files = list(input_folder.glob("settings*.json"))
        if not settings_files:
            # Also match files like DEBUG_settings.json
            settings_files = list(input_folder.glob("*settings*.json"))
        if settings_files:
            settings_file = settings_files[0]
        else:
            settings_file = input_folder.parent / "settings.json"
            if not settings_file.exists():
                settings_files = list(input_folder.parent.glob("settings*.json"))
                if not settings_files:
                    settings_files = list(input_folder.parent.glob("*settings*.json"))
                if settings_files:
                    settings_file = settings_files[0]
    
    if not settings_file.exists():
        print(f"ERROR: No settings*.json found in {input_folder} or {input_folder.parent}")
        print("Please create a settings.json file.")
        sys.exit(1)
    
    print(f"Settings File: {settings_file}")
    
    # -------------------------------------------------------------------------
    # Batch mode: combine all subdirectories into one dataset
    # -------------------------------------------------------------------------
    if args.batch:
        subdirs = sorted([d for d in input_folder.iterdir() if d.is_dir()])
        print(f"Batch Mode: Combining {len(subdirs)} subdirectories into one dataset")
        for sd in subdirs:
            print(f"  - {sd.name}")
        
        # Use the parent folder name for output
        from py_FDA_GPR_modules.pip0_dataloading import SettingsManager
        
        try:
            settings_manager = SettingsManager(
                settings_path=settings_file,
                input_folder=input_folder,  # Parent folder as input reference
            )
        except Exception as e:
            print(f"ERROR: Failed to load settings: {e}")
            sys.exit(1)
        
        print(f"\nOutput Base: {settings_manager.get_base_output_dir()}")
        print(f"Step(s) to run: {step}")
        
        # Run pipeline with combined data from all subdirectories
        success = run_step1_individual_gpr_batch(
            settings_manager=settings_manager,
            subdirectories=subdirs,
            verbose=verbose,
        ) if step in ["all", "1"] else True
        
        if success and step in ["all", "2", "2a", "2b"]:
            success = run_step2_batch_testing(
                settings_manager=settings_manager,
                verbose=verbose,
                combo_tags=combo_tags,
                norm_filter=norm_filter,
                method_filter=method_filter,
                run_summary_gpr=(step in ["all", "2", "2a"]),
                run_efficiency_eval=(step in ["all", "2", "2b"]),
            )
        
        print("\n" + "=" * 70)
        if success:
            print("Batch Pipeline completed successfully!")
            print(f"Output: {settings_manager.get_base_output_dir()}")
        else:
            print("Batch Pipeline completed with errors.")
            sys.exit(1)
        print("=" * 70)
    else:
        # -------------------------------------------------------------------------
        # Single folder mode
        # -------------------------------------------------------------------------
        from py_FDA_GPR_modules.pip0_dataloading import SettingsManager
        
        try:
            settings_manager = SettingsManager(
                settings_path=settings_file,
                input_folder=input_folder,
            )
        except Exception as e:
            print(f"ERROR: Failed to load settings: {e}")
            sys.exit(1)
        
        print(f"Input Folder:     {input_folder}")
        print(f"Output Base:      {settings_manager.get_base_output_dir()}")
        print(f"Step(s) to run:   {step}")
        
        success = True
        
        if step in ["all", "1"]:
            success = run_step1_individual_gpr(
                settings_manager=settings_manager,
                input_folder=input_folder,
                verbose=verbose,
            )
        
        if success and step in ["all", "2", "2a", "2b"]:
            success = run_step2_batch_testing(
                settings_manager=settings_manager,
                verbose=verbose,
                combo_tags=combo_tags,
                norm_filter=norm_filter,
                method_filter=method_filter,
                run_summary_gpr=(step in ["all", "2", "2a"]),
                run_efficiency_eval=(step in ["all", "2", "2b"]),
            )
        
        print("\n" + "=" * 70)
        if success:
            print("Pipeline completed successfully!")
            print(f"Output: {settings_manager.get_base_output_dir()}")
        else:
            print("Pipeline completed with errors.")
            sys.exit(1)
        print("=" * 70)


if __name__ == "__main__":
    main()
