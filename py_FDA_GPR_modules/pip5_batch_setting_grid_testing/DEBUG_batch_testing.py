# pip5_batch_setting_grid_testing/DEBUG_batch_testing.py
"""
DEBUG script for batch grid testing.

This script tests the pip5 batch testing module by running all 8 permutations
(6 iterative + 2 FGPR) using the output from pip2 DEBUG as input.

Uses SettingsManager to load all configuration from settings.json.
Settings are automatically saved to output for reproducibility.

Output: DEBUG_pip5_Batch_Setting_Output/ (in same folder as this script)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Settings file location
SETTINGS_FILE = PROJECT_ROOT / "py_FDA_GPR_modules" / "pip0_dataloading" / "DEBUG_pip0_input" / "DEBUG_settings.json"

# pip5 uses SCRIPT_DIR-relative output since it's not in the normal output_structure
OUTPUT_FOLDER = SCRIPT_DIR / "DEBUG_pip5_Batch_Setting_Output"


def main():
    """Run DEBUG batch testing."""
    print("=" * 60)
    print("DEBUG: Batch Grid Testing (pip5)")
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
        from py_FDA_GPR_modules.pip5_batch_setting_grid_testing import (
            BatchDirParams,
            BatchRunConfig,
            BatchTestingOrchestrator,
            build_testing_grid,
            discover_groupkeys,
        )
        print("  + pip5_batch_setting_grid_testing imported")
    except ImportError as e:
        print(f"  X Failed to import pip5: {e}")
        return
    
    try:
        from py_FDA_GPR_modules.pip4_efficiency_eval import ScaleParams
        print("  + pip4_efficiency_eval imported")
    except ImportError as e:
        print(f"  X Failed to import pip4: {e}")
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
    
    # Get input folder from settings (pip2 output)
    input_folder = settings_manager.get_pip2_output_dir()
    print(f"  + Input folder: {input_folder}")
    print(f"  + Output folder: {OUTPUT_FOLDER}")
    
    # -------------------------------------------------------------------------
    # Step 1: Check input
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] Checking input: {input_folder}")
    
    if not input_folder.exists():
        print(f"  X Input folder does not exist: {input_folder}")
        print("  Please run DEBUG_individual_gpr.py first.")
        return
    
    # Discover potentials
    potentials = discover_groupkeys(input_folder)
    print(f"  Found {len(potentials)} potentials: {potentials}")
    
    if len(potentials) == 0:
        print("  X No potentials found. Check input folder.")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Build testing grid
    # -------------------------------------------------------------------------
    # Read pip5 settings from JSON (with defaults)
    pip5_settings = settings_manager.get_raw("pip5_batch_testing")
    include_fgpr = pip5_settings.get("include_fgpr", False)
    
    grid_desc = "6 iterative + 2 FGPR = 8" if include_fgpr else "6 iterative"
    print(f"\n[Step 2] Building testing grid ({grid_desc} combinations)...")
    print(f"  include_fgpr: {include_fgpr} (from settings)")
    
    testing_grid = build_testing_grid(include_fgpr=include_fgpr)
    print(f"  {len(testing_grid)} combinations:")
    for opt in testing_grid:
        print(f"    - {opt.tag}")
    
    # -------------------------------------------------------------------------
    # Step 3: Configure batch testing - config from SettingsManager
    # -------------------------------------------------------------------------
    print("\n[Step 3] Configuring batch testing...")
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get configs from SettingsManager
    summary_gpr_config, summary_gpr_hyperparams = settings_manager.get_summary_gpr_configs(
        input_directory=input_folder,
        output_directory=OUTPUT_FOLDER,
    )
    
    _, _, efficiency_scapara, _ = settings_manager.get_efficiency_configs(
        indiv_dir=input_folder,
        output_dir=OUTPUT_FOLDER,
    )
    
    dir_params = BatchDirParams(
        input_dir=input_folder,
        base_output_dir=OUTPUT_FOLDER,
    )
    
    # Configure what to run - read from pip5 settings with defaults
    run_config = BatchRunConfig(
        run_summary_gpr=pip5_settings.get("run_summary_gpr", True),
        run_efficiency_eval=pip5_settings.get("run_efficiency_eval", True),
        run_combine_gprs=pip5_settings.get("run_combine_gprs", False),
        run_comparison_plots=pip5_settings.get("run_comparison_plots", True),
        export_summary_csvs=pip5_settings.get("export_summary_csvs", True),
        copy_artifacts=pip5_settings.get("copy_artifacts", True),
    )
    
    print(f"  run_summary_gpr: {run_config.run_summary_gpr}")
    print(f"  run_efficiency_eval: {run_config.run_efficiency_eval}")
    print(f"  run_comparison_plots: {run_config.run_comparison_plots}")
    print(f"  export_summary_csvs: {run_config.export_summary_csvs}")
    
    # -------------------------------------------------------------------------
    # Step 4: Run batch testing
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running batch testing...")
    
    orchestrator = BatchTestingOrchestrator(
        dir_params=dir_params,
        summary_gpr_config=summary_gpr_config,
        summary_gpr_hyperparams=summary_gpr_hyperparams,
        efficiency_scapara=efficiency_scapara,
        run_config=run_config,
        testing_grid=testing_grid,
        settings_manager=settings_manager,  # Pass for reproducibility
        verbose=True,
    )
    
    results = orchestrator.run_all()
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(OUTPUT_FOLDER, "settings_used.json")
    print(f"\nSettings saved to: {OUTPUT_FOLDER / 'settings_used.json'}")
    
    # -------------------------------------------------------------------------
    # Step 5: Results Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 5] Results Summary")
    print("=" * 60)
    
    for tag, combo_result in results.items():
        if tag.startswith("_"):
            continue  # Skip internal keys like _aggregate_csvs
        
        print(f"\n{tag}:")
        print(f"  Output: {combo_result.get('output_dir', 'N/A')}")
        
        if "summary_gpr" in combo_result:
            n_groups = len(combo_result["summary_gpr"])
            print(f"  Summary GPR: {n_groups} groups processed")
        elif "summary_gpr_error" in combo_result:
            print(f"  Summary GPR: ERROR - {combo_result['summary_gpr_error']}")
        
        if "efficiency" in combo_result:
            n_pots = len(combo_result["efficiency"])
            print(f"  Efficiency: {n_pots} potentials processed")
    
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
    
    # -------------------------------------------------------------------------
    # Step 6: List output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 6] Output Structure")
    print("=" * 60)
    
    print(f"\nOutput: {OUTPUT_FOLDER}")
    for item in sorted(OUTPUT_FOLDER.iterdir()):
        if item.is_dir():
            n_files = len(list(item.glob("*")))
            print(f"  {item.name}/  ({n_files} files)")
    
    print("\n" + "=" * 60)
    print("DEBUG completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
