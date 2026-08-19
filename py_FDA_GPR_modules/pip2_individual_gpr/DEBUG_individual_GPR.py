# pip2_individual_gpr/DEBUG_individual_gpr.py
"""
DEBUG Script for Individual GPR Processing.

This standalone script demonstrates the full pipeline:
    pip0 (DataLoader) -> pip1 (DataPreprocessor) -> pip2 (IndividualGPRProcessor)

Uses SettingsManager to load all configuration from DEBUG_settings.json.
Settings are automatically saved to output for reproducibility.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Settings file location
SETTINGS_FILE = PROJECT_ROOT / "py_FDA_GPR_modules" / "pip0_dataloading" / "DEBUG_pip0_input" / "DEBUG_settings.json"


def main():
    print("=" * 60)
    print("DEBUG: Individual GPR Processing Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Step 0: Import modules and load settings via SettingsManager
    # -------------------------------------------------------------------------
    print("\n[Step 0] Importing modules and loading settings...")
    
    try:
        from py_FDA_GPR_modules.pip0_dataloading import DataLoader, SettingsManager
        print("  + pip0_dataloading imported")
    except ImportError as e:
        print(f"  X Failed to import pip0_dataloading: {e}")
        return
    
    try:
        from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
        print("  + pip1_datapreprocessing imported")
    except ImportError as e:
        print(f"  X Failed to import pip1_datapreprocessing: {e}")
        return
    
    try:
        from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor
        print("  + pip2_individual_gpr imported")
    except ImportError as e:
        print(f"  X Failed to import pip2_individual_gpr: {e}")
        return
    
    # Load settings via SettingsManager
    if not SETTINGS_FILE.exists():
        print(f"  X Settings file not found: {SETTINGS_FILE}")
        return
    
    input_folder = SETTINGS_FILE.parent  # DEBUG_pip0_input folder
    settings_manager = SettingsManager(
        settings_path=SETTINGS_FILE,
        input_folder=input_folder,
    )
    print(f"  + Settings loaded from: {SETTINGS_FILE}")
    
    # Get paths from settings
    output_folder = settings_manager.get_pip2_output_dir()
    print(f"  + Output folder: {output_folder}")
    
    # Get filename parsing config
    filename_parsing_config = settings_manager.get_filename_parsing_config()
    print(f"  + Filename parsing: {len(filename_parsing_config.get('grouping_keys', []))} grouping keys, "
          f"{len(filename_parsing_config.get('metadata_keys', []))} metadata keys")
    
    # -------------------------------------------------------------------------
    # Step 1: Load data (pip0)
    # -------------------------------------------------------------------------
    print(f"\n[Step 1] Loading data from: {input_folder}")
    
    if not input_folder.exists():
        print(f"  X Input folder does not exist: {input_folder}")
        return
    
    loader = DataLoader(
        path_to_folder=input_folder,
        verbose=True,
        filename_parsing_config=filename_parsing_config
    )
    loading_result = loader.load_all()
    
    print(f"  Loaded {loading_result.num_curves} curves")
    print(f"  Primary key values: {loading_result.primary_key_values}")
    print(f"  Groups: {loading_result.groups}")
    
    # Show sample group_flags from first curve
    if loading_result.curves:
        sample_curve = loading_result.curves[0]
        print(f"  Sample group_flags: {sample_curve.group_flags}")
        print(f"  Sample metadata: {sample_curve.metadata}")
    
    if loading_result.num_curves == 0:
        print("  X No curves loaded. Check input folder.")
        return
    
    # -------------------------------------------------------------------------
    # Step 2: Preprocessing (pip1) - config from SettingsManager
    # -------------------------------------------------------------------------
    print("\n[Step 2] Preprocessing data...")
    
    preproc_cfg = settings_manager.get_preproc_config()
    print(f"  Y-Scaling: {preproc_cfg.y_scaling_method}")
    print(f"  X-Scaling: {preproc_cfg.x_scaling_method}")
    
    preprocessor = DataPreprocessor(config=preproc_cfg, verbose=True)
    preprocessing_result = preprocessor.preprocess_all(loading_result.curves)
    
    print(f"  Preprocessed {preprocessing_result.num_curves} curves")
    print(f"  Skipped {preprocessing_result.num_skipped} curves")
    
    if preprocessing_result.num_curves == 0:
        print("  X No curves preprocessed. Check data quality.")
        return
    
    # -------------------------------------------------------------------------
    # Step 3: Individual GPR Fitting (pip2) - config from SettingsManager
    # -------------------------------------------------------------------------
    print("\n[Step 3] Fitting GPR models...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    gpr_cfg, export_cfg = settings_manager.get_gpr_configs()
    print(f"  n_restarts_optimizer: {gpr_cfg.n_restarts_optimizer}")
    print(f"  alpha: {gpr_cfg.alpha}")
    
    gpr_processor = IndividualGPRProcessor(
        gpr_config=gpr_cfg,
        export_config=export_cfg,
        output_directory=output_folder,
        verbose=True,
    )
    
    gpr_result = gpr_processor.fit_all(
        preprocessing_result.curves,
        export_results=True,
        plot_results=True,
    )
    
    # -------------------------------------------------------------------------
    # Step 4: Results Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 4] Results Summary")
    print("=" * 60)
    print(gpr_result.summary())
    
    # Save settings to output for reproducibility
    settings_manager.save_to_output(output_folder, "settings_used.json")
    print(f"\nSettings saved to: {output_folder / 'settings_used.json'}")
    print(f"Output directory: {output_folder}")


if __name__ == "__main__":
    main()
