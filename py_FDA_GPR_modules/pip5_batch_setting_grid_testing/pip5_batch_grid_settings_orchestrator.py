# pip5_batch_setting_grid_testing/pip5_batch_grid_settings_orchestrator.py
"""
High-level orchestrator for batch grid testing.

Runs Summary GPR and efficiency evaluation across all testing option combinations.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace
from typing import Dict, List, Optional, Any, Union

from .batch_config import (
    BatchDirParams,
    BatchTestingOptions,
    BatchRunConfig,
)
from .batch_core import (
    build_testing_grid,
    discover_groupkeys,
    copy_artifacts,
    apply_options_to_config,
)

from .batch_comparison_plotting import (
    plot_all_comparisons,
    export_all_aggregate_csvs,
    BatchComparisonConfig,
)

# Import from pip3 for Summary GPR
from ..pip3_FDA_scoring_and_aggregations import (
    SummaryGPRConfig,
    SummaryGPRHyperParams,
    SummaryGPROrchestrator,
)

# Import from pip4 for efficiency evaluation
from ..pip4_efficiency_eval import (
    DirParams as EffDirParams,
    GlobalParams,
    ScaleParams,
    PlotParams,
    EfficiencyOrchestrator,
)

# Import settings manager
from ..pip0_dataloading.settings_manager import SettingsManager


class BatchTestingOrchestrator:
    """
    Orchestrator for running batch grid testing across all option combinations.
    
    Example
    -------
    >>> from py_FDA_GPR_modules.pip5_batch_setting_grid_testing import BatchTestingOrchestrator, BatchDirParams
    >>> 
    >>> # Option 1: Manual config
    >>> dir_params = BatchDirParams(input_dir=Path("output/individual_gprs"))
    >>> orchestrator = BatchTestingOrchestrator(dir_params)
    >>> results = orchestrator.run_all()
    >>> 
    >>> # Option 2: From settings JSON
    >>> orchestrator = BatchTestingOrchestrator.from_settings(
    ...     settings_path="_input/settings_template.json",
    ...     input_dir=Path("output/individual_gprs"),
    ... )
    >>> results = orchestrator.run_all()
    """
    
    def __init__(
        self,
        dir_params: BatchDirParams,
        *,
        summary_gpr_config: Optional[SummaryGPRConfig] = None,
        summary_gpr_hyperparams: Optional[SummaryGPRHyperParams] = None,
        efficiency_globpara: Optional[GlobalParams] = None,
        efficiency_scapara: Optional[ScaleParams] = None,
        efficiency_plotpara: Optional[PlotParams] = None,
        run_config: Optional[BatchRunConfig] = None,
        testing_grid: Optional[List[BatchTestingOptions]] = None,
        settings_manager: Optional[SettingsManager] = None,
        verbose: bool = True,
    ):
        """
        Initialize batch testing orchestrator.
        
        Parameters
        ----------
        dir_params : BatchDirParams
            Directory parameters.
        summary_gpr_config : SummaryGPRConfig, optional
            Base configuration for Summary GPR. If None, uses defaults.
        summary_gpr_hyperparams : SummaryGPRHyperParams, optional
            Hyperparameters for Summary GPR. If None, uses defaults.
        efficiency_globpara : GlobalParams, optional
            Global params for efficiency evaluation.
        efficiency_scapara : ScaleParams, optional
            Scale params for efficiency evaluation.
        efficiency_plotpara : PlotParams, optional
            Plot params for efficiency evaluation.
        run_config : BatchRunConfig, optional
            What to run in each batch iteration.
        testing_grid : List[BatchTestingOptions], optional
            List of option combinations to test. If None, uses default grid.
        settings_manager : SettingsManager, optional
            Settings manager for saving settings to output.
        verbose : bool
            Print progress.
        """
        self.dir_params = dir_params
        self.verbose = verbose
        self.settings_manager = settings_manager
        
        # Summary GPR settings
        self.summary_gpr_config = summary_gpr_config or SummaryGPRConfig(
            input_directory=dir_params.input_dir
        )
        self.summary_gpr_hyperparams = summary_gpr_hyperparams or SummaryGPRHyperParams()
        
        # Efficiency evaluation settings
        self.efficiency_globpara = efficiency_globpara or GlobalParams()
        self.efficiency_scapara = efficiency_scapara or ScaleParams()
        self.efficiency_plotpara = efficiency_plotpara or PlotParams()
        
        # Run configuration
        self.run_config = run_config or BatchRunConfig()
        
        # Testing grid
        self.testing_grid = testing_grid or build_testing_grid()
    
    @classmethod
    def from_settings(
        cls,
        settings_path: Optional[Union[str, Path]] = None,
        input_folder: Optional[Union[str, Path]] = None,
        input_dir: Optional[Path] = None,
        output_base_dir: Optional[Path] = None,
        *,
        run_config: Optional[BatchRunConfig] = None,
        testing_grid: Optional[List[BatchTestingOptions]] = None,
        verbose: bool = True,
    ) -> "BatchTestingOrchestrator":
        """
        Create orchestrator from a settings JSON file or input folder.
        
        Parameters
        ----------
        settings_path : str or Path, optional
            Path to settings JSON file. Use this OR input_folder.
        input_folder : str or Path, optional
            Path to input data folder (e.g., "_input/CCNF_CTAB_PT").
            Auto-detects settings file and generates output folder name.
        input_dir : Path, optional
            Directory containing individual GPR CSVs.
            Defaults to pip2 output directory from settings.
        output_base_dir : Path, optional
            Base directory for outputs. Defaults to pip5 output from settings.
        run_config : BatchRunConfig, optional
            What to run in each batch iteration.
        testing_grid : List[BatchTestingOptions], optional
            List of option combinations to test.
        verbose : bool
            Print progress.
            
        Returns
        -------
        BatchTestingOrchestrator
        
        Example
        -------
        >>> # Option 1: From input folder (recommended)
        >>> orchestrator = BatchTestingOrchestrator.from_settings(
        ...     input_folder="_input/CCNF_CTAB_PT"
        ... )
        >>> 
        >>> # Option 2: From specific settings file
        >>> orchestrator = BatchTestingOrchestrator.from_settings(
        ...     settings_path="_input/settings.json",
        ...     input_dir=Path("_output/individual_gpr"),
        ... )
        """
        # Create settings manager
        if input_folder is not None:
            manager = SettingsManager.from_input_folder(input_folder)
        elif settings_path is not None:
            manager = SettingsManager(settings_path=settings_path)
        else:
            raise ValueError("Must provide either settings_path or input_folder")
        
        # Determine input/output directories
        if input_dir is None:
            input_dir = manager.get_pip2_output_dir()
        if output_base_dir is None:
            output_base_dir = manager.get_pip5_output_dir()
        
        dir_params = BatchDirParams(
            input_dir=input_dir,
            base_output_dir=output_base_dir,
        )
        
        # Get configs from settings
        summary_cfg, summary_hp = manager.get_summary_gpr_configs(
            input_directory=input_dir,
            output_directory=output_base_dir,
        )
        
        _, glob_params, scale_params, plot_params = manager.get_efficiency_configs(
            indiv_dir=input_dir,
            output_dir=output_base_dir,
        )
        
        return cls(
            dir_params=dir_params,
            summary_gpr_config=summary_cfg,
            summary_gpr_hyperparams=summary_hp,
            efficiency_globpara=glob_params,
            efficiency_scapara=scale_params,
            efficiency_plotpara=plot_params,
            run_config=run_config,
            testing_grid=testing_grid,
            settings_manager=manager,
            verbose=verbose,
        )
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def run_single_combo(
        self,
        options: BatchTestingOptions,
    ) -> Dict[str, Any]:
        """
        Run all processing for a single option combination.
        
        Parameters
        ----------
        options : BatchTestingOptions
            Testing options for this run.
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary with keys for each processing step.
        """
        tag = options.tag
        output_dir = self.dir_params.get_combo_output_dir(tag)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subfolders for Summary GPR and Learning Curve
        summary_gpr_dir = output_dir / "summary_gpr"
        learning_curve_dir = output_dir / "learning_curve"
        os.makedirs(summary_gpr_dir, exist_ok=True)
        os.makedirs(learning_curve_dir, exist_ok=True)
        
        self._log(f"\n{'='*60}")
        self._log(f"Processing: {tag}")
        self._log(f"{'='*60}")
        
        results: Dict[str, Any] = {"tag": tag, "output_dir": output_dir}
        
        # Apply options to config
        combo_config = apply_options_to_config(options, self.summary_gpr_config)
        combo_config = replace(combo_config, output_directory=summary_gpr_dir)
        
        # Run Summary GPR
        if self.run_config.run_summary_gpr:
            self._log("\n[Step 1] Running Summary GPR...")
            try:
                summary_orchestrator = SummaryGPROrchestrator(
                    combo_config,
                    self.summary_gpr_hyperparams,
                    verbose=self.verbose,
                )
                summary_results = summary_orchestrator.process_all()
                results["summary_gpr"] = summary_results
                self._log(f"  + Processed {len(summary_results)} groups")
            except Exception as e:
                self._log(f"  X Summary GPR failed: {e}")
                results["summary_gpr_error"] = str(e)
        
        # Run Efficiency Evaluation (Learning Curve)
        # Skip learning curves for FGPR combos — precision-weighted
        # pooling is not meaningful in subset learning-curve analysis.
        skip_eff_for_advanced = (options.aggregation_method == "fgpr")
        if self.run_config.run_efficiency_eval and not skip_eff_for_advanced:
            self._log("\n[Step 2] Running Efficiency Evaluation (Learning Curve)...")
            try:
                eff_dir_params = EffDirParams(
                    indiv_dir=self.dir_params.input_dir,
                    output_dir=learning_curve_dir,
                )
                # Use learning_curve_dir for efficiency output
                eff_combo_config = replace(combo_config, output_directory=learning_curve_dir)
                eff_orchestrator = EfficiencyOrchestrator(
                    dirpara=eff_dir_params,
                    summary_gpr_config=eff_combo_config,
                    summary_gpr_hyperparams=self.summary_gpr_hyperparams,
                    globpara=self.efficiency_globpara,
                    scapara=self.efficiency_scapara,
                    plotpara=self.efficiency_plotpara,
                    verbose=self.verbose,
                    aggregation_method=options.aggregation_method,
                )
                eff_results = eff_orchestrator.process_all()
                results["efficiency"] = eff_results
                self._log(f"  + Processed {len(eff_results)} potentials")
            except Exception as e:
                self._log(f"  X Efficiency eval failed: {e}")
                results["efficiency_error"] = str(e)
        elif skip_eff_for_advanced and self.run_config.run_efficiency_eval:
            self._log(f"\n[Step 2] Skipped Learning Curve for {options.aggregation_method} combo")
            results["efficiency_skipped"] = options.aggregation_method
        
        # Copy artifacts
        if self.run_config.copy_artifacts:
            self._log("\n[Step 3] Copying artifacts...")
            n_copied = copy_artifacts(
                self.dir_params.input_dir,
                output_dir,
                verbose=self.verbose,
            )
            results["artifacts_copied"] = n_copied
        
        # Save settings to output for reproducibility
        if self.settings_manager is not None:
            self._log("\n[Step 4] Saving settings for reproducibility...")
            try:
                # Update settings with current combo options
                combo_settings_update = {
                    "aggregation": {
                        "weight_mode": options.weight_mode,
                        "weight_scope": options.weight_scope,
                        "normalization_summary": options.normalization_summary,
                        "aggregation_method": options.aggregation_method,
                    }
                }
                self.settings_manager.update_section("pip3_summary_gpr", combo_settings_update)
                
                # Save to output directory
                saved_path = self.settings_manager.save_to_output(output_dir)
                results["settings_saved"] = str(saved_path)
                self._log(f"  + Settings saved to {saved_path}")
            except Exception as e:
                self._log(f"  X Failed to save settings: {e}")
                results["settings_error"] = str(e)
        
        self._log(f"\n-> Finished {tag}")
        
        return results
    
    def run_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all option combinations in the testing grid.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results keyed by combination tag.
        """
        self._log(f"Starting batch testing with {len(self.testing_grid)} combinations")
        self._log(f"Input directory: {self.dir_params.input_dir}")
        self._log(f"Output base: {self.dir_params.base_output_dir}")
        self._log(f"\nCombinations to test:")
        for opt in self.testing_grid:
            self._log(f"  - {opt.tag}")
        
        all_results: Dict[str, Dict[str, Any]] = {}
        
        for options in self.testing_grid:
            combo_results = self.run_single_combo(options)
            all_results[options.tag] = combo_results
        
        # Run comparison plotting and CSV export after all permutations complete
        if self.run_config.run_comparison_plots or self.run_config.export_summary_csvs:
            self._run_post_processing(all_results)
        
        self._log(f"\n{'='*60}")
        self._log(f"Batch testing complete: {len(all_results)} combinations processed")
        self._log(f"{'='*60}")
        
        return all_results
    
    def _run_post_processing(self, all_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Run post-processing after all permutations complete.
        
        Generates comparison plots and exports aggregate CSVs.
        
        Parameters
        ----------
        all_results : Dict
            Results from all permutation runs.
        """
        base_dir = self.dir_params.base_output_dir
        output_dir = base_dir / "comparisons"
        
        # Get list of completed permutation tags
        permutation_tags = [tag for tag, res in all_results.items() 
                          if "summary_gpr_error" not in res or "efficiency_error" not in res]
        
        if not permutation_tags:
            self._log("\nNo successful permutations to post-process")
            return
        
        # Export aggregate CSVs
        if self.run_config.export_summary_csvs:
            self._log("\n" + "="*60)
            self._log("[Post-Processing] Exporting Aggregate CSVs...")
            self._log("="*60)
            try:
                csv_results = export_all_aggregate_csvs(
                    batch_output_dir=base_dir,
                    output_dir=output_dir,
                    permutation_tags=permutation_tags,
                    verbose=self.verbose,
                )
                all_results["_aggregate_csvs"] = {
                    "summary_gpr": {str(k): str(v) for k, v in csv_results.get("summary_gpr", {}).items()},
                    "efficiency": {str(k): str(v) for k, v in csv_results.get("efficiency", {}).items()},
                }
            except Exception as e:
                self._log(f"  X Failed to export CSVs: {e}")
                all_results["_aggregate_csvs_error"] = str(e)
        
        # Generate comparison plots
        if self.run_config.run_comparison_plots:
            self._log("\n" + "="*60)
            self._log("[Post-Processing] Generating Comparison Plots...")
            self._log("="*60)
            try:
                # Create config with axis labels from summary_gpr_config
                comparison_config = BatchComparisonConfig(
                    x_axis_label=self.summary_gpr_config.x_axis_label,
                    y_axis_label=self.summary_gpr_config.y_axis_label,
                )
                plot_results = plot_all_comparisons(
                    batch_output_dir=base_dir,
                    output_dir=output_dir,
                    permutation_tags=permutation_tags,
                    config=comparison_config,
                    plot_types=["combined", "summary"],
                    verbose=self.verbose,
                )
                all_results["_comparison_plots"] = {
                    k: [str(p) for p in v] for k, v in plot_results.items()
                }
            except Exception as e:
                self._log(f"  X Failed to generate plots: {e}")
                all_results["_comparison_plots_error"] = str(e)
