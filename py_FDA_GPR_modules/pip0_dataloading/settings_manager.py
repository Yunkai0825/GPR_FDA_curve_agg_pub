# py_FDA_GPR_modules/pip0_dataloading/settings_manager.py
"""
=============================================================================
Centralized Settings Manager for the GPR-FDA Pipeline
=============================================================================

Provides utilities for:
- Loading settings from JSON files (auto-detect from input data folder)
- Converting JSON settings to pipeline config dataclasses
- Managing output directory structure based on input folder name
- Saving settings to output directories for reproducibility
- Merging settings with overrides

Usage:
------
    # Load from specific input data folder (recommended)
    manager = SettingsManager.from_input_folder("_input/CCNF_CTAB_PT")
    
    # Output folder is auto-generated as "{input_folder_name}_output"
    # e.g., "_output/CCNF_CTAB_PT_output"
    
    # Get output directories for each pipeline step
    pip2_output = manager.get_pip2_output_dir()  # "_output/CCNF_CTAB_PT_output/individual_gpr"
    pip3_output = manager.get_pip3_output_dir()  # "_output/CCNF_CTAB_PT_output/summary_gpr"

Author: Yunkai Sun (C-STEEL, CSE, ANL)
=============================================================================
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, fields, is_dataclass
from typing import Dict, Any, Optional, Type, TypeVar, Union
from copy import deepcopy
import glob as glob_module

# Type variable for generic dataclass conversion
T = TypeVar("T")


class SettingsManager:
    """
    Manages loading, saving, and converting pipeline settings.
    
    Example
    -------
    >>> # Option 1: Load from input data folder (recommended)
    >>> manager = SettingsManager.from_input_folder("_input/CCNF_CTAB_PT")
    >>> 
    >>> # Output folder is auto-generated: "_output/CCNF_CTAB_PT_output"
    >>> pip2_output = manager.get_pip2_output_dir()
    >>> pip3_output = manager.get_pip3_output_dir()
    >>> 
    >>> # Option 2: Load from specific settings file
    >>> manager = SettingsManager("_input/settings_template.json")
    >>> 
    >>> # Get configs for each pipeline
    >>> preproc_cfg = manager.get_preproc_config()
    >>> gpr_cfg, export_cfg = manager.get_gpr_configs()
    >>> summary_cfg, summary_hp = manager.get_summary_gpr_configs()
    >>> 
    >>> # Save settings used to output directory
    >>> manager.save_to_output(manager.get_base_output_dir())
    """
    
    def __init__(
        self,
        settings_path: Optional[Union[str, Path]] = None,
        settings_dict: Optional[Dict[str, Any]] = None,
        input_folder: Optional[Union[str, Path]] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize settings manager.
        
        Parameters
        ----------
        settings_path : str or Path, optional
            Path to settings JSON file.
        settings_dict : dict, optional
            Settings dictionary (alternative to loading from file).
        input_folder : str or Path, optional
            Input data folder (e.g., "_input/CCNF_CTAB_PT").
            Used to auto-generate output folder name.
        workspace_root : str or Path, optional
            Root directory of the workspace. Defaults to parent of _input folder.
        """
        if settings_path is not None:
            self.settings_path = Path(settings_path)
            self.settings = self._load_json(self.settings_path)
        elif settings_dict is not None:
            self.settings_path = None
            self.settings = deepcopy(settings_dict)
        else:
            raise ValueError("Must provide either settings_path or settings_dict")
        
        # Store input folder info for output path generation
        self.input_folder = Path(input_folder) if input_folder else None
        
        # Determine workspace root
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        elif self.input_folder:
            # Go up until we find _input parent
            self.workspace_root = self._find_workspace_root(self.input_folder)
        elif self.settings_path:
            self.workspace_root = self._find_workspace_root(self.settings_path)
        else:
            self.workspace_root = Path.cwd()
        
        # Update metadata
        self._update_meta()
    
    @classmethod
    def from_input_folder(
        cls,
        input_folder: Union[str, Path],
        settings_filename: Optional[str] = None,
        workspace_root: Optional[Union[str, Path]] = None,
    ) -> "SettingsManager":
        """
        Create SettingsManager from an input data folder.
        
        Automatically finds settings JSON file in the folder (pattern: settings*.json).
        Output folder is auto-generated as "{folder_name}_output".
        
        Parameters
        ----------
        input_folder : str or Path
            Path to input data folder (e.g., "_input/CCNF_CTAB_PT").
        settings_filename : str, optional
            Specific settings file name. If None, auto-detects settings*.json.
        workspace_root : str or Path, optional
            Root directory of the workspace.
            
        Returns
        -------
        SettingsManager
        
        Example
        -------
        >>> manager = SettingsManager.from_input_folder("_input/CCNF_CTAB_PT")
        >>> print(manager.get_base_output_dir())
        # _output/CCNF_CTAB_PT_output
        """
        input_folder = Path(input_folder)
        
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Find settings file
        if settings_filename:
            settings_path = input_folder / settings_filename
        else:
            # Auto-detect settings*.json
            settings_files = list(input_folder.glob("settings*.json"))
            if not settings_files:
                raise FileNotFoundError(
                    f"No settings*.json file found in {input_folder}. "
                    "Please create a settings file (e.g., settings_CCNF_CTAB_PT.json)"
                )
            if len(settings_files) > 1:
                # Prefer file matching folder name
                folder_name = input_folder.name
                matching = [f for f in settings_files if folder_name in f.name]
                settings_path = matching[0] if matching else settings_files[0]
            else:
                settings_path = settings_files[0]
        
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        
        return cls(
            settings_path=settings_path,
            input_folder=input_folder,
            workspace_root=workspace_root,
        )
    
    def _find_workspace_root(self, path: Path) -> Path:
        """Find workspace root by looking for _input or _output parent."""
        path = Path(path).resolve()
        for parent in [path] + list(path.parents):
            if parent.name == "_input":
                return parent.parent
            if (parent / "_input").exists():
                return parent
        return path.parent
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load settings from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _update_meta(self):
        """Update metadata with current timestamp."""
        if "_meta" not in self.settings:
            self.settings["_meta"] = {}
        self.settings["_meta"]["loaded_at"] = datetime.now().isoformat()
        if self.input_folder:
            self.settings["_meta"]["input_folder"] = str(self.input_folder)
    
    # =========================================================================
    # Output Directory Management
    # =========================================================================
    
    def get_output_structure(self) -> Dict[str, str]:
        """Get output directory structure from settings."""
        return self.get_raw("output_structure")
    
    def get_base_output_dir(self) -> Path:
        """
        Get base output directory for this dataset.
        
        For production (base_output_dir="_output"):
            Returns "{workspace_root}/_output/{input_folder_name}_output".
        For DEBUG (base_output_dir="py_FDA_GPR_modules" or other):
            Returns "{workspace_root}/{base_output_dir}" without suffix.
        """
        structure = self.get_output_structure()
        base_dir = structure.get("base_output_dir", "_output")
        
        # Only add folder_name suffix for production output (base_dir starts with "_output")
        if self.input_folder and base_dir.startswith("_output"):
            folder_name = self.input_folder.name
            return self.workspace_root / base_dir / f"{folder_name}_output"
        else:
            return self.workspace_root / base_dir
    
    def get_pip2_output_dir(self) -> Path:
        """Get output directory for pip2 (individual GPR)."""
        structure = self.get_output_structure()
        subdir = structure.get("pip2_individual_gpr", "individual_gpr")
        return self.get_base_output_dir() / subdir
    
    def get_pip3_output_dir(self) -> Path:
        """Get output directory for pip3 (scoring & aggregations)."""
        structure = self.get_output_structure()
        subdir = structure.get("pip3_scoring_aggregations", "summary_gpr")
        return self.get_base_output_dir() / subdir
    
    def get_pip4_output_dir(self) -> Path:
        """Get output directory for pip4 (efficiency evaluation)."""
        structure = self.get_output_structure()
        subdir = structure.get("pip4_efficiency_eval", "efficiency_eval")
        return self.get_base_output_dir() / subdir
    
    def get_pip5_output_dir(self) -> Path:
        """Get output directory for pip5 (batch testing)."""
        structure = self.get_output_structure()
        subdir = structure.get("pip5_batch_testing", "batch_testing")
        return self.get_base_output_dir() / subdir
    
    # =========================================================================
    # Settings Access & Update
    # =========================================================================
    
    def get_raw(self, section: str) -> Dict[str, Any]:
        """Get raw settings dictionary for a section."""
        return deepcopy(self.settings.get(section, {}))
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """
        Update a section with new values (deep merge).
        
        Parameters
        ----------
        section : str
            Section name (e.g., "pip3_summary_gpr").
        updates : dict
            Dictionary of updates to apply.
        """
        if section not in self.settings:
            self.settings[section] = {}
        self._deep_merge(self.settings[section], updates)
    
    def _deep_merge(self, base: Dict, updates: Dict):
        """Recursively merge updates into base dict."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)
    
    # =========================================================================
    # pip0: Data Loading / Input Directory
    # =========================================================================
    
    def get_input_directory(self) -> Path:
        """
        Get the input data directory.
        
        If input_folder was set during initialization, returns that.
        Otherwise, constructs path from pip0 settings.
        
        Returns
        -------
        Path
            Absolute path to input data directory.
        """
        if self.input_folder:
            return self.input_folder.resolve()
        
        # Fallback to pip0 settings
        pip0 = self.get_raw("pip0_dataloading")
        input_subdir = pip0.get("input_subdir", ".")
        
        if self.settings_path:
            # Relative to settings file location
            base_path = self.settings_path.parent
        else:
            base_path = self.workspace_root / "_input"
        
        return (base_path / input_subdir).resolve()
    
    def get_data_subdirectories(self) -> list:
        """
        Get list of data subdirectories in the input folder.
        
        Returns
        -------
        list
            List of subdirectory paths containing data.
        """
        input_dir = self.get_input_directory()
        return [d for d in input_dir.iterdir() if d.is_dir()]
    
    def get_raw_io_config(self, path_to_folder: Optional[Path] = None):
        """
        Get RawIOCfg from settings.
        
        Parameters
        ----------
        path_to_folder : Path, optional
            Path to the data folder. If None, uses get_input_directory().
            
        Returns
        -------
        RawIOCfg
        """
        from . import RawIOCfg
        
        if path_to_folder is None:
            path_to_folder = self.get_input_directory()
        
        pip0 = self.get_raw("pip0_dataloading")
        return RawIOCfg(
            path_to_your_folder=path_to_folder,
            output_subdir=pip0.get("output_subdir", "output_directory"),
        )
    
    def get_filename_parsing_config(self) -> Dict[str, Any]:
        """
        Get filename parsing configuration for extracting grouping keys.
        
        Returns
        -------
        Dict[str, Any]
            Filename parsing configuration with keys:
            - file_extension: str (e.g., ".cor")
            - delimiter: str (e.g., "_")
            - grouping_keys: List[dict] with {name, token_index, dtype, regex_extract}
            - metadata_keys: List[dict] with {name, token_index, dtype, regex_extract}
            - primary_grouping_key: str (first grouping key by default)
            - fallback_to_file_content: bool
        """
        pip0 = self.get_raw("pip0_dataloading")
        parsing = pip0.get("filename_parsing", {})
        
        # Get grouping keys
        grouping_keys = parsing.get("grouping_keys", [])
        
        # Determine primary grouping key: explicit setting > first grouping key > empty
        primary_key = pip0.get("primary_grouping_key")
        if not primary_key and grouping_keys:
            primary_key = grouping_keys[0].get("name", "")
        primary_key = primary_key or ""
        
        return {
            "file_extension": pip0.get("file_extension", ".cor"),
            "delimiter": parsing.get("delimiter", "_"),
            "grouping_keys": grouping_keys,
            "metadata_keys": parsing.get("metadata_keys", []),
            "primary_grouping_key": primary_key,
            "fallback_to_file_content": pip0.get("fallback_to_file_content", True),
        }

    # =========================================================================
    # pip1: Data Preprocessing
    # =========================================================================
    
    def get_preproc_config(self):
        """
        Get PreprocCfg from settings.
        
        Returns
        -------
        PreprocCfg
        """
        from ..pip1_datapreprocessing import PreprocCfg
        
        pip1 = self.get_raw("pip1_datapreprocessing")
        
        # --- Strict: require all filtering/grid keys from JSON ---
        filtering = pip1.get("filtering")
        if filtering is None:
            raise KeyError("Missing 'pip1_datapreprocessing.filtering' section in JSON settings")
        for key in ("max_points_set", "min_x_cap", "max_x_cap", "min_curve_range", "y_threshold"):
            if key not in filtering:
                raise KeyError(f"Missing required key 'pip1_datapreprocessing.filtering.{key}' in JSON settings")

        x_scaling = pip1.get("x_scaling")
        if x_scaling is None:
            raise KeyError("Missing 'pip1_datapreprocessing.x_scaling' section in JSON settings")
        y_scaling = pip1.get("y_scaling")
        if y_scaling is None:
            raise KeyError("Missing 'pip1_datapreprocessing.y_scaling' section in JSON settings")

        return PreprocCfg(
            # Column names
            x_col=pip1.get("column_names", {}).get("x_col", "x"),
            y_col=pip1.get("column_names", {}).get("y_col", "y"),
            # Master filtering toggle
            enable_filtering=filtering.get("enable_filtering", True),
            # Filtering — strictly from JSON, no defaults
            max_points_set=filtering["max_points_set"],
            min_x_cap=filtering["min_x_cap"],
            max_x_cap=filtering["max_x_cap"],
            min_curve_range=filtering["min_curve_range"],
            y_threshold=filtering["y_threshold"],
            # X scaling
            x_scaling_method=x_scaling.get("method", "log"),
            x_scaling_params=x_scaling.get("params", {"base": "log10", "shift": 1e-9}),
            # Y scaling
            y_scaling_method=y_scaling.get("method", "peak"),
            y_scaling_params=y_scaling.get("params", {}),
            # Grouping
            group_round_digits=pip1.get("grouping", {}).get("group_round_digits", 2),
        )
    
    # =========================================================================
    # pip2: Individual GPR
    # =========================================================================
    
    def get_gpr_configs(self):
        """
        Get GPRCfg and ExportCfg from settings.
        
        Returns
        -------
        Tuple[GPRCfg, ExportCfg]
        """
        from ..pip2_individual_gpr import GPRCfg, ExportCfg, SharedGridConfig
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
        
        pip2 = self.get_raw("pip2_individual_gpr")
        gpr_params = pip2.get("gpr_params", {})
        kernel_cfg = pip2.get("kernel", {})
        export_params = pip2.get("export", {})
        shared_grid_cfg = pip2.get("shared_grid", {})
        posterior_cov_cfg = pip2.get("posterior_covariance", {})
        
        # Build kernel from config
        kernel = (
            ConstantKernel(
                1.0,
                tuple(kernel_cfg.get("constant_bounds", [1e-2, 1e4]))
            )
            * Matern(
                length_scale=kernel_cfg.get("matern_length_scale", 1.0),
                length_scale_bounds=tuple(kernel_cfg.get("matern_length_scale_bounds", [1e-2, 1e3])),
                nu=kernel_cfg.get("matern_nu", 1.5),
            )
            + WhiteKernel(
                noise_level=kernel_cfg.get("white_noise_level", 1.0),
                noise_level_bounds=tuple(kernel_cfg.get("white_noise_bounds", [1e-12, 1e1])),
            )
        )
        
        # Build shared grid config — strictly from JSON, no defaults
        for key in ("enabled", "method", "explicit_grid", "auto_num_points",
                    "auto_padding_fraction", "refit_on_full_data"):
            if key not in shared_grid_cfg:
                raise KeyError(f"Missing required key 'pip2_individual_gpr.shared_grid.{key}' in JSON settings")
        shared_grid = SharedGridConfig(
            enabled=shared_grid_cfg["enabled"],
            method=shared_grid_cfg["method"],
            explicit_grid=shared_grid_cfg["explicit_grid"],
            auto_num_points=shared_grid_cfg["auto_num_points"],
            auto_padding_fraction=shared_grid_cfg["auto_padding_fraction"],
            refit_on_full_data=shared_grid_cfg["refit_on_full_data"],
        )

        # GPR parameters — strictly from JSON for grid-related keys
        for key in ("num_curves_threshold", "num_X_pred_points_individual_default",
                    "num_X_pred_points_individual_high"):
            if key not in gpr_params:
                raise KeyError(f"Missing required key 'pip2_individual_gpr.gpr_params.{key}' in JSON settings")
        gpr_cfg = GPRCfg(
            kernel=kernel,
            n_restarts_optimizer=gpr_params.get("n_restarts_optimizer", 5),
            alpha=gpr_params.get("alpha", 0.1),
            normalize_y=gpr_params.get("normalize_y", True),
            num_curves_threshold=gpr_params["num_curves_threshold"],
            num_X_pred_points_individual_default=gpr_params["num_X_pred_points_individual_default"],
            num_X_pred_points_individual_high=gpr_params["num_X_pred_points_individual_high"],
            local_var_val_flag=gpr_params.get("local_var_val_flag", False),
            # Shared grid config
            shared_grid=shared_grid,
            # Posterior covariance options
            store_posterior_covariance=posterior_cov_cfg.get("store_full_covariance", True),
            covariance_storage_mode=posterior_cov_cfg.get("storage_mode", "full"),
            covariance_sparse_threshold=posterior_cov_cfg.get("sparse_threshold", 1e-6),
        )
        
        # Get column names and transform info from pip1 for axis labels
        pip1 = self.get_raw("pip1_datapreprocessing")
        col_names = pip1.get("column_names", {})
        x_scaling = pip1.get("x_scaling", {})
        y_scaling = pip1.get("y_scaling", {})
        
        export_cfg = ExportCfg(
            plot_individual_gpr=export_params.get("plot_individual_gpr", True),
            individual_curve_alpha=export_params.get("individual_curve_alpha", 0.20),
            plot_downsample_points=export_params.get("plot_downsample_points", 500),
            dpi=export_params.get("dpi", 300),
            max_points_to_save=export_params.get("max_points_to_save", 10000),
            x_col_name=col_names.get("x_col", "X"),
            y_col_name=col_names.get("y_col", "Y"),
            x_transform_method=x_scaling.get("method", ""),
            y_transform_method=y_scaling.get("method", ""),
        )
        
        return gpr_cfg, export_cfg
    
    # =========================================================================
    # pip3: Summary GPR
    # =========================================================================
    
    def get_summary_gpr_configs(
        self,
        input_directory: Optional[Path] = None,
        output_directory: Optional[Path] = None,
    ):
        """
        Get SummaryGPRConfig and SummaryGPRHyperParams from settings.
        
        Parameters
        ----------
        input_directory : Path, optional
            Directory containing individual GPR CSVs.
            Defaults to pip2 output directory.
        output_directory : Path, optional
            Output directory. Defaults to pip3 output directory.
            
        Returns
        -------
        Tuple[SummaryGPRConfig, SummaryGPRHyperParams]
        """
        from ..pip3_FDA_scoring_and_aggregations import SummaryGPRConfig, SummaryGPRHyperParams
        
        pip3 = self.get_raw("pip3_summary_gpr")
        agg = pip3.get("aggregation", {})
        plot = pip3.get("plotting", {})
        hp = pip3.get("hyperparams", {})
        
        # Use default output directories if not specified
        if input_directory is None:
            input_directory = self.get_pip2_output_dir()
        if output_directory is None:
            output_directory = self.get_pip3_output_dir()
        
        # Get column names from pip1 for axis labels
        # Summary GPR shows REAL scale data (inverse transformed), so no transform info needed
        pip1 = self.get_raw("pip1_datapreprocessing")
        col_names = pip1.get("column_names", {})
        x_col_name = col_names.get("x_col", "")
        y_col_name = col_names.get("y_col", "")
        
        # Get transform method for y-axis (used for normalized iteration plots)
        y_scaling = pip1.get("y_scaling", {})
        y_transform = y_scaling.get("method", "")
        
        # Axis labels: just column names for real-scale Summary GPR plots
        x_axis_label = x_col_name
        y_axis_label = y_col_name
        
        config = SummaryGPRConfig(
            input_directory=input_directory,
            output_directory=output_directory,
            file_pattern=pip3.get("file_pattern", "Individual_GPR_*.csv"),
            # Aggregation
            weight_mode=agg.get("weight_mode", "iterative"),
            weight_scope=agg.get("weight_scope", "curve"),
            include_within_variance=agg.get("include_within_variance", True),
            include_between_variance=agg.get("include_between_variance", True),
            variance_aggregation_scale=agg.get("variance_aggregation_scale", "real"),
            normalization_summary=agg.get("normalization_summary", True),
            # Operator fusion / FGPR
            enable_operator_fusion=agg.get("enable_operator_fusion", False),
            enable_fgpr=agg.get("enable_fgpr", agg.get("enable_operator_fusion", False)),
            enable_student_t=agg.get("enable_student_t", False),
            # Plotting
            plot_individual_gprs=plot.get("plot_individual_gprs", True),
            individual_curve_alpha=plot.get("individual_curve_alpha", 0.20),
            # Axis labels (with transform info from settings)
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            y_transform_method=y_transform,  # for normalized iteration plots
            min_time_cap=plot.get("min_time_cap", 1e-4),
            max_time_cap=plot.get("max_time_cap", None),
        )
        
        hyperparams = SummaryGPRHyperParams(
            max_iterations=hp.get("max_iterations"),
            convergence_tol=hp.get("convergence_tol", 1e-6),
            epsilon=hp.get("epsilon", 1e-12),
            confidence_level=hp.get("confidence_level", 0.75),
            num_interp_points=hp.get("num_interp_points", 500),
            fgpr_min_scale_factor_ratio=hp.get("fgpr_min_scale_factor_ratio", 0.01),
            fgpr_structured_btw=hp.get("fgpr_structured_btw", False),
            student_t_nu=hp.get("student_t_nu", 5.0),
            student_t_optimize_nu=hp.get("student_t_optimize_nu", True),
            student_t_nu_bounds=tuple(hp.get("student_t_nu_bounds", [1.0, 500.0])),
            student_t_nu_lb_adaptive=hp.get("student_t_nu_lb_adaptive", False),
            student_t_max_iterations=hp.get("student_t_max_iterations", 100),
            student_t_convergence_tol=hp.get("student_t_convergence_tol", 1e-6),
        )
        
        return config, hyperparams
    
    # =========================================================================
    # pip4: Efficiency Evaluation
    # =========================================================================
    
    def get_efficiency_configs(
        self,
        indiv_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Get efficiency evaluation configs from settings.
        
        Parameters
        ----------
        indiv_dir : Path, optional
            Directory containing individual GPR CSVs.
            Defaults to pip2 output directory.
        output_dir : Path, optional
            Output directory. Defaults to pip4 output directory.
            
        Returns
        -------
        Tuple[DirParams, GlobalParams, ScaleParams, PlotParams]
        """
        from ..pip4_efficiency_eval import DirParams, GlobalParams, ScaleParams, PlotParams
        
        pip4 = self.get_raw("pip4_efficiency_eval")
        glob = pip4.get("global", {})
        scale = pip4.get("scale", {})
        plot = pip4.get("plot", {})
        
        # Use default output directories if not specified
        if indiv_dir is None:
            indiv_dir = self.get_pip2_output_dir()
        if output_dir is None:
            output_dir = self.get_pip4_output_dir()
        
        dir_params = DirParams(
            indiv_dir=indiv_dir,
            output_dir=output_dir,
        )
        
        global_params = GlobalParams(
            metric=glob.get("metric", "rmse"),
            base_repeats=glob.get("base_repeats", 1000),
            max_enum=glob.get("max_enum", 1000),
            q_low=glob.get("q_low", 0.25),
            q_high=glob.get("q_high", 0.75),
            random_seed=glob.get("random_seed", 42),
        )
        
        scale_params = ScaleParams(
            use_log_error=scale.get("use_log_error", True),
            log_base_error=scale.get("log_base_error", "10"),
            eps_error=scale.get("eps_error", 1e-30),
            use_log_cost=scale.get("use_log_cost", True),
            log_base_cost=scale.get("log_base_cost", "10"),
            eps_cost=scale.get("eps_cost", 1e-30),
            normalize_w_rbar=scale.get("normalize_w_rbar", False),
        )
        
        figsize = plot.get("figsize", [10, 4])
        plot_params = PlotParams(
            figsize=tuple(figsize) if isinstance(figsize, list) else figsize,
            dpi=plot.get("dpi", 120),
            xlabel=plot.get("xlabel", "Number of curves"),
            ylabel=plot.get("ylabel", "Error metric"),
            time_label=plot.get("time_label", "CPU time (s)"),
        )
        
        return dir_params, global_params, scale_params, plot_params
    
    # =========================================================================
    # Saving / Export
    # =========================================================================
    
    def save_to_output(
        self,
        output_dir: Path,
        filename: str = "settings_used.json",
    ) -> Path:
        """
        Save current settings to output directory for reproducibility.
        
        Parameters
        ----------
        output_dir : Path
            Output directory.
        filename : str
            Output filename.
            
        Returns
        -------
        Path
            Path to saved settings file.
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Update metadata
        self.settings["_meta"]["saved_at"] = datetime.now().isoformat()
        if self.settings_path:
            self.settings["_meta"]["source_file"] = str(self.settings_path)
        
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=4)
        
        return output_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Return a copy of the settings dictionary."""
        return deepcopy(self.settings)
    
    def clone(self) -> "SettingsManager":
        """Create a copy of this settings manager."""
        new_manager = SettingsManager(settings_dict=self.settings)
        new_manager.input_folder = self.input_folder
        new_manager.workspace_root = self.workspace_root
        new_manager.settings_path = self.settings_path
        return new_manager


def load_settings(path: Union[str, Path]) -> SettingsManager:
    """
    Convenience function to load settings from a JSON file.
    
    Parameters
    ----------
    path : str or Path
        Path to settings JSON file.
        
    Returns
    -------
    SettingsManager
    """
    return SettingsManager(settings_path=path)


def get_default_settings_path() -> Path:
    """Get path to default settings template in workspace _input folder."""
    # Go up from pip0_dataloading to py_FDA_GPR_modules to workspace root
    workspace_root = Path(__file__).parent.parent.parent
    return workspace_root / "_input" / "settings_template.json"
