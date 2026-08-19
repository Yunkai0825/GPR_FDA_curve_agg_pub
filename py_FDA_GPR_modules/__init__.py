# __init__.py
"""
GPR-FDA: Gaussian Process Regression for Functional Data Analysis

A modular pipeline for processing electrochemical transient data using
Gaussian Process Regression (GPR) and aggregating individual models into
summary GPR models.

Modules:
--------
- pip0_dataloading: Raw data I/O and .cor file parsing
- pip1_datapreprocessing: Data filtering, normalization, and downsampling
- pip2_individual_gpr: Individual GPR model fitting
- pip3_FDA_scoring_and_aggregations: Summary GPR scoring and aggregation algorithms
- pip4_efficiency_eval: Data efficiency evaluation via Monte-Carlo learning curves
- pip5_batch_setting_grid_testing: Batch driver for testing algorithm options

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

__version__ = "1.0.0"

# Import from new modular structure
from .pip0_dataloading import (
    # Primary entry point - Orchestrator
    DataLoader,
    LoadedCurve,
    DataLoadingResult,
    load_data,
    # Configuration
    RawIOCfg,
    # Low-level functions (backward compatibility)
    compute_file_hash,
    find_cor_files,
    parse_cor_file,
    extract_curve_data,
    find_data_start,
    read_data_points,
)

from .pip1_datapreprocessing import (
    # Primary entry point - Orchestrator
    DataPreprocessor,
    PreprocessedCurve,
    PreprocessingResult,
    preprocess_curves,
    # Configuration
    PreprocCfg,
    # Scaling class - handles all normalization/transformation
    ScalingInfo,
    # Low-level functions (generic names)
    downsample_data,
    apply_x_filter,
    filter_by_y_threshold,
)

from .pip2_individual_gpr import (
    # Primary entry point - Orchestrator
    IndividualGPRProcessor,
    GPRFitResult,
    GPRProcessingResult,
    fit_individual_gprs,
    # Configuration
    GPRCfg,
    ExportCfg,
    # Core GPR functions
    PosteriorResult,
    perform_gpr,
    generate_predictions,
    validate_gpr,
    compute_full_posterior_covariance,
    # Utility functions
    group_curves_by_primary_key,
    group_curves_by_key,
    save_skipped_samples_summary,
    plot_individual_gpr,
    export_gpr_result_to_csv,
    export_validation_results,
)

# Import from pip3 for summary GPR functionality
from .pip3_FDA_scoring_and_aggregations import (
    # From pip3_loader
    IndividualGPRData,
    load_all_individual_gprs,
    group_gprs_by_key,
    # Core aggregation
    compute_summary_gpr,
    SummaryGPRResult,
    SummaryGPROrchestrator,
)

# Import from pip4 for efficiency evaluation
from .pip4_efficiency_eval import (
    DirParams,
    GlobalParams,
    ScaleParams,
    PlotParams,
    EfficiencyOrchestrator,
)

from .pip0_dataloading import (
    SettingsManager,
    load_settings,
    get_default_settings_path,
)

__all__ = [
    "__version__",
    # pip0_dataloading - Data Loading Orchestrator (Primary Entry Point)
    "DataLoader",
    "LoadedCurve",
    "DataLoadingResult",
    "load_data",
    # pip0_dataloading - Configuration
    "RawIOCfg",
    # pip0_dataloading - Low-level functions (backward compatibility)
    "compute_file_hash",
    "find_cor_files",
    "parse_cor_file",
    "extract_curve_data",
    "find_data_start",
    "read_data_points",
    # pip1_datapreprocessing - Preprocessing Orchestrator (Primary Entry Point)
    "DataPreprocessor",
    "PreprocessedCurve",
    "PreprocessingResult",
    "preprocess_curves",
    # pip1_datapreprocessing - Configuration
    "PreprocCfg",
    # pip1_datapreprocessing - Scaling class
    "ScalingInfo",
    # pip1_datapreprocessing - Low-level functions (generic names)
    "downsample_data",
    "apply_x_filter",
    "filter_by_y_threshold",
    # pip2_individual_gpr - GPR Orchestrator (Primary Entry Point)
    "IndividualGPRProcessor",
    "GPRFitResult",
    "GPRProcessingResult",
    "fit_individual_gprs",
    # pip2_individual_gpr - Configuration
    "GPRCfg",
    "ExportCfg",
    # pip2_individual_gpr - Core GPR functions
    "PosteriorResult",
    "perform_gpr",
    "generate_predictions",
    "validate_gpr",
    "compute_full_posterior_covariance",
    # pip2_individual_gpr - Utility functions
    "group_curves_by_primary_key",
    "group_curves_by_key",
    "save_skipped_samples_summary",
    "plot_individual_gpr",
    "export_gpr_result_to_csv",
    "export_validation_results",
    # pip3 - Summary GPR aggregation
    "IndividualGPRData",
    "load_all_individual_gprs",
    "group_gprs_by_key",
    "compute_summary_gpr",
    "SummaryGPRResult",
    "SummaryGPROrchestrator",
    # pip4 - Efficiency evaluation
    "DirParams",
    "GlobalParams",
    "ScaleParams",
    "PlotParams",
    "EfficiencyOrchestrator",
    # settings_manager - Centralized configuration
    "SettingsManager",
    "load_settings",
    "get_default_settings_path",
]
