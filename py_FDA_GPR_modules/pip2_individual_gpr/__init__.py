# pip2_individual_gpr/__init__.py
"""
Individual GPR Module for the GPR-FDA Pipeline.

This module handles:
- GPR model fitting to preprocessed curves
- Validation metrics computation
- Uncertainty estimation (including local uncertainty)
- Export to CSV and plotting

Primary Entry Point:
--------------------
    IndividualGPRProcessor - Orchestrator class for GPR fitting
    fit_individual_gprs    - Convenience function for quick fitting

Example:
--------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
    >>> from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
    >>> from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor, fit_individual_gprs
    >>> 
    >>> # Load and preprocess data
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> preprocessor = DataPreprocessor()
    >>> preprocessed = preprocessor.preprocess_all(loader.load_all().curves)
    >>> 
    >>> # Option 1: Using the orchestrator class
    >>> gpr_processor = IndividualGPRProcessor(output_directory="/path/to/output")
    >>> result = gpr_processor.fit_all(preprocessed.curves)
    >>> print(result.summary())
    >>> 
    >>> # Option 2: Using the convenience function
    >>> result = fit_individual_gprs(preprocessed.curves, output_directory="/path/to/output")
    >>> 
    >>> # Access fitted GPR models
    >>> for gpr_result in result.results:
    ...     print(gpr_result.sample_id, gpr_result.validation_rmse)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Primary entry point - Orchestrator
from .pip2_individual_gpr_processor_orchestrator import (
    IndividualGPRProcessor,
    GPRFitResult,
    GPRProcessingResult,
    fit_individual_gprs,
)

# Configuration
from .gpr_config import GPRCfg, ExportCfg, SharedGridConfig

# Core GPR functions (theoretical framework based on Dirac notation derivation)
from .gpr_functions import (
    PosteriorResult,
    perform_gpr,
    generate_predictions,
    validate_gpr,
    compute_full_posterior_covariance,
    refit_with_frozen_hyperparameters,
    regulate_to_shared_grid,
    full_regulation_workflow,
)

# Utility functions
from .gpr_utilities import (
    group_curves_by_primary_key,
    group_curves_by_key,
    save_skipped_samples_summary,
    plot_individual_gpr,
    export_gpr_result_to_csv,
    export_covariance_matrix,
    export_diagnostic_data,
    export_validation_results,
    load_gpr_csv_with_metadata,
)

# Visualization utilities
from .gpr_utilities import (
    plot_posterior_covariance_diagnostics,
    save_posterior_covariance,
)

# Reader functions - reconstruct GPR results from exported CSVs
from .gpr_reader import (
    ReconstructedPosterior,
    ReconstructedGPRResult,
    load_gpr_result_csv,
    load_covariance_csv,
    reconstruct_gpr_result,
    reconstruct_all_gpr_results,
    plot_reconstructed_gpr,
    verify_reconstruction,
)

__all__ = [
    # Primary entry point
    "IndividualGPRProcessor",
    "GPRFitResult",
    "GPRProcessingResult",
    "fit_individual_gprs",
    # Configuration
    "GPRCfg",
    "ExportCfg",
    "SharedGridConfig",
    # Core GPR functions
    "PosteriorResult",
    "perform_gpr",
    "generate_predictions",
    "validate_gpr",
    "compute_full_posterior_covariance",
    "refit_with_frozen_hyperparameters",
    "regulate_to_shared_grid",
    "full_regulation_workflow",
    # Utility functions
    "group_curves_by_primary_key",
    "group_curves_by_key",
    "save_skipped_samples_summary",
    "plot_individual_gpr",
    "export_gpr_result_to_csv",
    "export_covariance_matrix",
    "export_diagnostic_data",
    "export_validation_results",
    "load_gpr_csv_with_metadata",
    # Visualization utilities
    "plot_posterior_covariance_diagnostics",
    "save_posterior_covariance",
    # Reader functions
    "ReconstructedPosterior",
    "ReconstructedGPRResult",
    "load_gpr_result_csv",
    "load_covariance_csv",
    "reconstruct_gpr_result",
    "reconstruct_all_gpr_results",
    "plot_reconstructed_gpr",
    "verify_reconstruction",
]
