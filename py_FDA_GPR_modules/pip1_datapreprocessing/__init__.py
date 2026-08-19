# pip1_datapreprocessing/__init__.py
"""
Data Preprocessing Module for the GPR-FDA Pipeline.

This module handles:
- Configuration for preprocessing parameters
- X/Y filtering and transformation
- Y normalization (peak or middle-average)
- Data downsampling in transformed x-space

Primary Entry Point:
--------------------
    DataPreprocessor  - Orchestrator class for all preprocessing operations
    preprocess_curves - Convenience function for quick preprocessing

Example:
--------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
    >>> from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor, preprocess_curves
    >>> 
    >>> # Load data from pip0
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> loaded = loader.load_all()
    >>> 
    >>> # Option 1: Using the orchestrator class
    >>> preprocessor = DataPreprocessor()
    >>> result = preprocessor.preprocess_all(loaded.curves)
    >>> print(result.summary())
    >>> 
    >>> # Option 2: Using the convenience function
    >>> result = preprocess_curves(loaded.curves)
    >>> 
    >>> # Access preprocessed data ready for GPR
    >>> for curve in result.curves:
    ...     print(curve.sample_id, curve.x_train.shape, curve.x_scaling)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Primary entry point - Orchestrator
from .pip1_data_preprocessor_orchestrator import (
    DataPreprocessor,
    PreprocessedCurve,
    PreprocessingResult,
    preprocess_curves,
)

# Configuration
from .preproc_config import PreprocCfg

# Low-level preprocessing functions (for advanced use)
from .preprocessing_functions import (
    apply_x_filter,
    filter_by_y_threshold,
    # Scaling class - handles all normalization/transformation
    ScalingInfo,
)

# Downsampling functions (new module with smart downsampling)
from .preprc_downsampling import (
    downsample_data,
    downsample_with_result,
    downsample_uniform_bins,
    downsample_adaptive,
    downsample_gradient_preserving,
    downsample_feature_aware,
    DownsampleResult,
)

__all__ = [
    # Primary entry point
    "DataPreprocessor",
    "PreprocessedCurve",
    "PreprocessingResult",
    "preprocess_curves",
    # Configuration
    "PreprocCfg",
    # Low-level functions
    "apply_x_filter",
    "filter_by_y_threshold",
    # Scaling class
    "ScalingInfo",
    # Downsampling functions
    "downsample_data",
    "downsample_with_result",
    "downsample_uniform_bins",
    "downsample_adaptive",
    "downsample_gradient_preserving",
    "downsample_feature_aware",
    "DownsampleResult",
]
