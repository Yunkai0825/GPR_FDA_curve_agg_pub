# pip0_dataloading/__init__.py
"""
Data Loading Module for the GPR-FDA Pipeline.

This module handles:
- Configuration for raw data I/O paths
- Finding and de-duplicating .cor files
- Parsing CorrWare .cor files to extract potentiostatic transient data

Primary Entry Point:
--------------------
    DataLoader - Orchestrator class for all data loading operations
    load_data  - Convenience function for quick loading

Example:
--------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader, load_data
    >>> 
    >>> # Option 1: Using the orchestrator class
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> result = loader.load_all()
    >>> print(result.summary())
    >>> 
    >>> # Option 2: Using the convenience function
    >>> result = load_data("/path/to/data")
    >>> for curve in result.curves:
    ...     print(curve.sample_id, curve.potential)

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

# Primary entry point - Orchestrator
from .pip0_data_loader_orchestrator import (
    DataLoader,
    LoadedCurve,
    DataLoadingResult,
    load_data,
)

# Configuration
from .raw_io_config import RawIOCfg

# Low-level parsing functions (for backward compatibility)
from .cor_file_parser import (
    compute_file_hash,
    find_cor_files,
    parse_cor_file,
    extract_curve_data,
    find_data_start,
    read_data_points,
)

# Settings Manager
from .settings_manager import (
    SettingsManager,
    load_settings,
    get_default_settings_path,
)

# Filename Parser (for extracting grouping keys from filenames)
from .filename_parser import (
    parse_filename,
    extract_grouping_value,
    batch_parse_filenames,
    get_unique_grouping_values,
    group_files_by_key,
    format_group_key_title,
)

__all__ = [
    # Primary entry point
    "DataLoader",
    "LoadedCurve",
    "DataLoadingResult",
    "load_data",
    # Configuration
    "RawIOCfg",
    # Settings Manager
    "SettingsManager",
    "load_settings",
    "get_default_settings_path",
    # Filename Parser
    "parse_filename",
    "extract_grouping_value",
    "batch_parse_filenames",
    "get_unique_grouping_values",
    "group_files_by_key",
    "format_group_key_title",
    # Low-level functions (backward compatibility)
    "compute_file_hash",
    "find_cor_files",
    "parse_cor_file",
    "extract_curve_data",
    "find_data_start",
    "read_data_points",
]
