# pip0_dataloading - Data Loading Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module handles raw data loading from CorrWare `.cor` files, including:
- Recursive file discovery with duplicate detection (MD5 hash)
- Parsing potentiostatic transient experiment data
- Extracting time-series (E, I, T) data and metadata (potential, sample_id)
- Centralized settings management via JSON configuration

---

## Module Structure

```
pip0_dataloading/
├── __init__.py                      # Public API exports
├── pip0_data_loader_orchestrator.py # Main orchestrator class (DataLoader, LoadedCurve, DataLoadingResult)
├── cor_file_parser.py               # Low-level .cor file parsing (parse_cor_file, find_cor_files, compute_file_hash)
├── filename_parser.py               # Filename parsing for grouping key extraction
├── raw_io_config.py                 # Configuration dataclass (RawIOCfg)
├── settings_manager.py              # Centralized JSON settings management (SettingsManager)
├── DEBUG_pip0_input/                # Debug input folder
└── README.md                        # This file
```

---

## Quick Start

```python
from py_FDA_GPR_modules.pip0_dataloading import DataLoader, load_data

# Option 1: Using the orchestrator class
loader = DataLoader(path_to_folder="/path/to/data")
result = loader.load_all()
print(result.summary())

# Option 2: Using the convenience function
result = load_data("/path/to/data")

# Access loaded curves
for curve in result.curves:
    print(curve.sample_id, curve.group_flags, curve.num_points)

# Access curves grouped by primary key (e.g., potential)
primary_key = "potential"  # Configured in settings.json
for key_value, curves in result.curves_by_primary_key.items():
    print(f"Group '{key_value}': {len(curves)} curves")
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `DataLoader` | `pip0_data_loader_orchestrator.py` | Main orchestrator for data loading operations |
| `LoadedCurve` | `pip0_data_loader_orchestrator.py` | Container for a single loaded curve with metadata |
| `DataLoadingResult` | `pip0_data_loader_orchestrator.py` | Result container with all curves and groupings |
| `RawIOCfg` | `raw_io_config.py` | Configuration dataclass for I/O paths |
| `SettingsManager` | `settings_manager.py` | Centralized settings management for all pipeline modules |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `load_data(path)` | `pip0_data_loader_orchestrator.py` | Convenience function for quick loading |
| `parse_cor_file(path)` | `cor_file_parser.py` | Parse a single .cor file |
| `find_cor_files(cfg)` | `cor_file_parser.py` | Find all unique .cor files (deduped by hash) |
| `compute_file_hash(path)` | `cor_file_parser.py` | Compute MD5 hash for duplicate detection |
| `parse_filename(filename, config)` | `filename_parser.py` | Parse filename to extract grouping keys and metadata |
| `extract_grouping_value(filename, config)` | `filename_parser.py` | Extract a single grouping value from filename |
| `batch_parse_filenames(file_paths, config)` | `filename_parser.py` | Parse multiple filenames at once |
| `get_unique_grouping_values(file_paths, config)` | `filename_parser.py` | Get unique grouping values across files |
| `group_files_by_key(file_paths, config)` | `filename_parser.py` | Group files by a grouping key value |

---

## File Details

### `pip0_data_loader_orchestrator.py`

Main orchestrator for data loading.

**Classes:**
- `DataLoader`: Coordinates file discovery, parsing, and grouping
  - `load_all()` → `DataLoadingResult`: Load all .cor files
  - `parse_file(path)` → `List[LoadedCurve]`: Parse all curves from one file
  - `load_as_dicts()` → `List[dict]`: Load curves as plain dictionaries
  
- `LoadedCurve`: Container for loaded curve data
  - `sample_id`, `file_path`, `group_flags`, `data_points`
  - Properties: `x_raw`, `y_raw`, `num_points`
  - Methods:
    - `get_primary_value(primary_key)` → Get value for specified primary grouping key
    - `to_dict()` → Convert the curve and metadata to a dictionary
  
- `DataLoadingResult`: Result container
  - `curves`, `curves_by_primary_key`, `curves_by_group`
  - `primary_key_values` → Sorted list of unique primary key values
  - `primary_grouping_key` → Name of the configured primary key
  - `num_files_processed`, `num_duplicates_skipped`, `errors`
  - Methods:
    - `summary()` → Human-readable summary
    - `get_curves_for_primary_key(value)` → Get curves for a specific primary key value

Filtering is applied while loading with
`DataLoader.load_all(filter_primary_values=[...])` or the equivalent
`load_data(..., filter_primary_values=[...])` argument.

### `cor_file_parser.py`

Low-level .cor file parsing functions.

**Functions:**
- `compute_file_hash(file_path)` → `str`: MD5 hash for duplicate detection
- `find_cor_files(raw_cfg)` → `List[Path]`: Recursively find unique .cor files
- `parse_cor_file(file_path)` → `List[dict]`: Parse .cor file, extract all curves
- `extract_curve_data(lines, start_index, file_path)` → `Tuple[dict, int]`: Extract single curve
- `find_data_start(lines, index, file_path)` → `int`: Find "End Comments" marker
- `read_data_points(lines, start_index)` → `Tuple[List[dict], int]`: Parse E, I, T columns

### `raw_io_config.py`

Configuration dataclass for I/O paths.

**Classes:**
- `RawIOCfg`: Simple config with `path_to_your_folder` and `output_subdir`

### `filename_parser.py`

Filename parsing for extracting grouping keys and metadata from filenames.

**Functions:**
- `parse_filename(filename, parsing_config)` → `Dict[str, Any]`: Parse a filename based on token configuration
- `extract_grouping_value(filename, parsing_config, key_name)` → `Any`: Extract a single grouping value
- `batch_parse_filenames(file_paths, parsing_config)` → `List[Dict]`: Parse multiple filenames
- `get_unique_grouping_values(file_paths, parsing_config, key_name, round_digits)` → `List`: Get sorted unique values
- `group_files_by_key(file_paths, parsing_config, key_name, round_digits)` → `Dict[Any, List[Path]]`: Group files by key

**Example Usage:**
```python
from pathlib import Path

from py_FDA_GPR_modules.pip0_dataloading import SettingsManager, parse_filename, group_files_by_key

# Get parsing config from settings
manager = SettingsManager.from_input_folder("_input/dataset")
parsing_config = manager.get_filename_parsing_config()

# Parse a single filename
# Filename: -1.95V_pH1.48_Na2SO4_H3Cit_NiFeCoCu_acidB(BWR6)_Au-4R2_CTAB_diaphram_20230616.cor
parsed = parse_filename("-1.95V_pH1.48_sample.cor", parsing_config)
# Returns: {"potential": -1.95, "pH": 1.48, "sample_id": "...", "_parsing_success": True, ...}

# Group files by potential
files = list(Path("_input/dataset").glob("*.cor"))
groups = group_files_by_key(files, parsing_config, key_name="potential", round_digits=2)
# Returns: {-1.95: [file1, file2], -2.05: [file3, file4], ...}
```

### `settings_manager.py`

Centralized JSON settings management for the entire pipeline.

**Classes:**
- `SettingsManager`: Loads and provides settings for all pipeline stages
  - `from_input_folder(folder)`: Auto-detect settings file in input folder
  - `get_filename_parsing_config()` → `Dict`: Get pip0 filename parsing config
  - `get_preproc_config()` → `PreprocCfg`: Get pip1 preprocessing config
  - `get_gpr_configs()` → `Tuple[GPRCfg, ExportCfg]`: Get pip2 GPR configs
  - `get_summary_gpr_configs()` → `Tuple[SummaryGPRConfig, SummaryGPRHyperParams]`: Get pip3 configs
  - `get_efficiency_configs()` → `Tuple[DirParams, GlobalParams, ScaleParams, PlotParams]`: Get pip4 configs
  - `get_pip2_output_dir()`, `get_pip3_output_dir()`, etc.: Get output directories

---

## .cor File Format Reference

CorrWare `.cor` files are ASCII text with structure:

```
Begin Experiment: Potentiostatic
Potential:	-1.95
Pstat Title:	Sample_Name
Data Points:	5500
End Comments
-1.950000	-0.024200	0.100000    ← E (V), I (A/cm²), T (s)
...
End Experiment
```
---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           settings*.json                                     │
│                                                                             │
│  "pip0_dataloading": {                                                      │
│      "filename_parsing": {                                                  │
│          "delimiter": "_",                                                  │
│          "grouping_keys": [                                                 │
│              {"name": "potential", "token_index": 0, "dtype": "float",     │
│               "regex_extract": "^([+-]?\\d+\\.?\\d*)V$"},                  │
│              {"name": "pH", "token_index": 1, "dtype": "float",            │
│               "regex_extract": "^pH(\\d+\\.?\\d*)$"}                       │
│          ]                                                                  │
│      },                                                                     │
│      "primary_grouping_key": "potential",  ← Configurable (default: first) │
│      "fallback_to_file_content": false                                      │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SettingsManager.get_filename_parsing_config()                              │
│  → Returns dict with delimiter, grouping_keys, metadata_keys, etc.         │
│  config["primary_grouping_key"] → Derived from settings                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  main_GPR_FDA_pipeline.py                                                   │
│                                                                             │
│  filename_parsing_config = settings_manager.get_filename_parsing_config()  │
│  loader = DataLoader(..., filename_parsing_config=filename_parsing_config) │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataLoader.parse_file(file_path)                                           │
│                                                                             │
│  1. parse_filename(file_path, config) → Extract from filename              │
│     "-1.95V_pH1.48_..." → {"potential": -1.95, "pH": 1.48}                 │
│                                                                             │
│  2. If primary_grouping_key not found and fallback_to_file_content=True:   │
│     → Use value from parse_cor_file() (reads "Potential:" from file)       │
│                                                                             │
│  3. Return LoadedCurve with group_flags populated                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LoadedCurve.group_flags                                                    │
│  → {"potential": -1.95, "pH": 1.48, "sample_id": "Au-4R2", ...}            │
│                                                                             │
│  LoadedCurve.get_primary_value(primary_key)                                 │
│  → Returns group_flags[primary_key] (e.g., -1.95 for "potential")          │
│                                                                             │
│  DataLoader builds a group key from configured grouping flags:               │
│  → "pH=1.48|potential=-1.95"                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataLoadingResult.curves_by_primary_key                                    │
│  → {-1.95: [curve1, curve2], -2.05: [curve3, curve4], ...}                 │
│                                                                             │
│  DataLoadingResult.primary_key_values                                       │
│  → [-1.95, -2.05, -2.15] (sorted unique values)                            │
│                                                                             │
│  → Passed to pip1, pip2, pip3... for processing per group                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Group Keys (Generic Grouping System)

The pipeline uses a **configurable primary grouping key** system rather than hardcoded field names:

- **Primary Grouping Key**: The main dimension for aggregating curves (e.g., "potential", "pH", "temperature")
- **Group Key String**: Internal format `"key1=value1|key2=value2"` for multi-key grouping
- **Filename-Safe Format**: Converted to `"key1_value1_key2_value2"` for file names

**Example:**
```python
# Internal group key
group_key = "pH=1.48|potential=-1.95"

# Filename-safe version
filename_safe = "pH_1.48_potential_-1.95"
```

### Settings-Driven Configuration

All grouping and parsing behavior is controlled by `settings*.json`:

1. **`grouping_keys`**: Keys extracted from filenames for grouping
2. **`primary_grouping_key`**: Which key to use as the primary grouping dimension
3. **`metadata_keys`**: Additional metadata (not used for grouping)

If `primary_grouping_key` is not specified, it defaults to the first key in `grouping_keys`.

---

## Output to pip1

`DataLoadingResult.curves` contains `LoadedCurve` objects ready for pip1:

```python
curve.sample_id        # str: Sample identifier
curve.group_flags      # Dict: All grouping keys and values (e.g., {"potential": -1.95, "pH": 1.48})
curve.x_raw            # np.ndarray: Time in seconds
curve.y_raw            # np.ndarray: Current in A/cm²

# Get primary grouping value (e.g., potential)
primary_key = "potential"
curve.get_primary_value(primary_key)  # Returns -1.95
```

