# pip1_datapreprocessing - Data Preprocessing Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module preprocesses raw electrochemical transient data for GPR fitting:
- X (time) filtering and log transformation
- Y (current) normalization (peak or middle-average)
- Data downsampling in log-time space
- Tracking transformations via `ScalingInfo` for inverse operations

---

## Module Structure

```
pip1_datapreprocessing/
├── __init__.py                          # Public API exports
├── pip1_data_preprocessor_orchestrator.py # Main orchestrator (DataPreprocessor, PreprocessedCurve, PreprocessingResult)
├── preprocessing_functions.py           # Core functions + ScalingInfo class
├── preprc_downsampling.py               # Downsampling strategies (downsample_uniform_bins, DownsampleResult)
├── preproc_config.py                    # Configuration dataclass (PreprocCfg)
└── README.md                            # This file
```

---

## Quick Start

```python
from py_FDA_GPR_modules.pip0_dataloading import DataLoader
from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor, PreprocCfg

# Load raw data from pip0
loader = DataLoader(path_to_folder="/path/to/data")
loaded = loader.load_all()

# Preprocess with custom config
cfg = PreprocCfg(
    x_col='T (Seconds)',
    y_col='I (A/cm2)',
    max_points_set=500,
    min_x_cap=1e-4,
    max_x_cap=500.0,
    min_curve_range=60.0,
    y_threshold=0.01,
    x_scaling_method='log',
    x_scaling_params={'base': 'log10', 'shift': 1e-9},
    y_scaling_method='peak',
    y_scaling_params={},
)

preprocessor = DataPreprocessor(config=cfg)
result = preprocessor.preprocess_all(loaded.curves)
print(result.summary())
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `DataPreprocessor` | `pip1_data_preprocessor_orchestrator.py` | Main orchestrator for preprocessing |
| `PreprocessedCurve` | `pip1_data_preprocessor_orchestrator.py` | Container for preprocessed curve with transformations |
| `PreprocessingResult` | `pip1_data_preprocessor_orchestrator.py` | Result container with curves and groupings |
| `PreprocCfg` | `preproc_config.py` | Configuration dataclass |
| `ScalingInfo` | `preprocessing_functions.py` | Transformation tracking with `transform()`/`inverse_transform()` |
| `DownsampleResult` | `preprc_downsampling.py` | Result from downsampling operations |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `preprocess_curves(curves, config)` | `pip1_data_preprocessor_orchestrator.py` | Convenience function |
| `apply_x_filter(df, min_x, max_x, x_col)` | `preprocessing_functions.py` | Filter by x range |
| `filter_by_y_threshold(df, threshold, y_col)` | `preprocessing_functions.py` | Filter by y threshold |
| `downsample_uniform_bins(...)` | `preprc_downsampling.py` | Bin-based downsampling |

---

## File Details

### `pip1_data_preprocessor_orchestrator.py`

Main orchestrator for preprocessing operations.

**Classes:**
- `DataPreprocessor`: Coordinates filtering, transformation, downsampling
  - `preprocess_single(curve_data, group_flags)` → `Tuple[PreprocessedCurve, str]`
  - `preprocess_all(curves)` → `PreprocessingResult`
  
- `PreprocessedCurve`: Container with all preprocessing stages
  - `x_original`, `y_original` [seconds, A/cm²]
  - `x_transformed`, `y_transformed` [log(s), normalized] - full data
  - `x_train_transformed`, `y_train_transformed` [log(s), normalized] - downsampled
  - `x_scaling`, `y_scaling`: ScalingInfo objects
  - `get_validation_data()`: Returns full transformed data
  
- `PreprocessingResult`: Batch result container
  - `curves`, `curves_by_group`, `curves_by_primary_key`, `skipped`

### `preprocessing_functions.py`

Core preprocessing functions and ScalingInfo class.

**Classes:**
- `ScalingInfo`: Encapsulates transformation with inverse capability
  - `method`, `params`, `transform_func`, `inverse_func`
  - `transform(data)` → transformed data
  - `inverse_transform(data)` → original data
  - Factory methods: `identity()`, `log_transform()`, `divide_by_factor()`, 
    `standardize()`, `from_peak_normalization()`, `from_middle_average()`, `from_normalization()`

**Functions:**
- `apply_x_filter(df, min_x, max_x, x_col)`: Filter DataFrame by x range
- `filter_by_y_threshold(df, threshold, y_col)`: Filter by |y| threshold

### `preprc_downsampling.py`

Downsampling strategies for reducing data points.

**Classes:**
- `DownsampleResult`: Container for downsampling output

**Functions:**
- `downsample_uniform_bins(df, max_points, x_col, selection)`: Uniform-bin downsampling
- `downsample_data(df, max_points, ..., method)`: Dispatch to a selected strategy
- Available strategies: uniform, adaptive, gradient-preserving, and feature-aware

### `preproc_config.py`

Configuration dataclass for preprocessing parameters.

**Classes:**
- `PreprocCfg`: All preprocessing parameters
  - Column names: `x_col`, `y_col`
  - Filtering: `enable_filtering`, `max_points_set`, `min_x_cap`, `max_x_cap`, `min_curve_range`, `y_threshold`
  - X scaling: `x_scaling_method`, `x_scaling_params`
  - Y scaling: `y_scaling_method`, `y_scaling_params`
  - Grouping: `group_round_digits`

---

## Data Flow

```
LoadedCurve (pip0)
    │  x_raw: seconds, y_raw: A/cm²
    ▼
┌─────────────────────────────────┐
│ 1. X Filter (min/max_x_cap)     │
│ 2. X Transform (log if enabled) │
│ 3. Y Normalize (peak/middle)    │
│ 4. Y Threshold filter           │
│ 5. Downsample to max_points     │
└─────────────────────────────────┘
    │
    ▼
PreprocessedCurve (to pip2)
    x_train_transformed: log(seconds)
    y_train_transformed: normalized
    x_scaling, y_scaling: ScalingInfo
```

---

## Output to pip2

`PreprocessingResult.curves` contains `PreprocessedCurve` objects:

```python
curve.x_train_transformed    # np.ndarray (n, 1): Training x in log(seconds)
curve.y_train_transformed    # np.ndarray (n,): Training y normalized
curve.x_scaling              # ScalingInfo: For inverse x transform
curve.y_scaling              # ScalingInfo: For inverse y transform
curve.get_validation_data()  # Full transformed data for validation
```
