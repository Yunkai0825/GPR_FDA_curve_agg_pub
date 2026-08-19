# pip2_individual_gpr - Individual GPR Fitting Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module fits Gaussian Process Regression (GPR) models to individual preprocessed curves:
- Fits GPR with Matern + WhiteKernel to each curve
- Generates predictions in both transformed and original coordinate spaces
- Computes validation metrics (MAE, RMSE)
- Exports CSV files with comprehensive metadata headers
- Creates visualization plots

---

## Module Structure

```
pip2_individual_gpr/
├── __init__.py                                 # Public API exports
├── pip2_individual_gpr_processor_orchestrator.py  # Main orchestrator (IndividualGPRProcessor, GPRFitResult, GPRProcessingResult)
├── gpr_config.py                               # Configuration dataclasses (GPRCfg, ExportCfg)
├── gpr_functions.py                            # Fitting, prediction, validation, and grid regulation
├── gpr_reader.py                               # Lossless reconstruction from exported CSV files
├── gpr_utilities.py                            # Plotting and export utilities
├── DEBUG_individual_GPR.py                     # Debug/test script
├── DEBUG_pip2_individual_GPR_Output/           # Debug output folder
└── README.md                                   # This file
```

---

## Quick Start

```python
from py_FDA_GPR_modules.pip0_dataloading import DataLoader, SettingsManager
from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor

# Load settings, data, and preprocessing configuration
manager = SettingsManager.from_input_folder("/path/to/input_folder")
loader = DataLoader(
    path_to_folder=manager.get_input_directory(),
    filename_parsing_config=manager.get_filename_parsing_config(),
)
preprocessor = DataPreprocessor(config=manager.get_preproc_config())
preprocessed = preprocessor.preprocess_all(loader.load_all().curves)

# Fit GPR models
gpr_cfg, export_cfg = manager.get_gpr_configs()
gpr_processor = IndividualGPRProcessor(
    gpr_config=gpr_cfg,
    export_config=export_cfg,
    output_directory="/path/to/output"
)
result = gpr_processor.fit_all(preprocessed.curves)
print(result.summary())
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `IndividualGPRProcessor` | `pip2_individual_gpr_processor_orchestrator.py` | Main orchestrator for GPR fitting |
| `GPRFitResult` | `pip2_individual_gpr_processor_orchestrator.py` | Result for single GPR fit |
| `GPRProcessingResult` | `pip2_individual_gpr_processor_orchestrator.py` | Batch processing result |
| `GPRCfg` | `gpr_config.py` | GPR algorithm configuration |
| `ExportCfg` | `gpr_config.py` | Export and plotting configuration |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `fit_individual_gprs(curves)` | `pip2_individual_gpr_processor_orchestrator.py` | Convenience function |
| `perform_gpr(x, y, cfg)` | `gpr_functions.py` | Fit GPR model |
| `validate_gpr(x, y, gpr, ...)` | `gpr_functions.py` | Compute validation metrics |
| `generate_predictions(gpr, ...)` | `gpr_functions.py` | Generate predictions with uncertainty |
| `compute_full_posterior_covariance(...)` | `gpr_functions.py` | Evaluate the latent posterior covariance |
| `refit_with_frozen_hyperparameters(...)` | `gpr_functions.py` | Refit on full data without reoptimizing the kernel |
| `regulate_to_shared_grid(...)` | `gpr_functions.py` | Evaluate a fitted model on a common group grid |
| `group_curves_by_primary_key(curves)` | `gpr_utilities.py` | Group by primary grouping key |
| `export_gpr_result_to_csv(result, ...)` | `gpr_utilities.py` | Export with metadata header |
| `export_covariance_matrix(result, ...)` | `gpr_utilities.py` | Export normalized posterior covariance for pip3 |
| `plot_individual_gpr(...)` | `gpr_utilities.py` | Create visualization plot |
| `save_skipped_samples_summary(...)` | `gpr_utilities.py` | Export skipped sample info |
| `reconstruct_gpr_result(...)` | `gpr_reader.py` | Reconstruct a result from prediction and covariance CSVs |

---

## File Details

### `pip2_individual_gpr_processor_orchestrator.py`

Main orchestrator for GPR fitting operations.

**Classes:**
- `IndividualGPRProcessor`: Coordinates GPR fitting, validation, export
  - `fit_single(curve, X_pred, index_id)` → `GPRFitResult`
  - `fit_all(curves)` → `GPRProcessingResult`
  
- `GPRFitResult`: Single curve result with predictions in both spaces
  - Identifiers: `sample_id`, `group_flags`, `index_id`
  - Transformed: `x_pred_transformed`, `y_pred_transformed`, `y_std_transformed`
  - Original: `x_pred`, `y_pred`, `y_std`
  - Scaling: `x_scaling`, `y_scaling`
  - Validation: `validation_mae`, `validation_rmse`
  - Model: `gpr_model`, `scaler_X`, `scaler_y`, `hyperparams`
  - Posterior framework: `posterior`, `validation_metrics_extended`, `physical_scale_factor`
  
- `GPRProcessingResult`: Batch result
  - `results`, `results_by_group`, `skipped`, `config`

### `gpr_config.py`

Configuration dataclasses.

**Classes:**
- `GPRCfg`: GPR algorithm parameters
  - `kernel`, `n_restarts_optimizer`, `alpha`, `normalize_y`
  - `num_X_pred_points_individual_default/high`: Prediction resolution
  - `shared_grid`: `SharedGridConfig` for group-wide prediction grids
  - `store_posterior_covariance`: Store covariance for curve aggregation
  - `covariance_storage_mode`: `full`, `diagonal`, `sparse`, or `cholesky`

- `SharedGridConfig`: Shared-grid and frozen-hyperparameter refit controls
  - `enabled`, `method`, `explicit_grid`, `auto_num_points`, `auto_padding_fraction`
  - `refit_on_full_data`

- `ExportCfg`: Export settings
  - `plot_individual_gpr`, `save_individual_csv`
  - `dpi`, `plot_format`

### `gpr_functions.py`

Core GPR fitting and prediction functions. The public API uses one unified
posterior implementation rather than separate legacy/v2 entry points.

- `PosteriorResult`: Dataclass for complete posterior distribution
  - Contains mean, std, full covariance matrix, scaling factors
  - Methods: `get_mean_original_units()`, `get_std_original_units()`, `get_covariance_original_units()`

- `perform_gpr(x_train, y_train, gpr_cfg)`: Fit and optimize a GPR model

- `generate_predictions(gpr, scaler_X, scaler_y, X_pred, physical_scale_factor, gpr_cfg)`:
  - Returns `PosteriorResult` with the configured covariance representation

- `validate_gpr(x_val, y_val, gpr, scaler_X, scaler_y)`:
  - Returns MAE, RMSE, NLPD, and one-/two-sigma calibration metrics

- `compute_full_posterior_covariance(gpr, scaler_X, X_pred)`:
  - Returns full posterior covariance matrix C_post

- `full_regulation_workflow(...)`: Fit, refit with frozen hyperparameters, and evaluate on a shared grid

### `gpr_utilities.py`

Plotting, export, and grouping utilities.

**Functions:**
- `group_curves_by_primary_key(curves, primary_key, round_digits)`: Group by primary key value
- `group_curves_by_key(curves, key_func)`: Generic grouping
- `export_gpr_result_to_csv(result, group_key, output_dir)`: Export with metadata
- `plot_individual_gpr(...)`: Create plot with training data and predictions
- `save_skipped_samples_summary(skipped, output_dir)`: Export skip reasons

**Posterior and reconstruction utilities:**
- `plot_posterior_covariance_diagnostics(result, output_dir)`: Visualize covariance structure
- `save_posterior_covariance(result, output_dir, format)`: Optional NPZ/NPY/CSV serialization
- `export_covariance_matrix(result, output_dir)`: Standard pipeline covariance CSV export

### `gpr_reader.py`

Reconstruct exported GPR results without refitting the original models.

**Classes:**
- `ReconstructedPosterior`: Normalized posterior with original-unit accessors
- `ReconstructedGPRResult`: Prediction grid, metadata, and reconstructed posterior

**Functions:**
- `reconstruct_gpr_result(gpr_csv_path, covariance_csv_path, auto_find_covariance)`
- `reconstruct_all_gpr_results(directory, pattern)`
- `verify_reconstruction(original_result, reconstructed_result, tolerance)`

---

## Posterior Framework

The aggregation-ready posterior keeps physical preprocessing scale separate
from the statistical standardization used during fitting.

### Posterior Distribution

$$P(|g\rangle||f\rangle) \propto \exp\left(-\frac{1}{2}\langle g-m_{post}|\hat{C}_{post}^{-1}|g-m_{post}\rangle\right)$$

### Scaling Separation

1. **Physical scaling** ($s_r$): From pip1 normalization (e.g., steady-state current)
2. **Statistical scaling**: StandardScaler applied in GPR fitting

### Variance Propagation

When transforming back to original units:
$$\sigma^2_{original} = s_r^2 \cdot \sigma^2_{stat} \cdot \sigma^2_{normalized}$$

### Covariance for Aggregation

The posterior covariance $\hat{C}_{post}$ is needed for proper curve aggregation in pip3:
$$\hat{C}_{agg}^{-1} = \sum_r w_r (\hat{C}_{post,r} + \hat{C}_{btw})^{-1}$$

---

## CSV Output Format

Each `Individual_GPR_*.csv` contains metadata header + data:

```
# METADATA_START
# sample_id: sample_001
# index_id: 1
# group_key: potential=-1.95
# group_flags: {"potential": -1.95}
# x_scaling_method: log_log10
# x_scaling_params: {"shift": 1e-09}
# y_scaling_method: peak
# y_scaling_params: {"factor": 0.025}
# validation_mae: 0.0234
# validation_rmse: 0.0312
# METADATA_END
x_pred_transformed,x_pred_original,y_pred_normalized,y_std_normalized
```

---

## Covariance Output Format

With `store_posterior_covariance=True`, the standard pipeline export is:

```
Covariance_Matrix_{sample_id}.csv
├── metadata header         # sample ID, dimensions, scaling methods, units
└── square covariance table # Normalized-y covariance indexed by x_pred
```

`save_posterior_covariance(...)` can additionally create `Posterior_*.npz`,
but that optional format is not the normal orchestrator output consumed by
pip3.

---

## Output to pip3

`GPRProcessingResult.results` contains `GPRFitResult` objects. 
pip3 loads from CSV files with metadata to reconstruct ScalingInfo.

**For curve aggregation**, pip3 needs:
- Normalized posterior mean/std from `Individual_GPR_*.csv`
- Posterior covariance from the matching `Covariance_Matrix_*.csv`
- Physical scale factors for proper variance weighting
- Common prediction grid (x_pred_transformed)
