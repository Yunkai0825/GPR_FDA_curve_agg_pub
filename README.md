# GPR-FDA: Gaussian Process Regression for Functional Data Analysis

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

**Version:** 1.0.0

## Overview

GPR-FDA is a complete pipeline for analyzing electrochemical potentiostatic transient data using Gaussian Process Regression (GPR). The pipeline:

1. **Loads** raw `.cor` files from CorrWare with configurable grouping keys
2. **Preprocesses** time-series data (log transformation, normalization, downsampling)
3. **Fits** individual GPR models to each curve
4. **Aggregates** individual GPRs into summary curves with variance-based weighting
5. **Evaluates** data efficiency via Monte-Carlo learning curves
6. **Batch tests** all algorithm parameter combinations for comparison

**Key Feature**: The primary grouping key (e.g., "potential", "pH", "temperature") is fully configurable via settings.json, allowing the pipeline to work with various experimental parameters.

---

## Installation

GPR-FDA requires Python 3.10 or newer. Create an isolated environment and install the direct runtime dependencies from `requirements.txt`:

```bash
git clone https://github.com/Yunkai0825/GPR_FDA_curve_agg_pub.git
cd GPR_FDA_curve_agg_pub

python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell (run this instead of the line above)
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the command-line interface after installation:

```bash
python main_GPR_FDA_pipeline.py --help
```

The required third-party packages and their roles are documented in [`requirements.txt`](requirements.txt). A virtual environment is recommended to avoid conflicts with system Python packages.

---

## Quick Start

```bash
# Run the preserved CCNF_CTAB_PT dataset across all configured subfolders
python main_GPR_FDA_pipeline.py --path "./_input/CCNF_CTAB_PT" --batch

# Run only Individual GPR (Step 1)
python main_GPR_FDA_pipeline.py --path /path/to/data --step 1

# Run only Batch Testing (Step 2)
python main_GPR_FDA_pipeline.py --path /path/to/data --step 2

# Run only summary aggregation with FGPR (Step 2a)
python main_GPR_FDA_pipeline.py --path /path/to/data --step 2a --method fgpr

# Combine all experiment subfolders under one parent folder
python main_GPR_FDA_pipeline.py --path /path/to/parent --batch
```

---

## Directory Structure

```
GPR_FDA_curve_agg_pub/
├── main_GPR_FDA_pipeline.py           # Main entry point
├── requirements.txt                   # Python runtime dependencies
├── CITATION.cff                       # Machine-readable citation metadata
├── py_FDA_GPR_modules/                # Core pipeline modules
│   ├── __init__.py
│   ├── pip0_dataloading/              # Step 1.1: Load .cor files
│   ├── pip1_datapreprocessing/        # Step 1.2: Preprocess curves
│   ├── pip2_individual_gpr/           # Step 1.3: Fit individual GPRs
│   ├── pip3_FDA_scoring_and_aggregations/  # Summary GPR aggregation
│   ├── pip4_efficiency_eval/          # Data efficiency evaluation
│   └── pip5_batch_setting_grid_testing/    # Batch grid testing
├── _input/                            # Input data folders
│   ├── _template_settings.json        # Complete settings template
│   └── {dataset_name}/
│       ├── settings*.json             # Dataset-level pipeline settings
│       └── {experiment_folder}/
│           └── *.cor                  # Raw CorrWare files
├── _output/                           # Output results
│   └── {dataset_name}_output/
│       ├── individual_GPR/            # Step 1 output
│       └── batch_testing/             # Step 2 output
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           main_GPR_FDA_pipeline.py                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Individual GPR Processing                                   │   │
│  │                                                                     │   │
│  │   ┌──────────┐    ┌──────────────┐    ┌─────────────────┐          │   │
│  │   │   pip0   │ -> │     pip1     │ -> │      pip2       │          │   │
│  │   │ Loading  │    │ Preprocessing│    │ Individual GPR  │          │   │
│  │   └──────────┘    └──────────────┘    └─────────────────┘          │   │
│  │        │                 │                     │                    │   │
│  │   .cor files     x/y filtering          GPR fitting                │   │
│  │   MD5 dedup      log transform          predictions                │   │
│  │   metadata       normalization          CSV + plots                │   │
│  │                  downsampling                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Batch Grid Testing (6 baseline + optional methods)          │   │
│  │                                                                     │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │     pip5     │ -> │     pip3     │ -> │     pip4     │         │   │
│  │   │ Orchestrator │    │ Summary GPR  │    │  Efficiency  │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  │        │                     │                    │                 │   │
│  │   up to 10 combos     uncertainty models    learning curves        │   │
│  │   comparison plots    variance aggregation  MC sampling            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input/Output Specification

### Input Requirements

1. **Data Files**: CorrWare `.cor` files in the experiment folder
   - Format: Potentiostatic transient experiments
   - Contains: Time (T), Current (I), Potential (E) columns
   - Example: `-1.95V_pH1.48_Na2SO4_sample.cor`

2. **Settings File**: `settings*.json` in the dataset folder (or the data folder for a single-folder run)
   - Controls all preprocessing and algorithm parameters
   - See [Settings JSON Reference](#settings-json-reference) below

### Input Folder Structure

```
_input/
└── CCNF_CTAB_PT/                          # Dataset folder
    ├── settings_summary_GPR_CCNF_CTAB_PT.json # Dataset settings
    ├── full_pipeline_output.log              # Successful reference-run log
    ├── 20230422 FeNiCoCu acidB-BU/        # Experiment folders
    │   └── *.cor
    ├── ...
    └── 20230616 FeNiCoCu acidB-BWR6/
        └── *.cor
```

### Output Structure

```
_output/
└── CCNF_CTAB_PT_output/                  # Preserved full-dataset run
    │
    ├── individual_GPR/                     # Step 1 output
    │   ├── Individual_GPR_*.csv               # Normalized predictions + metadata
    │   ├── Covariance_Matrix_*.csv            # Posterior covariance matrices
    │   ├── GPR_Plot_*.png                     # Individual curve plots
    │   ├── Validation_Results_potential_*.csv # Validation metrics per potential
    │   └── settings_used.json                 # Settings snapshot
    │
    └── batch_testing/                      # Step 2 output
        ├── NS_norm__WM_equal__WS_curve/       # Permutation 1
        │   ├── summary_gpr/
        │   │   └── iterative/
        │   │       ├── Summary_GPR_potential_*.csv
        │   │       └── Summary_GPR_potential_*.png
        │   └── learning_curve/
        │       ├── LearningCurve_*_summary.csv
        │       ├── LearningCurve_*_detailed.csv
        │       └── Efficiency_LearningCurve_*.png
        ├── NS_norm__WM_iterative__WS_curve/   # Permutation 2
        ├── NS_norm__WM_iterative__WS_point/   # Permutation 3
        ├── NS_real__WM_equal__WS_curve/       # Permutation 4
        ├── NS_real__WM_iterative__WS_curve/   # Permutation 5
        ├── NS_real__WM_iterative__WS_point/   # Permutation 6
        ├── NS_norm__AM_fgpr/                  # Optional FGPR, normalized scale
        ├── NS_real__AM_fgpr/                  # Optional FGPR, real scale
        ├── NS_norm__AM_student_t/             # Optional Student-t, normalized scale
        ├── NS_real__AM_student_t/             # Optional Student-t, real scale
        ├── comparisons/                       # Cross-permutation plots
        │   ├── SummaryGPR_Comparison_*.png
        │   ├── Efficiency_Comparison_*.png
        │   └── Combined_Comparison_*.png
        └── settings_used.json
```

The repository's preserved reference output is
`_output/CCNF_CTAB_PT_output/` (approximately 2.58 GB). It corresponds to the
versioned input under `_input/CCNF_CTAB_PT/` and is retained as the reference
result for the release.
`_input/CCNF_CTAB_PT/full_pipeline_output.log` records the successful
CCNF_CTAB_PT batch run that produced it. Superseded run logs and intermediate
output trees are not included in the publication repository.

### Output File Formats

| File Pattern | Description | Key Columns |
|--------------|-------------|-------------|
| `Individual_GPR_*.csv` | Individual GPR predictions | `x_pred_transformed`, `x_pred_original`, `y_pred_normalized`, `y_std_normalized` |
| `Covariance_Matrix_*.csv` | Individual posterior covariance | Square normalized covariance matrix indexed by the prediction grid |
| `Summary_GPR_*.csv` | Iterative aggregated summary curve | `x_transformed`, `x_real`, `y_real`, `y_normalised`, `Std_real`, `Std_normalised` |
| `FGPR_Curve_*.csv` | Functional-GPR aggregate | Real/normalized aggregate and predictive moments, `sigma_btw_squared`, `nll` |
| `StudentT_Curve_*.csv` | Robust Student-t aggregate | Real/normalized aggregate and predictive moments, `sigma_btw_squared`, `nu` |
| `LearningCurve_*_detailed.csv` | MC learning curve runs | `subset_size`, `mc_index`, `error`, `time_s`, `n_iterations`, `y_std`, `y_std_real`, `y_std_normalised` |
| `Converged_Weights_*.csv` | Final aggregation weights | `sample_id`, `weight` |
| `Validation_Results_*.csv` | GPR validation metrics | `sample_id`, `MAE`, `RMSE` |

---

## Settings JSON Reference

The settings JSON file controls all pipeline parameters. Place it in the same folder as your `.cor` files.

### Settings Template

The canonical, complete template is [`_input/_template_settings.json`](_input/_template_settings.json). Copy that file into a dataset folder and customize it for the experiment. The abbreviated example below shows the overall structure.

```json
{
    "_meta": {
        "version": "1.0.0",
        "description": "GPR-FDA Pipeline Settings",
        "author": "Your Name"
    },
    "output_structure": {
        "base_output_dir": "_output",
        "pip2_individual_gpr": "individual_GPR",
        "pip3_scoring_aggregations": "summary_gpr",
        "pip4_efficiency_eval": "learning_curve",
        "pip5_batch_testing": "batch_testing"
    },
    "pip0_dataloading": {
        "file_extension": ".cor",
        "filename_parsing": {
            "delimiter": "_",
            "grouping_keys": [
                {
                    "name": "potential",
                    "token_index": 0,
                    "dtype": "float",
                    "regex_extract": "^([+-]?\\d+\\.?\\d*)V$",
                    "description": "Applied potential in Volts (e.g., -1.95V → -1.95)"
                }
            ],
            "metadata_keys": [
                {
                    "name": "sample_id",
                    "token_index": 6,
                    "dtype": "string",
                    "description": "Sample identifier"
                }
            ]
        },
        "primary_grouping_key": "potential",
        "fallback_to_file_content": false
    },
    "pip1_datapreprocessing": {
        "column_names": {
            "x_col": "T (Seconds)",
            "y_col": "I (A/cm2)"
        },
        "filtering": {
            "max_points_set": 500,
            "min_x_cap": 1e-4,
            "max_x_cap": 500.0,
            "min_curve_range": 60.0,
            "y_threshold": 0.01
        },
        "x_scaling": {
            "method": "log",
            "params": {"base": "log10", "shift": 1e-9}
        },
        "y_scaling": {
            "method": "middle_average",
            "params": {"start_fraction": 0.5, "end_fraction": 0.9}
        },
        "grouping": {
            "group_round_digits": 2
        }
    },
    "pip2_individual_gpr": {
        "gpr_params": {
            "n_restarts_optimizer": 5,
            "alpha": 0.1,
            "normalize_y": true,
            "num_X_pred_points_individual_default": 500,
            "local_var_val_flag": true
        },
        "kernel": {
            "constant_bounds": [1e-2, 1e4],
            "matern_length_scale": 1.0,
            "matern_length_scale_bounds": [1e-2, 1e3],
            "matern_nu": 1.5,
            "white_noise_level": 1.0,
            "white_noise_bounds": [1e-12, 1e1]
        },
        "export": {
            "plot_individual_gpr": true,
            "dpi": 150
        }
    },
    "pip3_summary_gpr": {
        "aggregation": {
            "weight_mode": "iterative",
            "weight_scope": "curve",
            "include_within_variance": true,
            "include_between_variance": true,
            "variance_aggregation_scale": "normalised",
            "normalization_summary": true,
            "enable_fgpr": true,
            "enable_student_t": true
        },
        "plotting": {
            "plot_individual_gprs": true,
            "min_time_cap": 0.01,
            "max_time_cap": 500.0
        },
        "hyperparams": {
            "convergence_tol": 1e-6,
            "confidence_level": 0.95,
            "num_interp_points": 500,
            "fgpr_structured_btw": false,
            "student_t_nu": 5.0,
            "student_t_optimize_nu": true,
            "student_t_nu_bounds": [1.0, 500.0],
            "student_t_max_iterations": 100,
            "student_t_convergence_tol": 1e-6
        }
    },
    "pip4_efficiency_eval": {
        "global": {
            "metric": "rmse",
            "base_repeats": 50,
            "max_enum": 20,
            "random_seed": 42
        },
        "scale": {
            "use_log_error": true,
            "normalize_w_rbar": true
        }
    },
    "pip5_batch_testing": {
        "include_fgpr": true,
        "include_student_t": true,
        "run_summary_gpr": true,
        "run_efficiency_eval": true,
        "run_comparison_plots": true
    }
}
```

### Settings Field Reference

#### `pip0_dataloading` - Data Loading & Filename Parsing

This section configures how raw files are loaded and how grouping keys are extracted from filenames.

**File Settings:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file_extension` | string | `".cor"` | File extension to search for |
| `primary_grouping_key` | string | First key in `grouping_keys` | Primary key for grouping curves (e.g., "potential", "pH") |
| `fallback_to_file_content` | bool | `true` | If filename parsing fails, extract from file content |

**Filename Parsing:**

| Field | Type | Description |
|-------|------|-------------|
| `filename_parsing.delimiter` | string | Character to split filename into tokens (e.g., `"_"`) |
| `filename_parsing.grouping_keys` | array | Keys used for grouping curves (required for aggregation) |
| `filename_parsing.metadata_keys` | array | Additional metadata extracted but not used for grouping |

**Key Definition Schema:**

Each key in `grouping_keys` or `metadata_keys` follows this schema:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Key name (e.g., `"potential"`, `"pH"`) |
| `token_index` | int | Yes | 0-based index of token in filename. Use `-1` for last token. |
| `dtype` | string | Yes | Data type: `"float"`, `"int"`, or `"string"` |
| `regex_extract` | string | No | Regex pattern with capture group to extract value from token |
| `description` | string | No | Human-readable description |

**Example: Parsing This Filename**

Filename: `-1.95V_pH1.48_Na2SO4_H3Cit_NiFeCoCu_acidB(BWR6)_Au-4R2_CTAB_diaphram_20230616.cor`

Tokens (split by `_`):
```
[0] -1.95V          → potential: -1.95 (via regex "^([+-]?\d+\.?\d*)V$")
[1] pH1.48          → pH: 1.48 (via regex "^pH(\d+\.?\d*)$")
[2] Na2SO4          → electrolyte: "Na2SO4"
[3] H3Cit           → additive: "H3Cit"
[4] NiFeCoCu        → alloy: "NiFeCoCu"
[5] acidB(BWR6)     → (skipped)
[6] Au-4R2          → sample_id: "Au-4R2"
[7] CTAB            → (skipped)
[8] diaphram        → (skipped)
[9] 20230616.cor    → date: "20230616" (via regex on last token)
```

**Settings JSON Example:**

```json
"pip0_dataloading": {
    "file_extension": ".cor",
    "filename_parsing": {
        "delimiter": "_",
        "grouping_keys": [
            {
                "name": "potential",
                "token_index": 0,
                "dtype": "float",
                "regex_extract": "^([+-]?\\d+\\.?\\d*)V$",
                "description": "Applied potential in Volts"
            },
            {
                "name": "pH",
                "token_index": 1,
                "dtype": "float",
                "regex_extract": "^pH(\\d+\\.?\\d*)$",
                "description": "Solution pH value"
            }
        ],
        "metadata_keys": [
            {
                "name": "sample_id",
                "token_index": 6,
                "dtype": "string"
            }
        ]
    },
    "primary_grouping_key": "potential",
    "fallback_to_file_content": false
}
```

#### `output_structure` - Output Directory Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_output_dir` | string | `"_output"` | Base output directory |
| `pip2_individual_gpr` | string | `"individual_GPR"` | Subdirectory for individual GPR output |
| `pip3_scoring_aggregations` | string | `"summary_gpr"` | Subdirectory for summary GPR output |
| `pip4_efficiency_eval` | string | `"learning_curve"` | Subdirectory for efficiency evaluation |
| `pip5_batch_testing` | string | `"batch_testing"` | Subdirectory for batch-grid output and comparisons |

#### `pip1_datapreprocessing` - Preprocessing Parameters

**Column Names:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `column_names.x_col` | string | `"x"` | X-axis column name in input data |
| `column_names.y_col` | string | `"y"` | Y-axis column name in input data |

**Filtering:**

| Field | Type | Default | Unit | Description |
|-------|------|---------|------|-------------|
| `filtering.max_points_set` | int | `1000` | - | Maximum points after downsampling |
| `filtering.min_x_cap` | float | `1e-4` | seconds | Minimum time to include |
| `filtering.max_x_cap` | float | `1e4` | seconds | Maximum time to include |
| `filtering.min_curve_range` | float | `60.0` | seconds | Minimum curve duration required |
| `filtering.y_threshold` | float | `0.01` | normalized | Minimum |y| to keep after normalization |

**X Scaling:**

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `x_scaling.method` | string | `"log"`, `"identity"` | X transformation method |
| `x_scaling.params.base` | string | `"log10"`, `"e"` | Logarithm base |
| `x_scaling.params.shift` | float | `1e-9` | Shift for log(x + shift) |

**Y Scaling:**

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `y_scaling.method` | string | `"peak"`, `"middle_average"` | Y normalization method |
| `y_scaling.params.start_fraction` | float | `0.5` | Start fraction for middle_average |
| `y_scaling.params.end_fraction` | float | `0.9` | End fraction for middle_average |

- **`peak`**: Divide by max absolute value → range approximately [-1, 1]
- **`middle_average`**: Divide by mean of middle section → captures quasi-steady-state behavior

**Grouping:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `grouping.group_round_digits` | int | `2` | Decimal places for potential grouping |

#### `pip2_individual_gpr` - GPR Fitting Parameters

**GPR Algorithm:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpr_params.n_restarts_optimizer` | int | `5` | Number of optimizer restarts for kernel hyperparameters |
| `gpr_params.alpha` | float | `0.1` | Noise level (regularization) added to diagonal |
| `gpr_params.normalize_y` | bool | `true` | Whether to normalize target values internally |
| `gpr_params.num_X_pred_points_individual_default` | int | `500` | Prediction points for output |
| `gpr_params.local_var_val_flag` | bool | `true` | Enable local uncertainty from validation residuals |

**Kernel Configuration:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kernel.constant_bounds` | [float, float] | `[1e-2, 1e4]` | Bounds for constant kernel |
| `kernel.matern_length_scale` | float | `1.0` | Initial Matern length scale |
| `kernel.matern_length_scale_bounds` | [float, float] | `[1e-2, 1e3]` | Length scale bounds |
| `kernel.matern_nu` | float | `1.5` | Matern smoothness parameter (0.5, 1.5, 2.5) |
| `kernel.white_noise_level` | float | `1.0` | Initial white noise level |
| `kernel.white_noise_bounds` | [float, float] | `[1e-12, 1e1]` | White noise bounds |

**Export:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `export.plot_individual_gpr` | bool | `true` | Generate plot images |
| `export.dpi` | int | `150` | Plot resolution |

#### `pip3_summary_gpr` - Summary GPR Aggregation

**Aggregation:**

| Field | Type | Options | Description |
|-------|------|---------|-------------|
| `aggregation.weight_mode` | string | `"equal"`, `"iterative"` | Weight optimization mode |
| `aggregation.weight_scope` | string | `"curve"`, `"point"` | One weight per curve or per point |
| `aggregation.include_within_variance` | bool | `true` | Include within-model variance |
| `aggregation.include_between_variance` | bool | `true` | Include between-model variance |
| `aggregation.variance_aggregation_scale` | string | `"real"`, `"normalised"` | Scale for variance computation |
| `aggregation.normalization_summary` | bool | `true` | Normalize before aggregation |
| `aggregation.enable_fgpr` | bool | `true` | Enable covariance-aware functional GPR aggregation |
| `aggregation.enable_student_t` | bool | `true` | Enable robust Student-t curve aggregation |

**Plotting:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `plotting.plot_individual_gprs` | bool | `true` | Overlay individual curves |
| `plotting.individual_curve_alpha` | float | `0.20` | Individual curve transparency |
| `plotting.min_time_cap` | float | `1e-4` | Minimum time for x-axis |
| `plotting.max_time_cap` | float | `1e4` | Maximum time for x-axis |

**Hyperparameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hyperparams.max_iterations` | int/null | `null` | Max iterations (null = until convergence) |
| `hyperparams.convergence_tol` | float | `1e-6` | Weight convergence tolerance |
| `hyperparams.epsilon` | float | `1e-12` | Small constant for numerical stability |
| `hyperparams.confidence_level` | float | `0.95` | Confidence level for uncertainty bands |
| `hyperparams.num_interp_points` | int | `500` | Interpolation grid size |
| `hyperparams.fgpr_min_scale_factor_ratio` | float | `0.0` | Optional FGPR scale-factor outlier threshold; zero disables it |
| `hyperparams.fgpr_structured_btw` | bool | `false` | Use structured rather than scalar between-curve covariance |
| `hyperparams.student_t_nu` | float | `5.0` | Initial Student-t degrees of freedom |
| `hyperparams.student_t_optimize_nu` | bool | `true` | Optimize the Student-t degrees of freedom |
| `hyperparams.student_t_nu_bounds` | [float, float] | `[1.0, 500.0]` | Bounds used when optimizing degrees of freedom |
| `hyperparams.student_t_max_iterations` | int | `100` | Maximum Student-t aggregation iterations |
| `hyperparams.student_t_convergence_tol` | float | `1e-6` | Student-t weight-convergence tolerance |

#### `pip4_efficiency_eval` - Data Efficiency Evaluation

**Global:**

| Field | Type | Options/Default | Description |
|-------|------|-----------------|-------------|
| `global.metric` | string | `"rmse"`, `"mae"`, `"max"` | Error metric type |
| `global.base_repeats` | int | `1000` | MC repeats when combinatorics large |
| `global.max_enum` | int | `1000` | Threshold for exhaustive enumeration |
| `global.q_low` | float | `0.25` | Lower quantile for IQR |
| `global.q_high` | float | `0.75` | Upper quantile for IQR |
| `global.random_seed` | int | `42` | Random seed for reproducibility |

**Scale:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scale.use_log_error` | bool | `true` | Apply log transform to error |
| `scale.log_base_error` | string | `"10"` | Log base: `"10"` or `"e"` |
| `scale.normalize_w_rbar` | bool | `true` | Normalize error by reference average |

#### `pip5_batch_testing` - Batch Grid Controls

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `include_fgpr` | bool | `true` | Add normalized- and real-scale FGPR combinations |
| `include_student_t` | bool | `true` | Add normalized- and real-scale Student-t combinations |
| `run_summary_gpr` | bool | `true` | Run summary aggregation for selected combinations |
| `run_efficiency_eval` | bool | `true` | Run learning-curve evaluation for selected combinations |
| `run_comparison_plots` | bool | `true` | Produce cross-combination comparison plots |

---

## Batch Testing Grid

Step 2 always defines six non-redundant equal/iterative combinations. Enabling FGPR adds two combinations, and enabling Student-t adds another two, for up to ten combinations:

| # | normalization_summary | weight_mode | weight_scope | Folder Name |
|---|----------------------|-------------|--------------|-------------|
| 1 | True | equal | curve | `NS_norm__WM_equal__WS_curve` |
| 2 | True | iterative | curve | `NS_norm__WM_iterative__WS_curve` |
| 3 | True | iterative | point | `NS_norm__WM_iterative__WS_point` |
| 4 | False | equal | curve | `NS_real__WM_equal__WS_curve` |
| 5 | False | iterative | curve | `NS_real__WM_iterative__WS_curve` |
| 6 | False | iterative | point | `NS_real__WM_iterative__WS_point` |
| 7 | True | FGPR | curve | `NS_norm__AM_fgpr` |
| 8 | False | FGPR | curve | `NS_real__AM_fgpr` |
| 9 | True | Student-t | curve | `NS_norm__AM_student_t` |
| 10 | False | Student-t | curve | `NS_real__AM_student_t` |

The equal-plus-point configuration is excluded because it is redundant with equal-plus-curve. Set `pip5_batch_testing.include_fgpr` and `pip5_batch_testing.include_student_t` in the settings file to control the optional combinations.

---

## Module Reference

Each module has its own README.md with detailed documentation:

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| [pip0_dataloading](py_FDA_GPR_modules/pip0_dataloading/README.md) | Load .cor files | `DataLoader`, `SettingsManager` |
| [pip1_datapreprocessing](py_FDA_GPR_modules/pip1_datapreprocessing/README.md) | Preprocess curves | `DataPreprocessor`, `ScalingInfo` |
| [pip2_individual_gpr](py_FDA_GPR_modules/pip2_individual_gpr/README.md) | Fit individual GPRs | `IndividualGPRProcessor`, `GPRFitResult` |
| [pip3_FDA_scoring_and_aggregations](py_FDA_GPR_modules/pip3_FDA_scoring_and_aggregations/README.md) | Iterative, FGPR, and Student-t aggregation | `SummaryGPROrchestrator`, `compute_summary_gpr` |
| [pip4_efficiency_eval](py_FDA_GPR_modules/pip4_efficiency_eval/README.md) | Data efficiency | `EfficiencyOrchestrator`, `learning_curve` |
| [pip5_batch_setting_grid_testing](py_FDA_GPR_modules/pip5_batch_setting_grid_testing/README.md) | Batch testing | `BatchTestingOrchestrator` |

---

## Dependencies

The direct runtime dependencies are maintained in [`requirements.txt`](requirements.txt):

| Package | Minimum version | Used for |
|---------|-----------------|----------|
| NumPy | 1.24 | Arrays, linear algebra, and sampling |
| pandas | 1.5 | Tabular data processing and CSV I/O |
| SciPy | 1.10 | Optimization, statistics, interpolation, signal processing, and matrix routines |
| scikit-learn | 1.3 | Gaussian-process regression, kernels, scaling, and validation metrics |
| Matplotlib | 3.7 | Result and diagnostic plotting |

Use the [Installation](#installation) instructions instead of installing the packages individually.

---

## Usage Examples

### Example 1: Run Full Pipeline

```bash
python main_GPR_FDA_pipeline.py --path "./_input/CCNF_CTAB_PT" --batch
```

### Example 2: Programmatic Usage

```python
from py_FDA_GPR_modules.pip0_dataloading import DataLoader, SettingsManager
from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor

# Load dataset-level settings
input_folder = "_input/CCNF_CTAB_PT"
manager = SettingsManager.from_input_folder(input_folder)

# Step 1.1: Combine the configured experiment subfolders
loaded = DataLoader.load_from_subdirectories(
    parent_folder=input_folder,
    subdirectories=manager.get_data_subdirectories(),
    filename_parsing_config=manager.get_filename_parsing_config(),
)

# Step 1.2: Preprocess
preprocessor = DataPreprocessor(config=manager.get_preproc_config())
preprocessed = preprocessor.preprocess_all(loaded.curves)

# Step 1.3: Fit GPR
gpr_cfg, export_cfg = manager.get_gpr_configs()
processor = IndividualGPRProcessor(
    gpr_config=gpr_cfg,
    export_config=export_cfg,
    output_directory=manager.get_pip2_output_dir()
)
results = processor.fit_all(preprocessed.curves)
```

### Example 3: Run Summary GPR Only

```python
from pathlib import Path

from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import (
    SummaryGPROrchestrator,
    SummaryGPRConfig,
    SummaryGPRHyperParams,
)

cfg = SummaryGPRConfig(
    input_directory=Path("_output/.../individual_GPR"),
    weight_mode="iterative",
    weight_scope="curve",
)
hp = SummaryGPRHyperParams()

orchestrator = SummaryGPROrchestrator(cfg, hp)
results = orchestrator.process_all()
```

---

## Troubleshooting

### Common Issues

1. **"No settings*.json found"**
   - Ensure a `settings*.json` file exists in your input folder
   - Copy from an existing dataset and modify

2. **"No curves loaded"**
   - Check that `.cor` files are in the input folder
   - Verify `.cor` file format (must be CorrWare potentiostatic format)

3. **"No valid curves after preprocessing"**
   - Curves may be too short (check `min_curve_range`)
   - Time range may be outside `min_x_cap` to `max_x_cap`
   - Increase `max_x_cap` or decrease `min_curve_range`

4. **GPR fitting issues**
   - Try increasing `alpha` (regularization) if fits are unstable
   - Reduce `n_restarts_optimizer` for faster (less optimal) fits

---

## Notes for AI Assistants (Copilot)

This section provides guidance for AI coding assistants modifying this codebase for more general usages.

### Architecture Principles

1. **Pipeline Modularity**: Each `pip*` module is self-contained with its own `__init__.py` exposing public APIs. Modifications should maintain this separation.

2. **Settings-Driven Configuration**: All parameters flow through `SettingsManager`. To add new parameters:
   - Add the field to the settings JSON schema
   - Add a getter method in `settings_manager.py`
   - Update the relevant module's config dataclass

3. **Configurable Grouping System**: The primary grouping key is configurable rather than hardcoded:
   - Set `primary_grouping_key` in settings.json (defaults to first `grouping_keys` entry)
   - Use `curves_by_primary_key` instead of domain-specific accessors
   - Use `curve.get_primary_value(key)` to get grouping values

4. **Dataclass Contracts**: Each pipeline stage uses typed dataclasses for input/output:
   - `pip0`: `LoadedData` → `pip1`: `PreprocessedData` → `pip2`: `GPRFitResult`
   - Extend these dataclasses when adding new output fields

### Common Modification Patterns

#### Adding a New Data Format (Beyond .cor)

1. Create a new loader in `pip0_dataloading/` (e.g., `csv_loader.py`)
2. Implement the same interface as `DataLoader`:
   ```python
class CSVLoader:
    def load_all(self) -> LoadedData:
        ...
   ```
3. Update `__init__.py` to expose the new loader
4. Modify `main_GPR_FDA_pipeline.py` to detect and use the appropriate loader

#### Adding a New Preprocessing Method

1. Edit `py_FDA_GPR_modules/pip1_datapreprocessing/preprocessing_functions.py`
2. Add new method to `DataPreprocessor` class
3. Add corresponding settings in `pip1_datapreprocessing` JSON section
4. Update `settings_manager.py` → `get_preproc_config()`

#### Adding a New Kernel Type

1. Edit `_default_kernel()` in `py_FDA_GPR_modules/pip2_individual_gpr/gpr_config.py`
2. Update kernel construction in `py_FDA_GPR_modules/pip0_dataloading/settings_manager.py`
3. Add kernel parameters to the `pip2_individual_gpr.kernel` settings section
4. Update `GPRCfg` if the new kernel requires additional configuration fields

#### Adding a New Aggregation Method

1. Add the method implementation under `py_FDA_GPR_modules/pip3_FDA_scoring_and_aggregations/`
2. Integrate it in `pip3_summary_gpr_orchestrator.py` and export its public API from `__init__.py`
3. Add the method name and hyperparameters to the settings template and `SettingsManager`
4. Update `SummaryGPRConfig`, `SummaryGPRHyperParams`, batch-grid generation, and the pip3 module README

#### Adding a New Efficiency Metric

1. Edit `pip4_efficiency_eval/learning_curve.py`
2. Add metric computation in `error_metric()`
3. Add metric option to `pip4_efficiency_eval.global.metric` settings

### Key Files to Understand

| File | Purpose | Modify When... |
|------|---------|----------------|
| `settings_manager.py` | Central configuration | Adding any new parameter |
| `main_GPR_FDA_pipeline.py` | Entry point & orchestration | Changing pipeline flow |
| `pip*/README.md` | Module documentation | After any module changes |
| `pip*/__init__.py` | Public API exports | Adding new public classes |

### Testing Modifications

1. Use `_tmp_subset/` folder with small CSV files for quick testing
2. Run Step 1 only (`--step 1`) to test preprocessing/GPR changes
3. Run Step 2 only (`--step 2`) to test aggregation/efficiency changes
4. Check `settings_used.json` in output to verify settings propagation

### Extending to Non-Electrochemical Data

The pipeline is generalizable to any time-series regression problem:

1. **Input**: Replace `.cor` loader with your format (CSV, HDF5, etc.)
2. **Preprocessing**: Adjust `x_scaling`/`y_scaling` methods for your domain
3. **Grouping**: Modify `group_round_digits` or grouping logic for your metadata
4. **Output**: Column names are configurable; adjust as needed

#### Data Type Requirements

Each curve must be a 2D time-series with:
- **X column**: Independent variable (e.g., time in seconds)
- **Y column**: Dependent variable (e.g., current density)
- **Metadata**: At minimum, a grouping key (e.g., potential, temperature, sample ID)

The pipeline expects curves to be loaded as `LoadedCurve` objects:
```python
# Each LoadedCurve has:
curve.x_raw               # np.array: X values (e.g., time)
curve.y_raw               # np.array: Y values (e.g., current)
curve.group_flags         # Dict: All grouping keys and values
                          # e.g., {"potential": -1.95, "pH": 1.48}
curve.sample_id           # str: Unique identifier
curve.file_path           # Path: Source file

# Get primary grouping value:
primary_key = "potential"
curve.get_primary_value(primary_key)  # Returns -1.95
```

To adapt for different data types:
- Add a parser beside `pip0_dataloading/cor_file_parser.py` and call it from `pip0_data_loader_orchestrator.py`
- Ensure `x_raw`, `y_raw`, and `group_flags` fields are populated
- Configure grouping keys in settings.json

#### Grouping Mechanism

Curves are grouped by a **configurable primary grouping key** (default: first key in `grouping_keys`) for:
- Summary GPR aggregation (pip3): Combines all curves with same primary key value
- Efficiency evaluation (pip4): Evaluates learning curves per group

The grouping workflow:
1. **Configuration**: Set `primary_grouping_key` in settings.json (e.g., `"potential"`, `"pH"`, `"temperature"`)
2. **Extraction**: Each curve's grouping value is extracted from filename or file content
3. **Rounding**: Numeric values are rounded to `group_round_digits` decimal places
   - Example: `-1.953V` → `-1.95V` with `group_round_digits=2`
4. **Grouping**: Curves with identical rounded values form a group
5. **Processing**: Each group is processed independently through pip3/pip4

**Internal Group Key Format**: When using multiple grouping keys (e.g., pH and potential), groups are identified by a combined key string:
```
"pH=1.48|potential=-1.95"  # Internal format
"pH_1.48_potential_-1.95"  # Filename-safe format
```

To modify grouping for your domain:
```json
// In settings.json:
{
    "pip0_dataloading": {
        "filename_parsing": {
            "grouping_keys": [
                {"name": "temperature", "token_index": 0, "dtype": "float", "regex_extract": "^(\\d+)C$"},
                {"name": "sample_type", "token_index": 1, "dtype": "string"}
            ]
        },
        "primary_grouping_key": "temperature"  // Main dimension for aggregation
    },
    "pip1_datapreprocessing": {
        "grouping": {
            "group_round_digits": 1  // Round to 1 decimal (e.g., 25.3°C → 25.0°C)
        }
    }
}
```

For non-numeric grouping keys (e.g., categorical labels):
- Set `dtype: "string"` in the key definition
- String values are used as-is without rounding

### Code Style Conventions

- Type hints on all function signatures
- Dataclasses for configuration and results
- f-strings for string formatting
- Pathlib for all file operations
- Explicit `__all__` exports in `__init__.py`

---

## License

This project is released under the [MIT License](LICENSE).

Copyright © 2026 Yunkai0825.

---

## Citation

If you use this software, please cite:

> Yunkai Sun, FDA-GPR: Functional Data Analysis with Gaussian Process Regression for Electrochemical Transient Analysis, Argonne National Laboratory, 2024.
