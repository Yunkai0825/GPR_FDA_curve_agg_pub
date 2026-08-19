# pip5_batch_setting_grid_testing - Batch Grid Testing Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module runs Summary GPR and efficiency evaluation across all testing option combinations:
- Builds six non-redundant iterative combinations plus optional FGPR and Student-t pairs (up to ten total)
- Runs pip3 (Summary GPR) and pip4 (Efficiency) for each combination
- Generates comparison plots overlaying results from all permutations
- Exports aggregate CSVs for cross-permutation analysis

---

## Module Structure

```
pip5_batch_setting_grid_testing/
├── __init__.py                            # Public API exports
├── pip5_batch_grid_settings_orchestrator.py # Main orchestrator (BatchTestingOrchestrator)
├── batch_config.py                        # Configuration (BatchDirParams, BatchTestingOptions, BatchRunConfig)
├── batch_core.py                          # Core utilities (build_testing_grid, discover_groupkeys, apply_options_to_config)
├── batch_comparison_plotting.py           # Comparison plots across permutations
├── DEBUG_batch_testing.py                 # Debug/test script
├── DEBUG_pip5_Batch_Setting_Output/       # Debug output folder
└── README.md                              # This file
```

---

## Quick Start

```python
from pathlib import Path
from py_FDA_GPR_modules.pip5_batch_setting_grid_testing import (
    BatchTestingOrchestrator,
    BatchDirParams,
)

# Option 1: From input folder (auto-detect settings)
orchestrator = BatchTestingOrchestrator.from_settings(
    input_folder="_input/CCNF_CTAB_PT"
)
results = orchestrator.run_all()

# Option 2: Manual configuration
dir_params = BatchDirParams(input_dir=Path("output/individual_gprs"))
orchestrator = BatchTestingOrchestrator(dir_params)
results = orchestrator.run_all()
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `BatchTestingOrchestrator` | `pip5_batch_grid_settings_orchestrator.py` | Main orchestrator |
| `BatchDirParams` | `batch_config.py` | Directory configuration |
| `BatchTestingOptions` | `batch_config.py` | Single option combination |
| `BatchRunConfig` | `batch_config.py` | What to run in each iteration |
| `BatchComparisonConfig` | `batch_comparison_plotting.py` | Comparison plot configuration |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `build_testing_grid(include_fgpr, include_student_t)` | `batch_core.py` | Build 6–10 non-redundant combinations |
| `discover_groupkeys(input_dir)` | `batch_core.py` | Find potentials from filenames |
| `copy_artifacts(src, dst)` | `batch_core.py` | Copy generated outputs |
| `apply_options_to_config(options, cfg)` | `batch_core.py` | Apply options to SummaryGPRConfig |
| `plot_all_comparisons(...)` | `batch_comparison_plotting.py` | Generate all comparison plots |
| `export_all_aggregate_csvs(...)` | `batch_comparison_plotting.py` | Export cross-permutation CSVs |

---

## File Details

### `pip5_batch_grid_settings_orchestrator.py`

High-level orchestrator for batch grid testing.

**Classes:**
- `BatchTestingOrchestrator`: Runs all permutations
  - `from_settings(input_folder=...)`: Create from input folder
  - `from_settings(settings_path=...)`: Create from settings JSON
  - `run_single_combo(options)` → `Dict[str, Any]`
  - `run_all()` → `Dict[str, Dict]`, including configured comparison plots and aggregate CSV post-processing

### `batch_config.py`

Configuration dataclasses.

**Classes:**
- `BatchDirParams`: Directory parameters
  - `input_dir`: Directory with individual GPR CSVs
  - `base_output_dir`: Base for batch outputs
  - `get_combo_output_dir(tag)`: Get output dir for combination

- `BatchTestingOptions`: Single combination
  - `normalization_summary`: True/False
  - `weight_mode`: "equal"/"iterative"
  - `weight_scope`: "curve"/"point"
  - `aggregation_method`: "iterative"/"fgpr"/"student_t"
  - `tag` property: Folder name such as "NS_norm__WM_iterative__WS_curve" or "NS_norm__AM_student_t"

- `BatchRunConfig`: What to run
  - `run_summary_gpr`, `run_efficiency_eval`
  - `run_combine_gprs`, `run_comparison_plots`
  - `export_summary_csvs`, `copy_artifacts`

**Constants:**
- `ARTIFACT_PATTERNS`: List of file patterns to copy

### `batch_core.py`

Core utilities for batch testing.

**Functions:**
- `build_testing_grid(include_fgpr=False, include_student_t=False)` → `List[BatchTestingOptions]`:
  Returns six baseline combinations and, when enabled, two FGPR and/or two Student-t combinations
  
- `discover_groupkeys(input_dir, pattern)` → `List[str]`:
  Extract potential values from filenames
  
- `copy_artifacts(source_dir, dest_dir, patterns)` → `int`:
  Copy generated files
  
- `apply_options_to_config(options, summary_gpr_config)`:
  Create new config with options applied

### `batch_comparison_plotting.py`

Comparison plots across permutations.

**Classes:**
- `BatchComparisonConfig`: Plot settings
  - `summary_figsize`, `efficiency_figsize`
  - `summary_ci_alpha`, `efficiency_fill_alpha`
  - Legend, DPI, axis labels

**Functions:**
- `discover_permutations(batch_output_dir)`: Find permutation folders
- `discover_groupkeys(batch_output_dir, perm)`: Find potentials in permutation
- `load_summary_gpr_data(batch_output_dir, perm, potential)`: Load summary CSV
- `load_efficiency_data(...)`: Load learning curve CSV
- `plot_summary_gpr_comparison(...)`: Overlay summary curves from all permutations
- `plot_efficiency_comparison(...)`: Overlay learning curves
- `plot_all_comparisons(batch_output_dir, ...)`: Generate all plots
- `export_all_aggregate_csvs(...)`: Export combined CSVs

---

## Testing Grid

The six baseline combinations are:

| # | normalization_summary | weight_mode | weight_scope | Tag |
|---|----------------------|-------------|--------------|-----|
| 1 | True | equal | curve | NS_norm__WM_equal__WS_curve |
| 2 | True | iterative | curve | NS_norm__WM_iterative__WS_curve |
| 3 | True | iterative | point | NS_norm__WM_iterative__WS_point |
| 4 | False | equal | curve | NS_real__WM_equal__WS_curve |
| 5 | False | iterative | curve | NS_real__WM_iterative__WS_curve |
| 6 | False | iterative | point | NS_real__WM_iterative__WS_point |

Note: (weight_mode=equal, weight_scope=point) is excluded as redundant.

Optional combinations:

| # | normalization_summary | aggregation_method | Tag |
|---|----------------------|--------------------|-----|
| 7 | True | fgpr | NS_norm__AM_fgpr |
| 8 | False | fgpr | NS_real__AM_fgpr |
| 9 | True | student_t | NS_norm__AM_student_t |
| 10 | False | student_t | NS_real__AM_student_t |

Build all ten explicitly with:

```python
from py_FDA_GPR_modules.pip5_batch_setting_grid_testing import build_testing_grid

testing_grid = build_testing_grid(include_fgpr=True, include_student_t=True)
```

The main pipeline reads `pip5_batch_testing.include_fgpr` and
`pip5_batch_testing.include_student_t` from the settings JSON. Direct use of
`BatchTestingOrchestrator.from_settings(...)` uses the six-item default unless
a `testing_grid` is supplied.

---

## Output Structure

```
{base_output_dir}/
├── NS_norm__WM_equal__WS_curve/
│   ├── summary_gpr/
│   │   ├── Summary_GPR_potential_*.csv
│   │   └── Summary_GPR_potential_*.png
│   └── learning_curve/
│       ├── LearningCurve_*_summary.csv
│       └── LearningCurve_*.png
├── NS_norm__WM_iterative__WS_curve/
│   └── ...
├── NS_norm__AM_fgpr/
├── NS_norm__AM_student_t/
├── ... (other enabled permutations)
├── comparisons/
│   ├── SummaryGPR_Comparison_potential_*.png
│   ├── Efficiency_Comparison_potential_*.png
│   ├── Combined_Comparison_potential_*.png
│   └── Aggregate_*.csv
└── settings_used.json
```

---

## Workflow

1. **Initialize**: Load settings, build testing grid
2. **For each permutation**:
   - Apply options to SummaryGPRConfig
   - Run Summary GPR (pip3)
   - Run Efficiency Evaluation (pip4)
   - Save outputs to permutation subfolder
3. **After all permutations**:
   - Generate comparison plots overlaying all permutations
   - Export aggregate CSVs for cross-permutation analysis
