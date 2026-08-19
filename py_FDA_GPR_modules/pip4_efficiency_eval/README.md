# pip4_efficiency_eval - Data Efficiency Evaluation Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module evaluates data efficiency via Monte-Carlo learning curve analysis:
- Determines how many curves are needed to reproduce full-data summary GPR
- Runs Monte-Carlo sampling across subset sizes
- Evaluates iterative, FGPR, or Student-t aggregation through `aggregation_method`
- Computes error metrics comparing subsets to the method-matched full-data reference
- Generates learning-curve, convergence, and between-curve-variance diagnostics

---

## Module Structure

```
pip4_efficiency_eval/
├── __init__.py                        # Public API exports
├── pip4_efficiency_eval_orchestrator.py # Main orchestrator (EfficiencyOrchestrator)
├── efficiency_config.py               # Configuration (DirParams, GlobalParams, ScaleParams, PlotParams)
├── efficiency_core.py                 # Core algorithms + re-exports (process_potential_learning_curve)
├── learning_curve.py                  # Learning curve computation (learning_curve, fast_summary_gpr_core, SubsetResult, LearningCurveResult)
├── mc_sampling.py                     # Monte Carlo sampling (mc_repeats, balanced_subset, generate_balanced_subsets)
├── efficiency_plotting.py             # Visualization utilities
├── DEBUG_efficiency_eval.py           # Debug/test script
├── DEBUG_pip4_Efficiency_Output/      # Debug output folder
└── README.md                          # This file
```

---

## Quick Start

```python
from pathlib import Path
from py_FDA_GPR_modules.pip4_efficiency_eval import (
    EfficiencyOrchestrator,
    DirParams,
)
from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import (
    SummaryGPRConfig,
    SummaryGPRHyperParams,
)

dirpara = DirParams(indiv_dir=Path("output/individual_gprs"))
summary_cfg = SummaryGPRConfig(input_directory=dirpara.indiv_dir)
summary_hp = SummaryGPRHyperParams()

orchestrator = EfficiencyOrchestrator(
    dirpara=dirpara,
    summary_gpr_config=summary_cfg,
    summary_gpr_hyperparams=summary_hp,
    aggregation_method="iterative",  # or "fgpr" / "student_t"
)
results = orchestrator.process_all()
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `EfficiencyOrchestrator` | `pip4_efficiency_eval_orchestrator.py` | Main orchestrator |
| `DirParams` | `efficiency_config.py` | I/O directory configuration |
| `GlobalParams` | `efficiency_config.py` | Global experiment parameters |
| `ScaleParams` | `efficiency_config.py` | Axis scale / error transformation |
| `PlotParams` | `efficiency_config.py` | Plot configuration |
| `SubsetResult` | `learning_curve.py` | Result from single subset run |
| `LearningCurveResult` | `learning_curve.py` | Complete learning curve result |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `process_potential_learning_curve(...)` | `efficiency_core.py` | Process one potential group |
| `learning_curve(gpr_pool, ref, sizes, ...)` | `learning_curve.py` | MC learning curve computation |
| `fast_summary_gpr_core(gpr_pool, ...)` | `learning_curve.py` | Quick summary GPR for subsets |
| `error_metric(ref, cand, kind, ...)` | `learning_curve.py` | Compute RMSE/MAE/max error |
| `summarize_learning_curve(df, ...)` | `learning_curve.py` | Create wide summary CSV |
| `mc_repeats(n_total, m, globpara)` | `mc_sampling.py` | Determine MC repeat count |
| `balanced_subset(idx, occ, m, rng)` | `mc_sampling.py` | Balanced sampling |
| `generate_balanced_subsets(...)` | `mc_sampling.py` | Generate multiple subsets |

---

## File Details

### `pip4_efficiency_eval_orchestrator.py`

High-level orchestrator for efficiency evaluation.

**Classes:**
- `EfficiencyOrchestrator`: Coordinates loading, processing, plotting
  - `process_group(group_key, gpr_list)` → `LearningCurveResult`
  - `process_all()` → `Dict[str, LearningCurveResult]`
  - `aggregation_method`: `"iterative"`, `"fgpr"`, or `"student_t"`

### `efficiency_config.py`

Configuration dataclasses.

**Classes:**
- `DirParams`: I/O paths
  - `indiv_dir`: Input directory with individual GPR CSVs
  - `output_dir`: Output directory

- `GlobalParams`: Experiment knobs
  - `metric`: "rmse", "mae", or "max"
  - `base_repeats`: MC repeats when combinatorics large
  - `max_enum`: Threshold for exhaustive enumeration
  - `q_low`, `q_high`: Quantiles for IQR
  - `random_seed`

- `ScaleParams`: Axis/error scaling
  - `use_log_error`, `log_base_error`
  - `normalize_w_rbar`: Normalize error by reference average
  - `use_log_cost`, `log_base_cost`

- `PlotParams`: Plot settings
  - `figsize`, `dpi`, `xlabel`, `ylabel`

### `efficiency_core.py`

Core algorithms and re-exports for backward compatibility.

**Functions:**
- `process_potential_learning_curve(gpr_list, summary_gpr_config, ...)`:
  Main processing for one potential group
  1. Computes full-data reference
  2. Defines subset sizes (every 2 curves from 2 to N)
  3. Runs learning curve analysis

**Re-exports:** All from `mc_sampling.py` and `learning_curve.py`

### `learning_curve.py`

Learning curve computation.

**Classes:**
- `SubsetResult`: Single subset run result
  - `y_mean`, `y_std_real`, `y_std_norm`, `y_bar`
  - `n_iterations`, `elapsed_time`

- `LearningCurveResult`: Complete result
  - `summary`: Aggregated stats per subset_size (mean, median, IQR)
  - `detailed`: Individual MC runs with sample_indices
  - `cov_matrices`, `sigma_btw_pointwise_arrays`: Optional method diagnostics

**Functions:**
- `fast_summary_gpr_core(gpr_pool, ...)`: Run summary GPR on subset
- `error_metric(ref, ave_ref, cand, ave_bar, kind, scapara)`: Compute distance metric
- `learning_curve(gpr_pool, ref_mean, ref_bar, sizes, ...)`: Full MC analysis
- `learning_curve_layered(...)`: Layer-wise execution with resumable detailed CSV output
- `summarize_learning_curve(df, csv_stem, out_dir)`: Wide summary export

### `mc_sampling.py`

Monte Carlo sampling utilities.

**Functions:**
- `bounded_comb(n, k, cap)`: Capped combinatorial
- `mc_repeats(n_total, m, globpara)`: Determine repeat count
- `balanced_subset(idx, occ, m, rng)`: Select indices with minimal occurrence
- `generate_balanced_subsets(n_total, subset_size, n_repeats, seed)`: Generate subsets

### `efficiency_plotting.py`

Visualization utilities.

**Functions:**
- `aggregate_detailed_to_summary(df_detailed, q_low, q_high)`: Aggregate MC runs
- `plot_learning_curve(df_summary, group_key, output_path, ...)`: 4-panel plot
- `plot_learning_curve_from_detailed(df_detailed, ...)`: Load detailed, plot
- `plot_iteration_statistics_from_detailed(df_detailed, ...)`: Iteration stats plot
- `plot_sigma_btw_comparison(...)`, `export_sigma_btw_csv(...)`: Between-curve variance diagnostics
- `plot_covariance_heatmaps(...)`, `plot_covariance_diagonal(...)`: Covariance diagnostics

---

## Output Files

For each potential:
- `LearningCurve_{group}_summary.csv` - Transposed aggregate statistics
- `LearningCurve_{group}_detailed.csv` - Individual MC runs, selected indices, uncertainty, and between-curve variance
- `Efficiency_LearningCurve_{group}.png` - 4-panel learning curve plot
- `Iteration_Statistics_{potential}.png` - Iteration statistics
- `SigmaBtw_{method}_{group}.png` and `sigma_btw_{method}_{group}.csv` - Between-curve variance diagnostics
- Optional `cov_matrices/` and `sigma_btw_pointwise/` artifacts when supported by the aggregation method

---

## Algorithm

1. **Reference**: Run summary GPR on all N curves
2. **Subset sizes**: [2, 4, 6, ..., N]
3. **For each size m**:
   - Determine MC repeats: min(C(N,m), base_repeats)
   - For each repeat:
     - Sample m curves (balanced)
     - Run summary GPR
     - Compute error vs reference
   - Aggregate: mean, median, IQR, variance
4. **Output**: Learning curves showing error vs #curves
