# pip3_FDA_scoring_and_aggregations - Summary GPR Aggregation Module

**Author:** Yunkai Sun (C-STEEL, CSE, ANL)

## Overview

This module aggregates individual GPR models to create summary GPR curves:
- Loads individual GPR predictions with metadata from CSV files
- Reconstructs `ScalingInfo` objects from metadata
- Groups curves by common key (e.g., potential)
- Interpolates all curves to a common X grid
- Supports iterative weighted-sum, operator fusion, functional GPR (FGPR), and robust Student-t aggregation
- Supports scalar or structured FGPR between-curve covariance
- Produces aggregate and predictive uncertainty in real and normalized scales

---

## Module Structure

```
pip3_FDA_scoring_and_aggregations/
├── __init__.py                        # Public API exports
├── pip3_summary_gpr_orchestrator.py   # Main orchestrator (SummaryGPROrchestrator, SummaryGPRResult)
├── summary_gpr_config.py              # Configuration (SummaryGPRConfig, SummaryGPRHyperParams)
├── summary_gpr_loader.py              # Load GPRs with metadata (IndividualGPRData, load_all_individual_gprs, group_gprs_by_key)
├── summary_gpr_plotting.py            # Visualization utilities
├── comparison_orchestrator.py         # Cross-method comparisons
├── iterative_weight_sum_GPR/          # Method 1: Iterative variance-based weighted sum
│   ├── __init__.py
│   ├── iterative_gpr_core.py        # compute_summary_gpr, SummaryGPRCoreResult
│   ├── iterative_orchestrator.py    # Run/export/plot orchestration
│   ├── iterative_export_helpers.py
│   ├── iterative_plot_helpers.py
│   ├── variance_calculation_helpers/
│   │   ├── __init__.py
│   │   ├── within_variance.py         # Within-model variance computation
│   │   └── between_variance.py        # Between-model/total variance helpers
│   └── weight_calculation_helpers/
│       ├── __init__.py
│       ├── curvewise_weight_helpers.py # Curve-level weight optimization
│       └── pointwise_weight_helpers.py # Pointwise weight optimization
├── operator_fusion_noweight/          # Method 2: Precision-space operator fusion
│   ├── __init__.py
│   ├── operator_fusion_weight_helpers.py # Operator fusion + EB between-variance
│   ├── operator_fusion_orchestrator.py
│   ├── operator_fusion_export_helpers.py
│   └── operator_fusion_plot_helpers.py
├── functional_GPR/                    # Method 3: Functional GPR (FGPR)
│   ├── __init__.py
│   ├── fgpr_helpers.py                # Scalar/structured full-covariance aggregation
│   ├── fgpr_structured_btw.py         # Structured between-curve covariance
│   ├── fgpr_orchestrator.py
│   ├── fgpr_export_helpers.py
│   └── fgpr_plot_helpers.py
├── student_t_agg_iterative/           # Method 4: Robust Student-t aggregation
│   ├── __init__.py
│   ├── student_t_core.py
│   ├── student_t_orchestrator.py
│   ├── student_t_export_helpers.py
│   └── student_t_plot_helpers.py
├── DEBUG_summary_gpr.py               # Debug/test script
├── DEBUG_pip3_Summary_GPR_Output/     # Debug output folder
└── README.md                          # This file
```

---

## Quick Start

```python
from pathlib import Path
from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import (
    SummaryGPRConfig,
    SummaryGPRHyperParams,
    SummaryGPROrchestrator,
)

cfg = SummaryGPRConfig(
    input_directory=Path("output/Individual_GPRs/"),
    weight_mode="iterative",
    weight_scope="curve",
    normalization_summary=True,
)

hp = SummaryGPRHyperParams(convergence_tol=1e-6, confidence_level=0.75)

orchestrator = SummaryGPROrchestrator(cfg, hp, verbose=True)
results = orchestrator.process_all()

for group_key, result in results.items():
    print(f"{group_key}: {result.n_curves} curves")
```

---

## Public API

### Classes

| Class | File | Description |
|-------|------|-------------|
| `SummaryGPROrchestrator` | `pip3_summary_gpr_orchestrator.py` | Main orchestrator |
| `SummaryGPRResult` | `pip3_summary_gpr_orchestrator.py` | Result for one group |
| `SummaryGPRConfig` | `summary_gpr_config.py` | Aggregation configuration |
| `SummaryGPRHyperParams` | `summary_gpr_config.py` | Algorithm hyperparameters |
| `IndividualGPRData` | `summary_gpr_loader.py` | Container for loaded individual GPR |
| `SummaryGPRCoreResult` | `iterative_weight_sum_GPR/iterative_gpr_core.py` | Iterative algorithm result |
| `CurvewiseWeightResult` | `iterative_weight_sum_GPR/weight_calculation_helpers/curvewise_weight_helpers.py` | Curve-level weight optimization result |
| `PointwiseWeightResult` | `iterative_weight_sum_GPR/weight_calculation_helpers/pointwise_weight_helpers.py` | Pointwise weight optimization result |
| `OperatorFusionResult` | `operator_fusion_noweight/operator_fusion_weight_helpers.py` | Operator fusion weight result |
| `FGPRResult` | `functional_GPR/fgpr_helpers.py` | Functional GPR aggregation result |
| `StructuredBtwConfig` | `functional_GPR/fgpr_structured_btw.py` | Structured between-curve covariance controls |
| `StudentTResult` | `student_t_agg_iterative/student_t_core.py` | Robust Student-t aggregation result |

### Functions

| Function | File | Description |
|----------|------|-------------|
| `compute_summary_gpr(...)` | `iterative_weight_sum_GPR/iterative_gpr_core.py` | Iterative aggregation algorithm |
| `load_all_individual_gprs(dir, pattern)` | `summary_gpr_loader.py` | Load all GPR CSVs |
| `group_gprs_by_key(gprs, key_attr)` | `summary_gpr_loader.py` | Group by attribute |
| `compute_within_model_variances(S)` | `iterative_weight_sum_GPR/variance_calculation_helpers/within_variance.py` | Within-model variance |
| `compute_between_model_variances(y, y_mean)` | `iterative_weight_sum_GPR/variance_calculation_helpers/between_variance.py` | Between-model variance |
| `compute_weighted_mean(y, weights)` | `iterative_weight_sum_GPR/variance_calculation_helpers/between_variance.py` | Weighted average |
| `iterative_weight_optimization(...)` | `iterative_weight_sum_GPR/weight_calculation_helpers/curvewise_weight_helpers.py` | Curve-level optimization |
| `pointwise_weight_optimization(...)` | `iterative_weight_sum_GPR/weight_calculation_helpers/pointwise_weight_helpers.py` | Point-level optimization |
| `compute_operator_fusion(...)` | `operator_fusion_noweight/operator_fusion_weight_helpers.py` | Precision-space fusion |
| `compute_fgpr(...)` | `functional_GPR/fgpr_helpers.py` | Functional GPR aggregation |
| `fit_sigma_btw(...)` | `functional_GPR/fgpr_helpers.py` | Profile-likelihood σ_btw² optimization |
| `compute_fgpr_structured(...)` | `functional_GPR/fgpr_helpers.py` | FGPR with structured between-curve covariance |
| `compute_student_t_aggregation(...)` | `student_t_agg_iterative/student_t_core.py` | Robust EM/IRLS-style aggregation |

---

## File Details

### `pip3_summary_gpr_orchestrator.py`

High-level orchestrator for Summary GPR processing.

**Classes:**
- `SummaryGPROrchestrator`: Ties together loading, interpolation, aggregation, export
  - `process_group(group_key, gprs)` → `SummaryGPRResult`
  - `process_all()` → `Dict[str, SummaryGPRResult]`
  
- `SummaryGPRResult`: Result for one group
  - `x_pred_transformed`, `x_pred_original`
  - `y_mean`, `y_mean_norm`
  - `y_std_real`, `y_std_norm` (dual variance outputs)
  - `weights`, `weight_history`, `curve_history`
  - `y_scaling`, `x_scaling`

### `summary_gpr_config.py`

Configuration dataclasses.

**Classes:**
- `SummaryGPRConfig`: Aggregation settings
  - `input_directory`, `output_directory`, `file_pattern`
  - `weight_mode`: "equal" or "iterative"
  - `weight_scope`: "curve" or "point"
  - `variance_aggregation_scale`: "real" or "normalised"
  - `include_within_variance`, `include_between_variance`
  - `normalization_summary`, `plot_individual_gprs`, `process_groups`
  - `enable_operator_fusion`, `enable_fgpr`, `enable_student_t`

- `SummaryGPRHyperParams`: Algorithm parameters
  - `max_iterations`, `convergence_tol`, `epsilon`
  - `confidence_level`, `num_interp_points`
  - `fgpr_min_scale_factor_ratio`, `fgpr_structured_btw`
  - `student_t_nu`, `student_t_optimize_nu`, `student_t_nu_bounds`
  - `student_t_max_iterations`, `student_t_convergence_tol`

### `summary_gpr_loader.py`

Load individual GPR CSV files with metadata parsing.

**Classes:**
- `IndividualGPRData`: Container for loaded individual GPR
  - `sample_id`, `index_id`, `group_key`, `group_flags`
  - `x_pred_transformed`, `x_pred_original`, `y_pred`, `y_std`
  - `x_scaling`, `y_scaling` (reconstructed from metadata)
  - `y_pred_normalized`, `y_std_normalized` (computed properties)
  - `to_arrays(gpr_list)`: Convert list to arrays for compute_summary_gpr

**Functions:**
- `load_all_individual_gprs(directory, pattern)`: Load all matching CSVs
- `group_gprs_by_key(gprs, key_attr)`: Group by attribute (e.g., "group_flags.potential")

### `iterative_weight_sum_GPR/iterative_gpr_core.py`

Core Summary GPR aggregation function.

**Classes:**
- `SummaryGPRCoreResult`: Core algorithm output
  - `x_pred`, `y_mean`, `y_mean_norm`
  - `y_std_real`, `y_std_norm`
  - `weights`, `weight_history`, `curve_history`
  - `y_scaling`, `n_models`, `n_points`

**Functions:**
- `compute_summary_gpr(y_array, S_array, x_pred, y_scalings, ...)`:
  Main aggregation combining variance scoring and weight optimization

### `iterative_weight_sum_GPR/variance_calculation_helpers/`

Variance computation helpers for the iterative weighted-sum method.

**Functions:**
- `compute_within_model_variances(S_array)`: Average squared std per model
- `compute_between_model_variances(y_array, y_mean)`: Deviation from consensus
- `compute_total_model_variance(...)`: Combined variance for weighting
- `compute_pointwise_variance(...)`: Per-point variance
- `compute_weighted_mean(y_array, weights)`: Weighted average prediction
- `compute_final_variance(...)`: Final aggregated variance

### `iterative_weight_sum_GPR/weight_calculation_helpers/`

Weight refinement algorithms for the iterative weighted-sum method.

**Classes:**
- `CurvewiseWeightResult`: Curve-level optimization output
- `PointwiseWeightResult`: Point-level optimization output

**Functions:**
- `iterative_weight_optimization(y_array, S_within, ...)`: Curve-level weights
- `pointwise_weight_optimization(y_array, S_array, ...)`: Point-level weights
- `compute_weights_equal(n_models)`: Uniform weights

### `operator_fusion_noweight/`

Precision-space operator fusion aggregation.

**Classes:**
- `OperatorFusionResult`: Operator fusion output

**Functions:**
- `compute_operator_fusion(...)`: Iterative Mahalanobis-weighted precision fusion
- `_estimate_between_variance_eb(...)`: Empirical Bayes grid search for σ_btw²

### `functional_GPR/`

Functional Gaussian Process Regression (FGPR) aggregation.

**Classes:**
- `FGPRResult`: Full posterior aggregation result with predictive covariance

**Functions:**
- `compute_fgpr(...)`: FGPR aggregation with optimal σ_btw²
- `fit_sigma_btw(...)`: Profile NLL minimization for between-curve variance
- `compute_fgpr_structured(...)`: FGPR using white, smooth, offset, drift, and scale covariance components selected by `StructuredBtwConfig`

### `student_t_agg_iterative/`

Robust Student-t aggregation that iteratively estimates curve weights,
between-curve variance, and optionally the degrees of freedom.

**Classes:**
- `StudentTResult`: Aggregate/predictive covariance, weights, energies, and convergence histories
- `StudentTOrchestrator`: Run/export/plot orchestration

**Functions:**
- `compute_student_t_aggregation(...)`: Robust Student-t aggregation

### `summary_gpr_plotting.py`

Visualization utilities with core functions + CSV wrappers.

**Functions:**
- `plot_summary_gpr(x, y_mean, y_lower, y_upper, ...)`: Core plotting
- `plot_summary_gpr_from_csv(csv_path, ...)`: Load and plot
- `plot_weight_distribution_from_csv(...)`, `plot_weight_convergence_from_csv(...)`

---

## Iterative Weight Optimization Algorithm

Iterative optimization minimizes total variance:

1. Initialize: $w_i = 1/n$
2. Compute weighted mean: $\bar{y}(t) = \sum_i w_i y_i(t)$
3. Between-model variance: $D_i = \frac{1}{n_t} \sum_t (y_i(t) - \bar{y}(t))^2$
4. Within-model variance: $\sigma_i^2 = \frac{1}{n_t} \sum_t s_i(t)^2$
5. Update weights: $w_i \propto 1/(D_i + \sigma_i^2 + \epsilon)$
6. Normalize and repeat until convergence

---

## Output Files

Outputs are separated by aggregation method beneath the configured pip3
output directory:

```text
summary_gpr/
├── iterative/   # Summary_GPR_*, Converged_Weights_*, histories and plots
├── operator_fusion/ # OperatorFusion_Curve_*, weights and histories
├── fgpr/        # FGPR_Curve_*, covariance, diagnostics, histories and plots
└── student_t/   # StudentT_Curve_*, covariance, diagnostics, histories and plots
```

Each method writes real- and normalized-scale aggregate uncertainty. FGPR
and Student-t additionally export aggregate and predictive covariance files.
