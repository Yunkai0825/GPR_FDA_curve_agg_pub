# eval_scripts

Evaluation tests of the GPR-FDA aggregation of the CCNF_CTAB_PT chronoamperometry records. Each script runs one type of test. The scripts read only the released inputs (`_input/CCNF_CTAB_PT`) and the pipeline outputs (`_output/CCNF_CTAB_PT_output`) of this repository, and write their result tables as CSV files to `output/<test>/`.

`eval_hyperparameters.py` also loads `gpr_config.py` and `gpr_functions.py` (`hyperparams_to_physical`, `kernel_to_physical`) from `py_FDA_GPR_modules/pip2_individual_gpr`. No other pipeline module is used.

## Requirements

| Package | Version |
|---|---|
| Python | 3.13.5 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.4.2 |
| threadpoolctl | 3.5.0 |
| matplotlib | 3.10.0 |

## Running

From the repository root:

```bash
python -X utf8 -B eval_scripts/eval_band_coverage.py
```

Each script prints its self-checks and exits with code 1 when a check fails. Each run rewrites the tables of its test.

`--n-jobs K` sets the number of worker processes (default 3) in `eval_band_decomposition.py`, `eval_short_record_reinclusion.py`, `eval_weights_vs_metadata.py`, `eval_downsampling_and_grid.py`, `eval_time_axis_log_vs_linear.py` and `eval_training_points_per_curve.py` (GPR fits), and in `eval_residual_diagnostics.py` (reading the curves). The results do not depend on it. `eval_kernel_choice.py` and `eval_normalization.py` use fixed worker counts.

The number of BLAS threads per GPR fit changes the round-off of the Cholesky factorizations, and on a flat likelihood this can move the optimizer to another optimum. `_eval_refit.py` therefore runs every fit with a fixed thread count (threadpoolctl), set by each script: 1 for the fits run in worker processes, 3 for the single-record fits and the fits of `eval_normalization.py`, 6 for the likelihood fits and 3 for the grid fits of `eval_kernel_choice.py`.

The fit times in `eval_kernel_choice.py` are wall-clock measurements. Run that script alone on an otherwise idle machine.

## Tests

| Script | Test | Output tables |
|---|---|---|
| `eval_hyperparameters.py` | Fitted hyperparameters of the individual GPR fits in GPR data units, fit errors, bound check | `hyperparameters_per_curve`, `hyperparameter_summary`, `bound_check` |
| `eval_residual_diagnostics.py` | Residuals of the individual fits by log-time window: interval coverage, standardized residuals, lag-1 autocorrelation | `records`, `residuals_by_window`, `residuals_window_summary`, `posterior_sigma_by_window`, `posterior_sigma_window_summary`, `whole_record_lag1`, `whole_record_lag1_summary` |
| `eval_weight_convergence.py` | Convergence of curve-wise (GPR-FDA) and point-wise iterative weights | `rms_weight_change`, `iterations_to_tolerance`, `gprfda_iterates`, `pointwise_iterates`, `converged_aggregates`, `individual_posteriors` |
| `eval_band_decomposition.py` | Between-curve and within-curve parts of the band variance by record group | `band_terms_on_grid`, `band_share_window_medians`, `curve_groups` |
| `eval_band_coverage.py` | Coverage of the curves by the band, weighted and unweighted, in sample and leave one batch out | `coverage_by_grid_point`, `coverage_summary`, `headcount_by_potential`, `calibration_band`, `calibration_deviation` |
| `eval_short_record_reinclusion.py` | Records shorter than 60 s fitted and added to the pool | `aggregate_on_grid`, `reincluded_posterior`, `weights_by_curve`, `reincluded_records`, `reinclusion_summary` |
| `eval_kernel_choice.py` | Matérn ν = 0.5, 1.5, 2.5 and RBF kernels: likelihood, fit time, aggregate | `lml_per_curve`, `lml_summary`, `aggregate_by_kernel`, `shift_by_kernel`, `representative_record`, `representative_posterior`, `representative_points` |
| `eval_training_points_per_curve.py` | Number of uniform log10 t bins (training points) per curve | `aggregate_shift_by_bin_count`, `gprfda_by_bin_count`, `gprfda_by_bin_count_on_grid`, `weights_per_curve`, `representative_record_by_bin_count`, `representative_record_training_points` |
| `eval_time_axis_log_vs_linear.py` | log10 t against linear t as the GPR abscissa | `gprfda_linear_time`, `gprfda_linear_time_on_grid`, `weights_per_curve`, `representative_record_on_grid`, `representative_record_points`, `representative_record_training_points` |
| `eval_downsampling_and_grid.py` | Random against middle-point downsampling; range and size of the prediction grid | `aggregate_shift_by_variant`, `aggregate_shift_on_grid`, `weights_by_seed`, `weights_per_curve` |
| `eval_normalization.py` | Sample window of the scale factor; aggregation in current units without normalization | `window_per_curve`, `representative_record`, `window_shift_on_grid`, `window_summary`, `aggregate_shift_by_window`, `current_units_on_grid`, `current_units_per_curve`, `current_units_summary` |
| `eval_controlled_contamination.py` | Flagged records removed, and copies of recorded curves added | `record_classes`, `aggregate_with_contact_loss_copies`, `shift_by_contact_loss_copies`, `copy_weight_by_contact_loss_copies`, `clean_start_by_contact_loss_copies`, `shift_by_removal_and_outlier_copies`, `clean_start_after_removal`, `shift_by_contamination_check` |
| `eval_data_subsets.py` | Batches, Au records, highest-weight and best initial-stage curves left out | `subset_selection`, `reference_aggregate`, `aggregate_by_subset`, `gprfda_shift_by_subset`, `shift_by_subset` |
| `eval_weights_vs_metadata.py` | Converged weights against test chronology, batch, substrate and record length | `records_timeline`, `sittings`, `weights_by_curve`, `batch_summary`, `period_class_groups`, `weights_summary` |

The module docstring of each script states the method, the settings, and the columns and units of every output table.

## Output format

`output/<test>/<table>.csv`, where `<test>` is the script name without `eval_`:

- the first row is the header and every following row is data; nothing precedes or follows the table;
- comma separated, UTF-8, no index column;
- every column is numeric (flags are 0/1, infinity is `inf`, an empty cell is a value that is not defined for that row); text such as sample ids and variant labels is only in the last column, `notes`;
- column names are lower-case snake_case (except `mA`); a unit token closes the quantity it belongs to, as in `record_end_s` or `max_abs_shift_mA_after_10s`.

Units:

| Token | Quantity |
|---|---|
| `t_s` | time in s; on the prediction grid, t_s is 10^log10_t or, depending on the table, 10^log10_t − 1e-9 s (Summary_GPR `x_real`) |
| `_s` | other times and durations in s |
| `_h`, `_d` | durations in hours and days |
| `log10_t` | x = log10(t + 1e-9), the GPR abscissa |
| `_mA` | current in mA (the records' A cm⁻² on the nominal 1 cm² area × 1e3) |
| `_norm` | normalized current, I / s_r with s_r the scale factor of the record |
| `_norm2` | (normalized current)² |
| `_decades` | length in decades of log10 t |
| `_pct` | percent |
| `_v` | potential in V |
| `_over_band` | ratio to the GPR-FDA ±1σ band at the same grid point |

Shift-summary tables (`shift_by_*`, `aggregate_shift_by_*`) have one row per variant. The columns are:

1. the variant parameter(s);
2. `<method>_max` and `<method>_rms` for the methods `gprfda`, `pointwise`, `direct_obs` and `direct_norm`: the maximum and RMS over the grid of |variant mean − reference mean| / band;
3. `pointwise_iterations` and `pointwise_converged`;
4. `n_reference` and `n_variant`;
5. `notes`.

## Methods and settings

- **Individual GPR:** time t in [1e-4, 5000] s and x = log10(t + 1e-9). The scale factor s_r is the mean current over samples int(0.5 n) to int(0.9 n) − 1 of the time-sorted record. Points with |I/s_r| < 0.01 are dropped. At most 500 training points, the middle sample of each uniform bin in x. StandardScaler on x and y, `normalize_y=True`, `alpha=0`. Kernel C(1, (1e-2, 1e4)) × Matérn(1, (1e-2, 1e3), ν = 1.5) + White(1, (1e-12, 10)). L-BFGS-B with 5 restarts, `random_state=0`. A test that varies one of these settings states it in its docstring.
- **Aggregation:** curve-wise iterative weights (GPR-FDA) and point-wise iterative weights to an RMS weight change below 1e-6 (ε = 1e-12; point-wise at most 10000 iterations), and the direct averages of the currents and of the normalized currents. The pipeline point-wise outputs stop after 50 iterations; the self-checks against them use the same cap. The band is the square root of the weighted between-curve variance plus the weighted posterior variance.
- **Grid:** 500 points, log10 t = linspace(−1, 3.69897, 500).

## Shared modules

- `_eval_common.py` (numpy, pandas):
  - paths and loaders of the pipeline outputs;
  - the `.cor` parser and record metadata;
  - one aggregator per method, each reproducing the pipeline Summary_GPR, Converged_Weights and Weight_History;
  - band decomposition, score subsets and shift summaries;
  - self-checks and the CSV writer.
- `_eval_refit.py` (scikit-learn, joblib, threadpoolctl): preprocessing and downsampling of the raw records, the kernels, the individual GPR fit and the parallel map with a fixed BLAS thread count.
