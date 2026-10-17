"""Random choice of the training point in each log10 t bin, and size and range of the prediction grid, of the
individual GPR fits at -1.95 V (154 curves), and their effect on the aggregate of each weighting method.

Inputs
  Pipeline outputs in _output/CCNF_CTAB_PT_output: individual_GPR/Individual_GPR_<curve>.csv (posterior mean and sd on
  the shared grid, scale factor), individual_GPR/Diagnostic_Data_<curve>.csv (training rows = the pipeline training
  set, the middle sample of each non-empty uniform log10 t bin, 500 bins; validation rows = the filtered record,
  x = log10(t + 1e-9) with 8 decimals), individual_GPR/Validation_Results_potential_-1.95.csv (log marginal
  likelihood of each pipeline fit), batch_testing/NS_norm__WM_iterative__WS_curve/summary_gpr/iterative
  (Converged_Weights, Summary_GPR).

Method
  Individual GPR: Constant * Matern(nu = 1.5) + White, StandardScaler on x and y, normalize_y = True, alpha = 0,
  L-BFGS-B with 5 restarts, random_state = 0, one BLAS thread per fit, posterior sd from predict(return_std=True).
  Reference: every curve refitted on its pipeline training set and predicted on the shared grid (500 points,
  log10 t from -1 to 3.69897), on 250 and on 1000 points over the same range, and on 500 points from log10 t = -1 to
  log10 3000.
  Random selection: for seed s = 0, 1, 2, np.random.seed(s) once, then the filtered record of each of the 154 curves
  in Converged_Weights order binned into 500 uniform log10 t bins with one sample drawn per non-empty bin
  (np.random.randint on the global numpy state); refitted and predicted on the shared grid.
  Aggregates: GPR-FDA (curve-wise iterative weights), point-wise iterative weights run to convergence (at most 10000
  iterations), direct average of the observations and direct average of the normalized curves, weights re-iterated
  for every variant. A grid variant is aggregated on its own grid and its mean (current units) is interpolated with
  np.interp in log10 t onto the shared-grid points inside its range (the common points).
  Shift of a method = (variant mean - reference mean of the same method) / band of the reference GPR-FDA aggregate,
  current units, per common grid point; band^2 = sum_r w_r (I_r - mean)^2 + sum_r w_r sd_r^2.

Output tables (eval_scripts/output/downsampling_and_grid/)
  aggregate_shift_by_variant   one row per variant: random_seed (empty for a grid variant), n_grid_points,
                               grid_log10_t_max, n_common_points; gprfda_max, gprfda_rms, pointwise_max, pointwise_rms,
                               direct_obs_max, direct_obs_rms, direct_norm_max, direct_norm_rms (max and rms of the
                               absolute shift over the common points, unitless); pointwise_iterations,
                               pointwise_converged, n_reference, n_variant, notes.
  aggregate_shift_on_grid      t_s, log10_t; GPR-FDA shift per grid point (unitless): seed0_shift_over_band,
                               seed1_shift_over_band, seed2_shift_over_band, grid250_shift_over_band,
                               grid1000_shift_over_band, grid3000s_shift_over_band (empty above 3000 s).
  weights_per_curve            curve, record_end_s (10**max x - 1e-9 of the filtered record), complete (record
                               end >= 2990 s), contact_loss (pipeline weight < 1e-8), weight_pipeline,
                               weight_n_pipeline (pipeline weight x 154), weight_seed0, weight_seed1, weight_seed2,
                               weight_ratio_seed0, weight_ratio_seed1, weight_ratio_seed2 (weight / pipeline weight),
                               notes (curve id).
  weights_by_seed              random_seed, n_eff, n_eff_pipeline (1 / sum w^2), spearman_vs_pipeline (Spearman rank
                               correlation of the curve weights with the pipeline weights).

Run
  python -X utf8 -B eval_scripts/eval_downsampling_and_grid.py [--n-jobs K]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _eval_common as C
import _eval_refit as RF

POT = "-1.95"
SEEDS = (0, 1, 2)
N_BINS = 500
GRIDS = {"grid250": np.linspace(-1.0, 3.69897, 250), "grid1000": np.linspace(-1.0, 3.69897, 1000),
         "grid3000s": np.linspace(-1.0, np.log10(3000.0), 500)}


def pipeline_rows(sid):
    """(x, y) of the training rows and of the validation rows of the Diagnostic_Data file of a curve."""
    d = pd.read_csv(C.diagnostic_path(sid), comment="#")
    dt = d.data_type.astype(str).str.lower()
    tr = d[dt == "training"]
    va = d[dt == "validation"].reset_index(drop=True)
    xt = pd.to_numeric(tr.x.astype(str).str.strip().str.strip("[]")).to_numpy()
    xv = pd.to_numeric(va.x.astype(str).str.strip().str.strip("[]")).to_numpy()
    return xt, tr.y_observed.to_numpy(float), xv, va.y_observed.to_numpy(float)


def random_sets(full, seed):
    """One random sample per non-empty uniform log10 t bin of every record; np.random.seed(seed) once."""
    np.random.seed(seed)
    out = []
    for xv, yv in full:
        dds, _ = RF.downsample_uniform_bins(pd.DataFrame({"x_transformed": xv, "y_transformed": yv}), N_BINS,
                                            "x_transformed", "random")
        out.append((dds.x_transformed.to_numpy(), dds.y_transformed.to_numpy()))
    return out


def build_jobs(sids):
    """Fit jobs (reference on the pipeline training sets, random selections) and the record end of each curve."""
    rows = [pipeline_rows(s) for s in sids]
    tend = np.array([float(10 ** xv.max() - 1e-9) for _, _, xv, _ in rows])
    full = [(xv, yv) for _, _, xv, yv in rows]
    jobs = [dict(x=xt, y=yt, kernel="M1.5", extra_pred=GRIDS, std_from="std", tag=("reference", r))
            for r, (xt, yt, _, _) in enumerate(rows)]
    for s in SEEDS:
        jobs += [dict(x=x, y=y, kernel="M1.5", std_from="std", tag=(f"seed{s}", r))
                 for r, (x, y) in enumerate(random_sets(full, s))]
    return jobs, tend


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="parallel fits (each fit uses one BLAS thread)")
    args = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)
    print("reading the Diagnostic_Data files and drawing the random training sets ...", flush=True)
    jobs, tend = build_jobs(pool.sids)
    print(f"{len(jobs)} fits with {args.n_jobs} workers ...", flush=True)
    res = RF.fit_jobs(jobs, n_jobs=args.n_jobs, blas_threads=1)
    fit = {r.tag: r for r in res}

    def pool_of(job, grid=None):
        pr = [fit[(job, r)] for r in range(R)]
        if grid is None:
            return C.Pool(POT, pool.sids, pool.x, pool.t, np.array([p.mu for p in pr]), np.array([p.sd for p in pr]),
                          pool.F)
        return C.Pool(POT, pool.sids, GRIDS[grid], 10 ** GRIDS[grid] - C.X_SHIFT,
                      np.array([p.pred[grid][0] for p in pr]), np.array([p.pred[grid][1] for p in pr]), pool.F)

    agg_pipeline = C.aggregate_all(pool)
    ref = C.aggregate_all(pool_of("reference"))
    band = ref["gprfda"].band_real
    sm = C.load_summary(POT)
    C.check("pipeline aggregate recomputed from the pipeline posteriors",
            np.abs(agg_pipeline["gprfda"].mean_real / sm.y_real.to_numpy(float) - 1).max() < 1e-12)
    C.check("pipeline weights recomputed from the pipeline posteriors",
            np.abs(agg_pipeline["gprfda"].w - w_pipeline).max() < 1e-12)
    d_ref = np.abs(ref["gprfda"].mean_real - agg_pipeline["gprfda"].mean_real) / agg_pipeline["gprfda"].band_real
    C.check("reference refit reproduces the pipeline GPR-FDA aggregate", d_ref.max() < 1e-5,
            f"max |shift| {d_ref.max():.3g} band")
    vr = C.load_validation_results(POT).set_index("sample_id")
    dlml = np.array([fit[("reference", r)].lml_fit for r in range(R)]) - vr.loc[pool.sids, "lml_fit"].to_numpy(float)
    C.check("reference refit reaches the pipeline log marginal likelihood", np.abs(dlml).max() < 1e-6,
            f"max |dLML| {np.abs(dlml).max():.2g}")

    rows, grid_tab, aggs = [], pd.DataFrame({"t_s": C.T_GRID, "log10_t": C.GRID}), {}
    for s in SEEDS:
        aggs[s] = C.aggregate_all(pool_of(f"seed{s}"))
        rows.append(dict(random_seed=s, n_grid_points=C.N_GRID, grid_log10_t_max=C.GRID[-1], n_common_points=C.N_GRID,
                         **C.shift_summary(ref, aggs[s], band),
                         notes=f"one random sample per log10 t bin, np.random.seed({s})"))
        grid_tab[f"seed{s}_shift_over_band"] = (aggs[s]["gprfda"].mean_real - ref["gprfda"].mean_real) / band
    for g, gx in GRIDS.items():
        ag = C.aggregate_all(pool_of("reference", g))
        sel = (C.GRID >= gx[0] - 1e-12) & (C.GRID <= gx[-1] + 1e-12)
        rows.append(dict(random_seed=np.nan, n_grid_points=len(gx), grid_log10_t_max=gx[-1],
                         n_common_points=int(sel.sum()), **C.shift_summary(ref, ag, band, sel=sel, x_var=gx),
                         notes=f"prediction grid of {len(gx)} points from 10**{gx[0]:g} to 10**{gx[-1]:.5f} s"))
        sh = np.full(C.N_GRID, np.nan)
        sh[sel] = (np.interp(C.GRID[sel], gx, ag["gprfda"].mean_real) - ref["gprfda"].mean_real[sel]) / band[sel]
        grid_tab[f"{g}_shift_over_band"] = sh
    shifts = pd.DataFrame(rows)
    C.check("point-wise weights converged for every variant", bool(shifts.pointwise_converged.all()))

    wt = pd.DataFrame(dict(curve=np.arange(R), record_end_s=tend, complete=tend >= C.T_COMPLETE,
                           contact_loss=w_pipeline < 1e-8, weight_pipeline=w_pipeline,
                           weight_n_pipeline=w_pipeline * R))
    for s in SEEDS:
        wt[f"weight_seed{s}"] = aggs[s]["gprfda"].w
    for s in SEEDS:
        wt[f"weight_ratio_seed{s}"] = aggs[s]["gprfda"].w / w_pipeline
    wt["notes"] = pool.sids
    ws = pd.DataFrame([dict(random_seed=s, n_eff=aggs[s]["gprfda"].n_eff, n_eff_pipeline=C.n_eff(w_pipeline),
                            spearman_vs_pipeline=float(spearmanr(aggs[s]["gprfda"].w, w_pipeline).statistic))
                       for s in SEEDS])

    C.write_tables(__file__, {"aggregate_shift_by_variant": shifts, "aggregate_shift_on_grid": grid_tab,
                              "weights_per_curve": wt, "weights_by_seed": ws})
    print(f"{C.checks_failed()} failed checks; {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
