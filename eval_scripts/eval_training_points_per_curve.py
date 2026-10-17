"""Number of training points per curve (number of uniform log10 t bins of the downsampling) of the individual GPR
fits at -1.95 V (154 curves), and its effect on the individual fits and on the aggregate.

Inputs
  Pipeline outputs in _output/CCNF_CTAB_PT_output: individual_GPR/Individual_GPR_<curve>.csv (posterior mean and sd on
  the shared grid, scale factor), individual_GPR/Diagnostic_Data_<curve>.csv (training rows = the pipeline training
  set, the middle sample of each non-empty uniform log10 t bin, 500 bins; validation rows = the filtered record,
  x = log10(t + 1e-9) with 8 decimals), batch_testing/NS_norm__WM_iterative__WS_curve/summary_gpr/iterative
  (Converged_Weights, Summary_GPR); raw records in _input/CCNF_CTAB_PT (.cor).

Method
  Individual GPR: Constant * Matern(nu = 1.5) + White, StandardScaler on x and y, normalize_y = True, alpha = 0,
  L-BFGS-B with 5 restarts, random_state = 0, posterior sd from predict(return_std=True), prediction on the shared grid
  (500 points, log10 t from -1 to 3.69897); one BLAS thread per fit unless stated.
  Bin counts on the filtered records: the validation rows of each curve binned into B = 62, 125, 250, 1000, 2000
  uniform log10 t bins, middle sample of each non-empty bin, refitted. Reference: every curve refitted on its pipeline
  training set. Aggregates of the four methods (GPR-FDA curve-wise weights, point-wise weights run to convergence with
  at most 10000 iterations, direct average of the observations and of the normalized curves), weights re-iterated per
  bin count; shift = (variant mean - reference mean of the same method) / band of the reference GPR-FDA aggregate,
  current units, per grid point; band^2 = sum_r w_r (I_r - mean)^2 + sum_r w_r sd_r^2.
  Bin counts on the raw records: each raw record preprocessed as in the pipeline (1e-4 <= t <= 5000 s, normalization
  by the mean current of samples 50-90 %, |I / scale| < 0.01 removed, x = log10(t + 1e-9)) and downsampled with
  B = 50, 100, 200, 300, 500, 1000 bins (middle sample per non-empty bin), refitted; GPR-FDA weights re-iterated with
  the scale factors of this preprocessing. Reference: the GPR-FDA aggregate of the pipeline posteriors; shift over its
  band; mean and band of the Summary_GPR file for the deltas.
  Representative record: among the complete records (end >= 2990 s) with pipeline weight > 1e-6, the record of median
  pipeline weight. Its reference fit is the fit of its pipeline training set with three BLAS threads; the bin-count fits
  of the filtered record are compared with it in mA (scale factor f of the record, |f| for the sd).

Output tables (eval_scripts/output/training_points_per_curve/)
  aggregate_shift_by_bin_count          filtered-record bin counts: n_bins, points_median, points_min, points_max
                                        (training points per curve over the 154 curves); gprfda_max, gprfda_rms,
                                        pointwise_max, pointwise_rms, direct_obs_max, direct_obs_rms, direct_norm_max,
                                        direct_norm_rms (max and rms of the absolute shift over the grid, unitless);
                                        pointwise_iterations, pointwise_converged, n_reference, n_variant, notes.
  gprfda_by_bin_count                   raw-record bin counts: n_bins, points_median, points_min, points_max,
                                        gprfda_max, gprfda_rms (absolute shift over the pipeline band), n_eff,
                                        n_eff_pipeline (1 / sum w^2), spearman_vs_pipeline (Spearman rank correlation of
                                        the curve weights with the recomputed pipeline weights).
  gprfda_by_bin_count_on_grid           t_s, log10_t, pipeline_mean_norm, pipeline_band_norm, pipeline_mean_mA,
                                        pipeline_band_mA (Summary_GPR); per raw-record bin count B:
                                        raw_bins<B>_delta_mean_norm, raw_bins<B>_delta_mean_mA (aggregate minus
                                        pipeline aggregate), raw_bins<B>_shift_over_band.
  weights_per_curve                     curve, record_end_s (10**max x - 1e-9 of the filtered record), complete
                                        (record end >= 2990 s), contact_loss (pipeline weight < 1e-8),
                                        weight_pipeline, weight_n_pipeline (pipeline weight x 154),
                                        weight_raw_bins<B> and weight_ratio_raw_bins<B> (weight / pipeline weight)
                                        per raw-record bin count B, notes (curve id).
  representative_record_by_bin_count    t_s (10**log10_t), log10_t, reference_sd_mA; per filtered-record bin count
                                        B: bins<B>_delta_mean_mA ((mean - reference mean) x f), bins<B>_sd_mA;
                                        notes (curve id).
  representative_record_training_points n_bins (500 = pipeline training set), n_training_points, scale_factor_mA (f),
                                        notes (curve id).

Run
  python -X utf8 -B eval_scripts/eval_training_points_per_curve.py [--n-jobs K]
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
FILTERED_BINS = (62, 125, 250, 1000, 2000)
RAW_BINS = (50, 100, 200, 300, 500, 1000)
COST = {"bins2000": 0, "bins1000": 1, "raw1000": 2, "reference": 3, "raw500": 4, "bins250": 5, "raw300": 6,
        "raw200": 7, "bins125": 8, "raw100": 9, "bins62": 10, "raw50": 11}


def pipeline_rows(sid):
    """(x, y) of the training rows and of the validation rows of the Diagnostic_Data file of a curve."""
    d = pd.read_csv(C.diagnostic_path(sid), comment="#")
    dt = d.data_type.astype(str).str.lower()
    tr = d[dt == "training"]
    va = d[dt == "validation"].reset_index(drop=True)
    xt = pd.to_numeric(tr.x.astype(str).str.strip().str.strip("[]")).to_numpy()
    xv = pd.to_numeric(va.x.astype(str).str.strip().str.strip("[]")).to_numpy()
    return xt, tr.y_observed.to_numpy(float), xv, va.y_observed.to_numpy(float)


def representative_record(w, tend):
    """Position of the complete record (end >= 2990 s, pipeline weight > 1e-6) of median pipeline weight."""
    ok = (w > 1e-6) & (tend >= C.T_COMPLETE)
    comp = pd.DataFrame({"pos": np.flatnonzero(ok), "w": w[ok]}).sort_values("w")
    return int(comp.pos.iloc[len(comp) // 2])


def build_jobs(sids):
    """Fit jobs of the reference, the filtered-record and the raw-record bin counts, with per-curve training-point
    counts, raw-record scale factors, record ends and the pipeline training sets."""
    jobs, n_pts, scale, tend, train = [], {}, {b: np.zeros(len(sids)) for b in RAW_BINS}, np.zeros(len(sids)), []
    for r, sid in enumerate(sids):
        xt, yt, xv, yv = pipeline_rows(sid)
        train.append((xt, yt))
        tend[r] = float(10 ** xv.max() - 1e-9)
        jobs.append(dict(x=xt, y=yt, kernel="M1.5", std_from="std", tag=("reference", r)))
        df = pd.DataFrame({"x_transformed": xv, "y_transformed": yv})
        for b in FILTERED_BINS:
            dds, _ = RF.downsample_uniform_bins(df, b, "x_transformed", "middle")
            x, y = dds.x_transformed.to_numpy(), dds.y_transformed.to_numpy()
            n_pts[(f"bins{b}", r)] = len(y)
            jobs.append(dict(x=x, y=y, kernel="M1.5", std_from="std", tag=(f"bins{b}", r)))
        raw, _ = C.load_record(sid)
        for b in RAW_BINS:
            pre, why = RF.preprocess_record(raw, max_points=b)
            if pre is None:
                raise RuntimeError(f"{sid}: {why}")
            n_pts[(f"raw{b}", r)] = pre["n_train"]
            scale[b][r] = pre["scale"]
            jobs.append(dict(x=pre["x_train"], y=pre["y_train"], kernel="M1.5", std_from="std", tag=(f"raw{b}", r)))
    jobs.sort(key=lambda j: COST[j["tag"][0]])
    return jobs, n_pts, scale, tend, train


def counts(n_pts, job, R):
    """Training points per curve of one job type."""
    return np.array([n_pts[(job, r)] for r in range(R)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="parallel fits (each fit uses one BLAS thread)")
    args = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    w_file = C.load_weights(POT).converged_weight.to_numpy(float)
    print("reading the Diagnostic_Data files and the raw records ...", flush=True)
    jobs, n_pts, scale, tend, train = build_jobs(pool.sids)
    rex = representative_record(w_file, tend)
    print(f"{len(jobs)} fits with {args.n_jobs} workers ({time.time() - t0:.0f} s) ...", flush=True)
    res = RF.fit_jobs(jobs, n_jobs=args.n_jobs, blas_threads=1)
    fit = {r.tag: r for r in res}
    xt, yt = train[rex]
    rec_ref = RF.fit_jobs([dict(x=xt, y=yt, kernel="M1.5", std_from="std", tag=("record_reference", rex))],
                          n_jobs=1, blas_threads=3)[0]

    def arrays(job):
        return (np.array([fit[(job, r)].mu for r in range(R)]), np.array([fit[(job, r)].sd for r in range(R)]))

    agg_st = C.aggregate_all(pool)
    a0 = agg_st["gprfda"]
    sm = C.load_summary(POT)
    C.check("pipeline aggregate recomputed from the pipeline posteriors",
            np.abs(a0.mean_real / sm.y_real.to_numpy(float) - 1).max() < 1e-12)
    C.check("pipeline weights recomputed from the pipeline posteriors", np.abs(a0.w - w_file).max() < 1e-12)

    Yr, Sr = arrays("reference")
    ref = C.aggregate_all(C.Pool(POT, pool.sids, pool.x, pool.t, Yr, Sr, pool.F))
    band = ref["gprfda"].band_real
    d_ref = np.abs(ref["gprfda"].mean_real - a0.mean_real) / a0.band_real
    C.check("reference refit reproduces the pipeline GPR-FDA aggregate", d_ref.max() < 1e-5,
            f"max |shift| {d_ref.max():.3g} band")
    rows = []
    for b in FILTERED_BINS:
        Y, S = arrays(f"bins{b}")
        var = C.aggregate_all(C.Pool(POT, pool.sids, pool.x, pool.t, Y, S, pool.F))
        nb = counts(n_pts, f"bins{b}", R)
        rows.append(dict(n_bins=b, points_median=float(np.median(nb)), points_min=int(nb.min()),
                         points_max=int(nb.max()), **C.shift_summary(ref, var, band),
                         notes=f"{b} uniform log10 t bins of the filtered record, middle sample per bin"))
    shifts = pd.DataFrame(rows)
    C.check("point-wise weights converged for every bin count", bool(shifts.pointwise_converged.all()))

    n500 = counts(n_pts, "raw500", R)
    C.check("raw-record preprocessing with 500 bins gives the pipeline training-set sizes and scale factors",
            bool(np.array_equal(n500, [len(t[1]) for t in train])
                 and np.allclose(scale[500], pool.F, rtol=1e-12, atol=0)),
            f"max relative scale factor deviation {np.max(np.abs(scale[500] / pool.F - 1)):.1e}")
    y0n, s0n = sm.y_normalised.to_numpy(float), sm.Std_normalised.to_numpy(float)
    grid_tab = pd.DataFrame({"t_s": C.T_GRID, "log10_t": C.GRID, "pipeline_mean_norm": y0n, "pipeline_band_norm": s0n,
                             "pipeline_mean_mA": 1e3 * sm.y_real.to_numpy(float),
                             "pipeline_band_mA": 1e3 * sm.Std_real.to_numpy(float)})
    wt = pd.DataFrame(dict(curve=np.arange(R), record_end_s=tend, complete=tend >= C.T_COMPLETE,
                           contact_loss=w_file < 1e-8, weight_pipeline=w_file, weight_n_pipeline=w_file * R))
    summ = []
    for b in RAW_BINS:
        Y, S = arrays(f"raw{b}")
        a = C.aggregate_arrays(Y, S, scale[b], "gprfda")
        sh = (a.mean_real - a0.mean_real) / a0.band_real
        grid_tab[f"raw_bins{b}_delta_mean_norm"] = a.mean_norm - y0n
        grid_tab[f"raw_bins{b}_delta_mean_mA"] = 1e3 * a.mean_real - 1e3 * sm.y_real.to_numpy(float)
        grid_tab[f"raw_bins{b}_shift_over_band"] = sh
        wt[f"weight_raw_bins{b}"] = a.w
        nb = counts(n_pts, f"raw{b}", R)
        summ.append(dict(n_bins=b, points_median=float(np.median(nb)), points_min=int(nb.min()),
                         points_max=int(nb.max()), gprfda_max=float(np.abs(sh).max()),
                         gprfda_rms=float(np.sqrt(np.mean(sh ** 2))), n_eff=a.n_eff, n_eff_pipeline=a0.n_eff,
                         spearman_vs_pipeline=float(spearmanr(a.w, a0.w).statistic)))
    for b in RAW_BINS:
        wt[f"weight_ratio_raw_bins{b}"] = wt[f"weight_raw_bins{b}"] / w_file
    wt["notes"] = pool.sids
    summ = pd.DataFrame(summ)
    C.check("raw-record refit with 500 bins reproduces the pipeline GPR-FDA aggregate",
            float(summ.loc[summ.n_bins == 500, "gprfda_max"].iloc[0]) < 1e-5)

    sid = pool.sids[rex]
    f = 1e3 * scale[500][rex]
    af = abs(f)
    mu0, sd0 = rec_ref.mu, rec_ref.sd
    rec = pd.DataFrame({"t_s": C.T10_GRID, "log10_t": C.GRID, "reference_sd_mA": sd0 * af})
    pts = [dict(n_bins=500, n_training_points=len(yt), scale_factor_mA=f, notes=sid)]
    for b in FILTERED_BINS:
        p = fit[(f"bins{b}", rex)]
        rec[f"bins{b}_delta_mean_mA"] = (p.mu - mu0) * f
        rec[f"bins{b}_sd_mA"] = p.sd * af
        pts.append(dict(n_bins=b, n_training_points=p.n_train, scale_factor_mA=f, notes=sid))
    rec["notes"] = sid

    C.write_tables(__file__, {"aggregate_shift_by_bin_count": shifts, "gprfda_by_bin_count": summ,
                              "gprfda_by_bin_count_on_grid": grid_tab, "weights_per_curve": wt,
                              "representative_record_by_bin_count": rec,
                              "representative_record_training_points": pd.DataFrame(pts)})
    print(f"{C.checks_failed()} failed checks; {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
