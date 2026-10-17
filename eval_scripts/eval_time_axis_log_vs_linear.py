"""Time axis of the individual GPR fits at -1.95 V (154 curves): x = log10(t + 1e-9) of the pipeline against a linear
time axis x = t, for every curve and its effect on the GPR-FDA aggregate, and for one representative record.

Inputs
  Pipeline outputs in _output/CCNF_CTAB_PT_output: individual_GPR/Individual_GPR_<curve>.csv (posterior mean and sd on
  the shared grid, scale factor), individual_GPR/Diagnostic_Data_<curve>.csv of the representative record (training
  rows = the pipeline training set; validation rows = the filtered record, x with 8 decimals),
  batch_testing/NS_norm__WM_iterative__WS_curve/summary_gpr/iterative (Converged_Weights, Summary_GPR); raw records
  in _input/CCNF_CTAB_PT (.cor).

Method
  Each raw record is preprocessed as in the pipeline (1e-4 <= t <= 5000 s, normalization by the mean current of
  samples 50-90 %, |I / scale| < 0.01 removed) but with x = t (s): 500 uniform bins in t, middle sample of each
  non-empty bin, plus the first and last samples of the filtered record (edge points). Individual GPR:
  Constant * Matern(nu = 1.5) + White, StandardScaler on x and y, normalize_y = True, alpha = 0, L-BFGS-B with
  5 restarts, random_state = 0, one BLAS thread per fit, posterior sd from predict(return_std=True), prediction at
  the 500 shared grid times t = 10**log10_t - 1e-9 s.
  Aggregate: GPR-FDA curve-wise weights re-iterated on the linear-t posteriors with the scale factors of this
  preprocessing; shift = (aggregate mean - pipeline aggregate mean) / band of the GPR-FDA aggregate of the pipeline
  posteriors, current units, per grid point; mean and band of the Summary_GPR file for the deltas.
  Representative record: among the complete records (end >= 2990 s) with pipeline weight > 1e-6, the record of median
  pipeline weight. Its log10 t fit is the fit of its pipeline training set with three BLAS threads on the shared grid;
  both fits in mA with the scale factor f of the record (|f| for the sd); its measured points are the validation rows
  of its Diagnostic_Data file.

Output tables (eval_scripts/output/time_axis_log_vs_linear/)
  gprfda_linear_time                    n_bins, edge_points, points_median, points_min, points_max (training points
                                        per curve), gprfda_max, gprfda_rms (absolute shift over the pipeline band),
                                        n_eff, n_eff_pipeline (1 / sum w^2), spearman_vs_pipeline (Spearman rank
                                        correlation of the curve weights with the recomputed pipeline weights).
  gprfda_linear_time_on_grid            t_s, log10_t, pipeline_mean_norm, pipeline_band_norm, pipeline_mean_mA,
                                        pipeline_band_mA (Summary_GPR), linear_t_delta_mean_norm, linear_t_delta_mean_mA
                                        (aggregate minus pipeline aggregate), linear_t_shift_over_band.
  weights_per_curve                     curve, record_end_s (last time of the capped raw record), complete
                                        (record end >= 2990 s), contact_loss (pipeline weight < 1e-8),
                                        weight_pipeline, weight_n_pipeline (pipeline weight x 154), weight_linear_t,
                                        weight_ratio_linear_t (weight / pipeline weight), n_training_points, notes
                                        (curve id).
  representative_record_on_grid         t_s (10**log10_t), log10_t, log_t_mean_mA, log_t_sd_mA, linear_t_mean_mA,
                                        linear_t_sd_mA, linear_minus_log_t_mean_mA, notes (curve id).
  representative_record_points          t_s (10**x), log10_t (x), current_mA, residual_mA (current minus the log10 t
                                        mean interpolated in log10 t).
  representative_record_training_points linear_time_axis (0 = log10 t, pipeline training set; 1 = t), n_bins,
                                        edge_points, n_training_points, first_training_t_s, scale_factor_mA (f),
                                        notes (curve id).

Run
  python -X utf8 -B eval_scripts/eval_time_axis_log_vs_linear.py [--n-jobs K]
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
N_BINS = 500


def representative_record(w, tend):
    """Position of the complete record (end >= 2990 s, pipeline weight > 1e-6) of median pipeline weight."""
    ok = (w > 1e-6) & (tend >= C.T_COMPLETE)
    comp = pd.DataFrame({"pos": np.flatnonzero(ok), "w": w[ok]}).sort_values("w")
    return int(comp.pos.iloc[len(comp) // 2])


def record_rows(sid):
    """Training rows (x, y) and the validation rows (x, y) of the Diagnostic_Data file of a curve."""
    d = pd.read_csv(C.diagnostic_path(sid), comment="#")
    tr = d[d.data_type.astype(str).str.lower() == "training"]
    xt = pd.to_numeric(tr.x.astype(str).str.strip().str.strip("[]")).to_numpy()
    v = d.copy()
    for c in ("x", "y_observed"):
        v[c] = pd.to_numeric(v[c].astype(str).str.strip().str.strip("[]"), errors="coerce")
    v = v.dropna(subset=["x", "y_observed"])
    v = v[v.data_type.astype(str).str.lower().str.startswith("valid")]
    return xt, tr.y_observed.to_numpy(float), v.x.to_numpy(), v.y_observed.to_numpy()


def build_jobs(sids):
    """Linear-t fit jobs of all curves and their preprocessing results (scale, training points, record end)."""
    jobs, info = [], []
    for r, sid in enumerate(sids):
        raw, _ = C.load_record(sid)
        pre, why = RF.preprocess_record(raw, max_points=N_BINS, x_axis="linear", edge_points=True)
        if pre is None:
            raise RuntimeError(f"{sid}: {why}")
        info.append(dict(scale=pre["scale"], n_train=pre["n_train"], t_first=float(pre["t_train"].min()),
                         tend=pre["tmax_s"]))
        jobs.append(dict(x=pre["x_train"], y=pre["y_train"], kernel="M1.5", x_pred=C.T_GRID, std_from="std",
                         tag=("linear", r)))
    return jobs, pd.DataFrame(info)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="parallel fits (each fit uses one BLAS thread)")
    args = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)
    print("preprocessing the raw records ...", flush=True)
    jobs, info = build_jobs(pool.sids)
    tend = info.tend.to_numpy()
    rex = representative_record(w_pipeline, tend)
    sid = pool.sids[rex]
    xt, yt, xv, yv = record_rows(sid)
    print(f"{len(jobs)} fits with {args.n_jobs} workers ({time.time() - t0:.0f} s) ...", flush=True)
    res = RF.fit_jobs(jobs, n_jobs=args.n_jobs, blas_threads=1)
    fit = {r.tag: r for r in res}
    rec_ref = RF.fit_jobs([dict(x=xt, y=yt, kernel="M1.5", std_from="std", tag=("record_reference", rex))],
                          n_jobs=1, blas_threads=3)[0]

    a0 = C.aggregate(pool, "gprfda")
    sm = C.load_summary(POT)
    C.check("pipeline aggregate recomputed from the pipeline posteriors",
            np.abs(a0.mean_real / sm.y_real.to_numpy(float) - 1).max() < 1e-12)
    C.check("pipeline weights recomputed from the pipeline posteriors", np.abs(a0.w - w_pipeline).max() < 1e-12)
    sc = info.scale.to_numpy()
    C.check("scale factors of the preprocessing equal the pipeline scale factors",
            np.allclose(sc, pool.F, rtol=1e-12, atol=0), f"max relative {np.max(np.abs(sc / pool.F - 1)):.1e}")
    Y = np.array([fit[("linear", r)].mu for r in range(R)])
    S = np.array([fit[("linear", r)].sd for r in range(R)])
    a = C.aggregate_arrays(Y, S, sc, "gprfda")
    sh = (a.mean_real - a0.mean_real) / a0.band_real
    y0n = sm.y_normalised.to_numpy(float)
    grid_tab = pd.DataFrame({"t_s": C.T_GRID, "log10_t": C.GRID, "pipeline_mean_norm": y0n,
                             "pipeline_band_norm": sm.Std_normalised.to_numpy(float),
                             "pipeline_mean_mA": 1e3 * sm.y_real.to_numpy(float),
                             "pipeline_band_mA": 1e3 * sm.Std_real.to_numpy(float),
                             "linear_t_delta_mean_norm": a.mean_norm - y0n,
                             "linear_t_delta_mean_mA": 1e3 * a.mean_real - 1e3 * sm.y_real.to_numpy(float),
                             "linear_t_shift_over_band": sh})
    nb = info.n_train.to_numpy()
    summ = pd.DataFrame([dict(n_bins=N_BINS, edge_points=1, points_median=float(np.median(nb)),
                              points_min=int(nb.min()), points_max=int(nb.max()), gprfda_max=float(np.abs(sh).max()),
                              gprfda_rms=float(np.sqrt(np.mean(sh ** 2))), n_eff=a.n_eff, n_eff_pipeline=a0.n_eff,
                              spearman_vs_pipeline=float(spearmanr(a.w, a0.w).statistic))])
    wt = pd.DataFrame(dict(curve=np.arange(R), record_end_s=tend, complete=tend >= C.T_COMPLETE,
                           contact_loss=w_pipeline < 1e-8, weight_pipeline=w_pipeline,
                           weight_n_pipeline=w_pipeline * R, weight_linear_t=a.w,
                           weight_ratio_linear_t=a.w / w_pipeline, n_training_points=nb, notes=pool.sids))

    f = 1e3 * sc[rex]
    af = abs(f)
    lin = fit[("linear", rex)]
    il, ll = rec_ref.mu * f, lin.mu * f
    rec = pd.DataFrame({"t_s": C.T10_GRID, "log10_t": C.GRID, "log_t_mean_mA": il, "log_t_sd_mA": rec_ref.sd * af,
                        "linear_t_mean_mA": ll, "linear_t_sd_mA": lin.sd * af, "linear_minus_log_t_mean_mA": ll - il,
                        "notes": sid})
    f_mA = 1e3 * pool.F[rex]
    pts = pd.DataFrame({"t_s": 10 ** xv, "log10_t": xv, "current_mA": yv * f_mA,
                        "residual_mA": (yv - np.interp(xv, C.GRID, rec_ref.mu)) * f_mA})
    tps = pd.DataFrame([dict(linear_time_axis=0, n_bins=N_BINS, edge_points=0, n_training_points=len(yt),
                             first_training_t_s=float(10 ** xt.min() - 1e-9), scale_factor_mA=f, notes=sid),
                        dict(linear_time_axis=1, n_bins=N_BINS, edge_points=1, n_training_points=lin.n_train,
                             first_training_t_s=float(info.t_first[rex]), scale_factor_mA=f, notes=sid)])

    C.write_tables(__file__, {"gprfda_linear_time": summ, "gprfda_linear_time_on_grid": grid_tab,
                              "weights_per_curve": wt, "representative_record_on_grid": rec,
                              "representative_record_points": pts, "representative_record_training_points": tps})
    print(f"{C.checks_failed()} failed checks; {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
