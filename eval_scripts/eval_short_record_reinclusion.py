"""Re-inclusion of the records shorter than 60 s at -1.95 V: the records that the 60 s screening removed are fitted
with the pipeline GPR, appended to the 154 pipeline posteriors, and the curve-wise weights and the aggregate are
computed again for the enlarged pool.

Records
    Every .cor file of _input/CCNF_CTAB_PT with -1.95 V in its name, in the order of the start time in its header.
    Retained records are those of the pipeline run (Converged_Weights); the others are the removed records, each with a
    raw time range below 60 s (self-check). Record end = largest sample time of the raw record.

Fits
    Each removed record goes through the pipeline preprocessing with the 60 s rule switched off: samples sorted by T,
    1e-4 <= T <= 5000 s, x = log10(T + 1e-9), y = I / s with s the mean current of the samples int(0.5 n) ..
    int(0.9 n) - 1, |y| < 0.01 dropped, at most 500 training points (middle sample of each uniform bin in x). GPR:
    StandardScaler on x and y, ConstantKernel(1, (1e-2, 1e4)) * Matern(1, (1e-2, 1e3), nu = 1.5) +
    WhiteKernel(1, (1e-12, 10)), alpha 0, normalize_y, L-BFGS-B with 5 restarts, random_state 0, posterior mean and
    sd (predict with return_std) on the 500-point grid x = linspace(-1, 3.69897, 500). One BLAS thread per fit; the fits
    run in --n-jobs joblib workers (the results do not depend on it).

Aggregation
    Curve-wise weights on the normalized posteriors of the 154 pipeline curves (pipeline pool) and of the enlarged pool
    (retained curves first, then the removed records in start-time order): s2w_r = mean_j S_r(j)^2; w = 1/R; repeat
    ybar = sum_r w_r Y_r, D_r = mean_j (Y_r - ybar)^2, w_new = 1/(D_r + s2w_r + 1e-12) normalized to 1,
    delta = ||w_new - w|| / sqrt(R), w = w_new, until delta < 1e-6. Normalized mean ybar = sum_r w_r Y_r; mean =
    ybar s_agg with s_agg = sum_r w_r s_r; band^2 = sum_r w_r (Y_r s_r - mean)^2 + sum_r w_r (S_r s_r)^2 (current
    units), sum_r w_r (Y_r - ybar)^2 + sum_r w_r S_r^2 (normalized). Shift = (mean_all - mean_pipeline) /
    band_pipeline.

Self-checks
    The pipeline pool aggregate reproduces Converged_Weights and Summary_GPR. Twelve retained records spread over the
    record length (positions int(linspace(0, n - 1, 12)) of the retained records sorted by record end) are rebuilt
    from the raw records: training sets and scale factors equal the pipeline ones, and their refits reproduce the
    pipeline posteriors and log marginal likelihoods.

Output tables (eval_scripts/output/short_record_reinclusion/)
    aggregate_on_grid       log10_t, t_s (= 10^log10_t), pipeline_mean_mA, pipeline_band_mA, mean_mA, band_mA,
                            shift_mA (mean_mA - pipeline_mean_mA), shift_over_band, band_ratio (band / pipeline band),
                            pipeline_mean_norm, pipeline_band_norm, mean_norm, band_norm
    reincluded_posterior    curve (position in the enlarged pool), log10_t, t_s, mean_norm, sd_norm, mean_mA
                            (mean_norm x s, A cm^-2 x 1e3), measured (1 when log10_t <= log10(t_end + 1e-9) + 1e-4,
                            t_end = last sample time within 1e-4 <= T <= 5000 s); notes = file
    weights_by_curve        curve, reincluded (0/1), record_end_s, complete (record end >= 2990 s), contact_loss
                            (pipeline weight < 1e-8), substrate_au (0/1), scale_factor_mA, weight_pipeline, weight,
                            weight_n_pipeline (w_r x 154), weight_n (w_r x pool size);
                            notes = file
    reincluded_records      curve, record_end_s, n_samples, n_capped, n_removed_threshold, n_train, scale_factor_mA,
                            mean_sq_sd_norm2 (mean over the grid of S_r^2), mean_sq_dev_norm2 (mean over the grid
                            of (Y_r - normalized mean)^2), weight, weight_n; notes = file
    reinclusion_summary     one row: n_pipeline, n_reincluded, n_pool, max_abs_shift_mA, t_max_abs_shift_mA_s,
                            rms_shift_mA, max_abs_shift_over_band, t_max_abs_shift_s, rms_shift_over_band,
                            max_abs_shift_over_band_to_2990s, max_abs_shift_norm_over_band_norm, max_rel_change_mean,
                            weight_reincluded, weight_reincluded_uniform, max_weight_n_reincluded,
                            median_weight_n_reincluded, n_reincluded_weight_n_above_01, n_reincluded_weight_n_below_1e3,
                            median_weight_n_truncated, median_weight_n_complete (retained curves, enlarged pool),
                            band_ratio_median, band_ratio_max_abs_dev, s_agg_rel_change, max_weight_pipeline,
                            max_weight, n_eff_pipeline, n_eff, iterations_pipeline, iterations,
                            spearman_weights_retained, median_mean_sq_sd_norm2_reincluded,
                            median_mean_sq_sd_norm2_retained, median_mean_sq_dev_norm2_reincluded,
                            median_mean_sq_dev_norm2_retained

Run
    python -X utf8 -B eval_scripts/eval_short_record_reinclusion.py [--n-jobs K]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _eval_common as C
import _eval_refit as RF

POT = "-1.95"
N_VAL = 12
X_COL = "T (Seconds)"
HDR = re.compile(r"Date:\s*(\d{2})-(\d{2})-(\d{4})\s+Time:\s*(\d{2}):(\d{2}):(\d{2})")
TG = C.T10_GRID


def header_start(path):
    """Start date and time in the header of a .cor file."""
    with open(path, "rb") as fh:
        txt = fh.read(6000).decode("latin-1")
    m = HDR.search(txt)
    return datetime(int(m[3]), int(m[1]), int(m[2]), int(m[4]), int(m[5]), int(m[6]))


def records_by_start(pot):
    """Files of one potential (file name) sorted by header start time -> list of file names."""
    idx = C.raw_index()
    files = sorted(f for f in idx if C.record_meta(f)["potential_V"] == float(pot))
    start = {f: header_start(idx[f][0]) for f in files}
    return sorted(files, key=lambda f: start[f])


def pipeline_lml(sid):
    """Log marginal likelihood of the pipeline fit (Individual_GPR header)."""
    hdr = C.read_header(C.IND / f"Individual_GPR_{sid}.csv")
    return float(json.loads(hdr["hyperparams"])["log_marginal_likelihood"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="joblib workers for the GPR fits")
    a = ap.parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    st = C.load_summary(POT)
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)

    ref = C.aggregate(pool, "gprfda")
    w_ref, nit_ref = ref.w, ref.iterations
    d_w = float(np.abs(w_ref - w_pipeline).max())
    d_m = float(np.max(np.abs(ref.mean_real - st.y_real.to_numpy(float)) / np.abs(st.y_real.to_numpy(float))))
    d_b = float(np.max(np.abs(ref.band_real - st.Std_real.to_numpy(float)) / st.Std_real.to_numpy(float)))
    C.check("pipeline pool aggregate = Converged_Weights and Summary_GPR", d_w < 1e-12 and d_m < 1e-10 and d_b < 1e-10,
            f"weights {d_w:.1e}; mean {d_m:.1e}; band {d_b:.1e} (relative); {nit_ref} iterations")

    files = records_by_start(POT)
    in_pool = {C.split_sid(s)[0]: i for i, s in enumerate(pool.sids)}
    tmax, raw_df = {}, {}
    for f in files:
        df, _ = C.load_record(C.find_raw(f))
        tmax[f] = float(df[X_COL].max())
        raw_df[f] = df
    retained = [f for f in files if f in in_pool]
    removed = [f for f in files if f not in in_pool]
    rng = np.array([float(raw_df[f][X_COL].max() - raw_df[f][X_COL].min()) for f in removed])
    C.check("retained records = pipeline pool; removed records shorter than 60 s",
            len(retained) == R and len(removed) > 0 and (rng < C.T_MIN_RECORD).all(),
            f"{len(files)} files, {len(retained)} retained, {len(removed)} removed; raw range of the removed "
            f"{rng.min():.2f}-{rng.max():.2f} s")
    print(f"{len(files)} raw records read in {time.time() - t0:.0f} s", flush=True)

    pre = {}
    for f in removed:
        pre[f], why = RF.preprocess_record(raw_df[f], apply_min_range=False)
        if pre[f] is None:
            C.check(f"preprocessing of {f}", False, why)
    removed = [f for f in removed if pre[f] is not None]

    ret = pd.DataFrame({"file": retained, "tmax_s": [tmax[f] for f in retained]}).sort_values("tmax_s")
    pick = ret.iloc[np.unique(np.linspace(0, len(ret) - 1, N_VAL).astype(int))].file.tolist()
    val = []
    for f in pick:
        p, _ = RF.preprocess_record(raw_df[f], apply_min_range=True)
        sid = pool.sids[in_pool[f]]
        xs, ys = C.pipeline_training_set(sid)
        same_n = len(xs) == p["n_train"]
        val.append(dict(file=f, r=in_pool[f], pre=p, same_n=same_n,
                        dy=float(np.max(np.abs(p["y_train"] / ys - 1))) if same_n else np.inf,
                        dx=float(np.abs(p["x_train"] - xs).max()) if same_n else np.inf,
                        ds=abs(p["scale"] / pool.F[in_pool[f]] - 1), lml=pipeline_lml(sid)))

    jobs = [dict(x=v["pre"]["x_train"], y=v["pre"]["y_train"], std_from="std") for v in val]
    jobs += [dict(x=pre[f]["x_train"], y=pre[f]["y_train"], std_from="std") for f in removed]
    tf = time.time()
    fits = RF.fit_jobs(jobs, n_jobs=a.n_jobs, verbose=0, blas_threads=1)
    print(f"{len(fits)} GPR fits in {time.time() - tf:.0f} s", flush=True)
    fv, fx = fits[:len(val)], fits[len(val):]

    dmu = np.array([np.abs(f.mu - pool.Y[v["r"]]).max() for v, f in zip(val, fv)])
    dsd = np.array([np.abs(f.sd - pool.S[v["r"]]).max() for v, f in zip(val, fv)])
    dl = np.array([abs(f.lml_fit - v["lml"]) for v, f in zip(val, fv)])
    C.check("retained records rebuilt from the raw records = pipeline training sets and scale factors",
            all(v["same_n"] for v in val) and max(v["dy"] for v in val) < 1e-12 and max(v["dx"] for v in val) < 1e-12
            and max(v["ds"] for v in val) < 1e-12,
            f"{len(val)} records; max relative |dy| {max(v['dy'] for v in val):.1e}; max |dx| "
            f"{max(v['dx'] for v in val):.1e}; max relative |ds| {max(v['ds'] for v in val):.1e}")
    C.check("refits of the rebuilt training sets reproduce the pipeline posteriors",
            dmu.max() < 1e-6 and dsd.max() < 1e-6 and dl.max() < 1e-6,
            f"max |d mean| {dmu.max():.1e} (median {np.median(dmu):.1e}); max |d sd| {dsd.max():.1e}; "
            f"max |d LML| {dl.max():.1e}")

    Yx = np.array([f.mu for f in fx])
    Sx = np.array([f.sd for f in fx])
    Fx = np.array([pre[f]["scale"] for f in removed])
    Ya, Sa, Fa = np.vstack([pool.Y, Yx]), np.vstack([pool.S, Sx]), np.concatenate([pool.F, Fx])
    N = len(Ya)
    n_x = len(removed)
    allc = C.aggregate_arrays(Ya, Sa, Fa, "gprfda")
    w_all, nit_all = allc.w, allc.iterations
    C.check("curve-wise weights of the enlarged pool converged", allc.converged, f"{nit_all} iterations")
    wx = w_all[R:]
    w_ex = float(wx.sum())
    sh = (allc.mean_real - ref.mean_real) / ref.band_real
    shn = (allc.mean_norm - ref.mean_norm) / ref.band_norm
    shm = 1e3 * allc.mean_real - 1e3 * ref.mean_real
    br = allc.band_real / ref.band_real
    jm, jmm = int(np.argmax(np.abs(sh))), int(np.argmax(np.abs(shm)))
    s2w = np.mean(Sa ** 2, axis=1)
    dev = np.mean((Ya - allc.mean_norm) ** 2, axis=1)

    rec_end = np.array([tmax[C.split_sid(s)[0]] for s in pool.sids] + [tmax[f] for f in removed])
    complete = rec_end >= C.T_COMPLETE
    cl = np.r_[w_pipeline < 1e-8, np.zeros(n_x, bool)]
    reinc = np.r_[np.zeros(R, bool), np.ones(n_x, bool)]
    names = [C.split_sid(s)[0] for s in pool.sids] + removed
    au = np.array([C.record_meta(f)["substrate"] == "Au" for f in names])
    wn = w_all * N

    agg = pd.DataFrame({"log10_t": C.GRID, "t_s": TG, "pipeline_mean_mA": 1e3 * ref.mean_real,
                        "pipeline_band_mA": 1e3 * ref.band_real, "mean_mA": 1e3 * allc.mean_real,
                        "band_mA": 1e3 * allc.band_real, "shift_mA": shm, "shift_over_band": sh, "band_ratio": br,
                        "pipeline_mean_norm": ref.mean_norm, "pipeline_band_norm": ref.band_norm,
                        "mean_norm": allc.mean_norm, "band_norm": allc.band_norm})
    post = []
    for k, f in enumerate(removed):
        xe = np.log10(pre[f]["tmax_s"] + C.X_SHIFT)
        post.append(pd.DataFrame({"curve": R + k, "log10_t": C.GRID, "t_s": TG, "mean_norm": Yx[k], "sd_norm": Sx[k],
                                  "mean_mA": 1e3 * (Yx[k] * Fx[k]), "measured": C.GRID <= xe + 1e-4, "notes": f}))
    weights = pd.DataFrame({"curve": np.arange(N), "reincluded": reinc, "record_end_s": rec_end, "complete": complete,
                            "contact_loss": cl, "substrate_au": au, "scale_factor_mA": 1e3 * Fa,
                            "weight_pipeline": np.r_[w_ref, np.full(n_x, np.nan)], "weight": w_all,
                            "weight_n_pipeline": np.r_[w_ref * R, np.full(n_x, np.nan)], "weight_n": wn,
                            "notes": names})
    recs = pd.DataFrame({"curve": R + np.arange(n_x), "record_end_s": rec_end[R:],
                         "n_samples": [pre[f]["n_raw"] for f in removed],
                         "n_capped": [pre[f]["n_capped"] for f in removed],
                         "n_removed_threshold": [pre[f]["n_removed_threshold"] for f in removed],
                         "n_train": [pre[f]["n_train"] for f in removed], "scale_factor_mA": 1e3 * Fx,
                         "mean_sq_sd_norm2": s2w[R:], "mean_sq_dev_norm2": dev[R:], "weight": wx, "weight_n": wx * N,
                         "notes": removed})
    ret_c, ret_t = complete[:R], ~complete[:R]
    summ = dict(n_pipeline=R, n_reincluded=n_x, n_pool=N,
                max_abs_shift_mA=float(np.abs(shm).max()), t_max_abs_shift_mA_s=float(TG[jmm]),
                rms_shift_mA=float(np.sqrt(np.mean(shm ** 2))),
                max_abs_shift_over_band=float(np.abs(sh).max()), t_max_abs_shift_s=float(TG[jm]),
                rms_shift_over_band=float(np.sqrt(np.mean(sh ** 2))),
                max_abs_shift_over_band_to_2990s=float(np.abs(sh[TG <= C.T_COMPLETE]).max()),
                max_abs_shift_norm_over_band_norm=float(np.abs(shn).max()),
                max_rel_change_mean=float(np.max(np.abs(allc.mean_real - ref.mean_real) / np.abs(ref.mean_real))),
                weight_reincluded=w_ex, weight_reincluded_uniform=n_x / N, max_weight_n_reincluded=float(wx.max() * N),
                median_weight_n_reincluded=float(np.median(wx) * N),
                n_reincluded_weight_n_above_01=int((wx * N > 0.1).sum()),
                n_reincluded_weight_n_below_1e3=int((wx * N < 1e-3).sum()),
                median_weight_n_truncated=float(np.median(wn[:R][ret_t])),
                median_weight_n_complete=float(np.median(wn[:R][ret_c])),
                band_ratio_median=float(np.median(br)), band_ratio_max_abs_dev=float(np.abs(br - 1).max()),
                s_agg_rel_change=float(allc.s_agg / ref.s_agg - 1), max_weight_pipeline=float(w_ref.max()),
                max_weight=float(w_all.max()), n_eff_pipeline=ref.n_eff, n_eff=allc.n_eff,
                iterations_pipeline=nit_ref, iterations=nit_all,
                spearman_weights_retained=float(spearmanr(w_all[:R], w_ref).correlation),
                median_mean_sq_sd_norm2_reincluded=float(np.median(s2w[R:])),
                median_mean_sq_sd_norm2_retained=float(np.median(s2w[:R])),
                median_mean_sq_dev_norm2_reincluded=float(np.median(dev[R:])),
                median_mean_sq_dev_norm2_retained=float(np.median(dev[:R])))
    print(f"{R} + {n_x} = {N} curves: weight of the re-included records {w_ex:.5f}; max |shift| "
          f"{summ['max_abs_shift_over_band']:.4f} band at {summ['t_max_abs_shift_s']:.3f} s "
          f"({summ['max_abs_shift_mA']:.4f} mA); N_eff {ref.n_eff:.2f} -> {allc.n_eff:.2f}", flush=True)

    C.write_tables(__file__, {
        "aggregate_on_grid": agg,
        "reincluded_posterior": pd.concat(post, ignore_index=True),
        "weights_by_curve": weights,
        "reincluded_records": recs,
        "reinclusion_summary": pd.DataFrame([summ]),
    })
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
