"""Sensitivity of the GPR-FDA aggregate at -1.95 V to the current normalization: the sample window of the per-curve
scale factor, and the aggregation in current units without normalization.

Scale-factor window
    Each of the 154 pipeline curves is rebuilt from its raw record (_input/CCNF_CTAB_PT): samples sorted by T (pandas
    sort_values) and kept within 1e-4 <= T <= 5000 s. The scale factor of a window (a, b) is the mean current of the
    samples int(a n) .. int(b n) - 1. The pipeline window is (0.50, 0.90); the tested windows are (0.25, 0.75),
    (0.40, 0.80), (0.60, 0.95), (0.70, 1.00) and (0.50, 1.00). The individual GPR is invariant to y -> c y, so the
    normalized posterior of a curve for another window is the pipeline posterior (_output/.../individual_GPR) times
    k = s_pipeline / s_window, its sd |S| times |k|. A curve whose set of samples with |I/s| >= 0.01 differs from
    that of the pipeline window is refitted from the raw record with the window's factor: y = I/s, |y| < 0.01
    dropped, x = log10(T + 1e-9), at most 500 training points (middle sample of each uniform bin in x),
    StandardScaler on x and y, ConstantKernel(1, (1e-2, 1e4)) * Matern(1, (1e-2, 1e3), nu = 1.5) +
    WhiteKernel(1, (1e-12, 10)), alpha 0, normalize_y, L-BFGS-B with 5 restarts, random_state 0, prediction on the
    pipeline grid with sd = sqrt(max(diag(cov), 0)). All fits run one after another with 3 BLAS threads.

    Window aggregate: GPR-FDA on the normalized posteriors Y, S of the window and the window factors s: curve-wise
    weights from w = 1/R with s2w_r = mean_j S_r(j)^2, ybar = sum_r w_r Y_r, D_r = mean_j (Y_r - ybar)^2,
    w_new = 1/(D_r + s2w_r + 1e-12) normalized to 1, until ||w_new - w|| / sqrt(R) < 1e-6; mean = ybar s_agg with
    s_agg = sum_r w_r s_r; band^2 = sum_r w_r (Y_r s_r - mean)^2 + sum_r w_r (S_r s_r)^2. Shift = (mean_window -
    mean_pipeline) / band_pipeline with the Summary_GPR mean and band of the pipeline run.

    Four aggregation methods (aggregate_shift_by_window): GPR-FDA as above, point-wise iterative weights (run to
    convergence, at most 10000 iterations), the direct average of the currents and the direct average of the
    normalized currents, each on the window posteriors and on the pipeline posteriors. Shift of a method = its mean
    on the window posteriors minus its mean on the pipeline posteriors, divided by the GPR-FDA band of the pipeline
    posteriors.

    Self-checks: the factors and training sets of the pipeline window are rebuilt exactly; the y -> c y invariance of
    the individual fit (the three curves with the largest |ln k(0.25, 0.75)| + |ln k(0.70, 1.00)| among the curves
    with pipeline weight > 1e-6 that are refitted in neither of these windows, and the highest-weight complete and
    truncated curves, each fitted with y and y k); refits with the pipeline factor reproduce the pipeline posteriors;
    the window aggregate with the pipeline window and the GPR-FDA aggregate of the pipeline posteriors reproduce
    Converged_Weights and Summary_GPR, the latter with the number of iterations in Weight_History.

Current units
    GPR-FDA weights on the pipeline posteriors in current units, Y_r s_r and S_r |s_r|, with the same update and stop
    rule; mean = sum_r w_r Y_r s_r, band^2 = sum_r w_r (Y_r s_r - mean)^2 + sum_r w_r (S_r s_r)^2. The reference is the
    GPR-FDA aggregate of the normalized pipeline posteriors.

Time windows of the shift statistics ([lo, hi) on t_s; t_s of the first grid point is 0.099999999 s)
    window_summary          01_1s [0.1, 1) s (without the first grid point), 1_100s [1, 100) s, 100_5000s
                            [100, 5000.000001) s
    current_units_summary   01_1s [0.0999, 1) s, 1_10s [1, 10) s, 10_100s [10, 100) s, 100_5000s [100, 5001) s

Output tables (eval_scripts/output/normalization/)
    window_per_curve        one row per curve: curve (index in Converged_Weights), record_end_s, complete
                            (record_end_s >= 2990), contact_loss (pipeline weight < 1e-8), scale_factor_mA (pipeline
                            window, A cm^-2 x 1e3), weight_pipeline (Converged_Weights), window_start_050_090_s and
                            window_end_050_090_s (times of the first and last sample of the window); per window tag
                            (025_075, 040_080, 060_095, 070_100, 050_100): scale_factor_ratio_<tag> = s_pipeline /
                            s_window, refitted_<tag> (0/1), weight_<tag> (GPR-FDA weights of the window aggregate),
                            window_start_<tag>_s, window_end_<tag>_s; notes = sample id
    representative_record   curve, t_s, current_mA: the raw samples (sorted, time caps) of the highest-weight complete
                            record and of the truncated record whose end is closest to the median end of the truncated
                            records; notes = sample id
    window_shift_on_grid    t_s, pipeline_mean_mA, pipeline_band_mA (Summary_GPR), shift_over_band_<tag>
    window_summary          one row per window: window_lo, window_hi, n_refitted, iterations, spearman_weights
                            (against Converged_Weights), n_eff, n_eff_pipeline, max_abs_shift_over_band,
                            rms_shift_over_band, t_max_abs_shift_s, max_abs_shift_over_band_<k> and
                            rms_shift_over_band_<k> for k = 01_1s, 1_100s, 100_5000s, band_ratio_median,
                            band_ratio_min, band_ratio_max, max_weight_ratio_up, max_weight_ratio_down (largest
                            weight_<tag> / weight_pipeline and its inverse over the curves without contact loss),
                            weight_share_truncated, weight_share_truncated_pipeline, s_agg_mA, notes
    aggregate_shift_by_window
                            one row per window: window_lo, window_hi, gprfda_max, gprfda_rms, pointwise_max,
                            pointwise_rms, direct_obs_max, direct_obs_rms, direct_norm_max, direct_norm_rms
                            (max and RMS over the grid of |mean_window - mean_pipeline| / GPR-FDA band of the
                            pipeline posteriors), pointwise_iterations, pointwise_converged, n_reference, n_variant,
                            notes
    current_units_on_grid   t_s, pipeline_mean_mA, pipeline_band_mA (Summary_GPR), mean_mA, band_mA, shift_mA
                            (mean_mA - pipeline_mean_mA), shift_over_band ((mean - reference mean) / reference band)
    current_units_per_curve curve, record_end_s, complete, contact_loss, scale_factor_mA, weight_pipeline (reference
                            weights), weight, notes
    current_units_summary   one row: max_abs_shift_over_band, rms_shift_over_band, t_max_abs_shift_s,
                            max_abs_shift_over_band_<k> for k = 01_1s, 1_10s, 10_100s, 100_5000s, max_rel_change_mean,
                            band_ratio_median, band_ratio_min, band_ratio_max, n_eff, iterations, spearman_weights,
                            spearman_weights_scale, spearman_weights_pipeline_scale, weight_share_truncated,
                            weight_share_truncated_pipeline, weight_share_15_largest_scale,
                            weight_share_15_largest_scale_pipeline, weight_share_15_smallest_scale,
                            weight_share_15_smallest_scale_pipeline, max_weight_n, max_weight_n_pipeline,
                            weight_contact_loss (3 smallest |s|), max_abs_shift_mA, t_max_abs_shift_mA_s,
                            max_abs_shift_mA_after_10s

Run
    python -X utf8 -B eval_scripts/eval_normalization.py
"""
import sys

sys.dont_write_bytecode = True

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _eval_common as C
import _eval_refit as RF

POT = "-1.95"
BASE = (0.50, 0.90)
WINDOWS = ((0.25, 0.75), (0.40, 0.80), (0.60, 0.95), (0.70, 1.00), (0.50, 1.00))
X_COL, Y_COL = "T (Seconds)", "I (A/cm2)"
BLAS_THREADS = 3
CL_WEIGHT = 1e-8
SHIFT_WINDOWS = (("01_1s", 0.1, 1.0), ("1_100s", 1.0, 100.0), ("100_5000s", 100.0, 5000.0 + 1e-6))
UNIT_WINDOWS = (("01_1s", 0.0999, 1.0), ("1_10s", 1.0, 10.0), ("10_100s", 10.0, 100.0), ("100_5000s", 100.0, 5001.0))


def tag(w):
    return f"{w[0]:.2f}_{w[1]:.2f}".replace(".", "")


def label(w):
    return f"window {w[0]:.2f}-{w[1]:.2f}"


def raw_record(sid):
    """Raw record sorted by T and kept within the time caps -> (T, I)."""
    df, _ = C.load_record(sid)
    d = df.sort_values(by=X_COL).reset_index(drop=True)
    d = d[(d[X_COL] >= C.X_CAPS[0]) & (d[X_COL] <= C.X_CAPS[1])]
    return d[X_COL].to_numpy(float), d[Y_COL].to_numpy(float)


def window_factor(I, w):
    """Mean current of the samples int(a n) .. int(b n) - 1 and the two sample indices."""
    n = len(I)
    i0, i1 = int(w[0] * n), int(w[1] * n)
    return float(np.mean(I[i0:i1])), i0, i1


def training_set(T, I, f):
    """Normalized record, |y| >= 0.01 kept, uniform-bin downsampling -> (x, y, keep mask)."""
    df = pd.DataFrame({"x_transformed": np.log10(T + C.X_SHIFT), "y_transformed": I / f})
    keep = df.y_transformed.abs() >= RF.Y_THRESHOLD
    ds, _ = RF.downsample_uniform_bins(df[keep].copy(), RF.MAX_POINTS)
    return ds.x_transformed.to_numpy(), ds.y_transformed.to_numpy(), keep.to_numpy()


def posterior(f):
    """Posterior of a fit with sd = sqrt(max(diag(cov), 0)): dict(mu, sd, n_negvar, theta, lml)."""
    neg = np.isnan(f.sd)
    return dict(mu=f.mu, sd=np.where(neg, 0.0, f.sd), n_negvar=int(neg.sum()),
                theta=np.log(np.array([f.C_fit, f.l_fit, f.noise_fit])), lml=f.lml_fit)


def aggregate_methods(Y, S, F):
    """Aggregates of the four methods on the curves Y, S with the factors F; point-wise run to convergence."""
    return {m: C.aggregate_arrays(Y, S, F, m, cap=C.POINTWISE_MAX_ITER if m == "pointwise" else None)
            for m in C.METHODS}


def shift_stats(d, t, windows, prefix="shift_over_band"):
    out = {f"max_abs_{prefix}": float(np.abs(d).max()), f"rms_{prefix}": float(np.sqrt(np.mean(d ** 2))),
           "t_max_abs_shift_s": float(t[int(np.argmax(np.abs(d)))])}
    for k, lo, hi in windows:
        m = (t >= lo) & (t < hi)
        out[f"max_abs_{prefix}_{k}"] = float(np.abs(d[m]).max())
        out[f"rms_{prefix}_{k}"] = float(np.sqrt(np.mean(d[m] ** 2)))
    return out


def window_curves(pool, k, refit, rows):
    """Pipeline posteriors times k (sd |S| |k|) with the refitted curves `rows` replaced -> (Y, S, n refitted)."""
    Yn = pool.Y * k[:, None]
    Sn = np.abs(pool.S) * np.abs(k)[:, None]
    for r in rows:
        Yn[r], Sn[r] = refit[int(r)]["mu"], refit[int(r)]["sd"]
    return Yn, Sn, len(rows)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)
    summ = C.load_summary(POT)
    y_ref, sd_ref = summ.y_real.to_numpy(float), summ.Std_real.to_numpy(float)
    tg = summ.x_real.to_numpy(float)
    f_pipeline = pool.F
    cl = w_pipeline < CL_WEIGHT

    raw, tmax = [], np.empty(R)
    fac = {w: np.empty(R) for w in (BASE,) + WINDOWS}
    span = {w: np.empty((R, 2)) for w in (BASE,) + WINDOWS}
    keep = {}
    n_bad, max_dx, max_dy, n_val_bad = 0, 0.0, 0.0, 0
    for r, sid in enumerate(pool.sids):
        T, I = raw_record(sid)
        raw.append((T, I))
        tmax[r] = T.max()
        for w in fac:
            fac[w][r], i0, i1 = window_factor(I, w)
            span[w][r] = T[i0], T[i1 - 1]
        xt, yt, kp = training_set(T, I, fac[BASE][r])
        keep[(r, BASE)] = kp
        xs, ys = C.pipeline_training_set(sid)
        n_val = int(C.read_header(C.diagnostic_path(sid))["n_validation"])
        if len(xs) != len(xt):
            n_bad += 1
        else:
            max_dx = max(max_dx, float(np.abs(xs - xt).max()))
            max_dy = max(max_dy, float(np.max(np.abs(ys - yt) / np.abs(yt))))
        n_val_bad += int(n_val != kp.sum())
    rel_f = float(np.max(np.abs(fac[BASE] - f_pipeline) / np.abs(f_pipeline)))
    C.check("pipeline scale factors rebuilt from the raw records", rel_f < 1e-6, f"max relative error {rel_f:.2e}")
    C.check("pipeline training sets rebuilt from the raw records", n_bad == 0 and max_dx <= 1e-12 and max_dy <= 1e-12
            and n_val_bad == 0, f"{R - n_bad}/{R} same size; max |dx| {max_dx:.2e}; max relative |dy| {max_dy:.2e}; "
            f"filtered-record size mismatches {n_val_bad}")
    complete = tmax >= C.T_COMPLETE
    refit_mask = {}
    for w in WINDOWS:
        ch = np.zeros(R, bool)
        for r in range(R):
            T, I = raw[r]
            keep[(r, w)] = np.abs(I / fac[w][r]) >= RF.Y_THRESHOLD
            ch[r] = not np.array_equal(keep[(r, w)], keep[(r, BASE)])
        refit_mask[w] = ch
    need = sorted({int(r) for w in WINDOWS for r in np.flatnonzero(refit_mask[w])})
    print(f"{R} raw records read in {time.time() - t0:.0f} s; curves to refit: {need}; per window: "
          + ", ".join(f"{tag(w)} {int(refit_mask[w].sum())}" for w in WINDOWS), flush=True)

    w25, w70 = (0.25, 0.75), (0.70, 1.00)
    lr = np.abs(np.log(np.abs(f_pipeline / fac[w25]))) + np.abs(np.log(np.abs(f_pipeline / fac[w70])))
    cand = [int(r) for r in np.argsort(-lr) if w_pipeline[r] > 1e-6 and not refit_mask[w25][r]
            and not refit_mask[w70][r]]
    pick = cand[:3] + [int(np.argmax(np.where(complete & (w_pipeline > 1e-6), w_pipeline, -1))),
                       int(np.argmax(np.where(~complete & (w_pipeline > 1e-6), w_pipeline, -1)))]
    jobs, meta = [], []
    for r in pick:
        T, I = raw[r]
        xt, yt, _ = training_set(T, I, f_pipeline[r])
        jobs.append(dict(x=xt, y=yt, x_pred=pool.x, std_from="cov"))
        meta.append(("invariance_base", r, BASE, 1.0))
        for w in (w25, w70):
            k = f_pipeline[r] / fac[w][r]
            jobs.append(dict(x=xt, y=yt * k, x_pred=pool.x, std_from="cov"))
            meta.append(("invariance_scaled", r, w, k))
    for r in need:
        T, I = raw[r]
        xt, yt, _ = training_set(T, I, f_pipeline[r])
        jobs.append(dict(x=xt, y=yt, x_pred=pool.x, std_from="cov"))
        meta.append(("pipeline_window", r, BASE, 1.0))
        for w in WINDOWS:
            if refit_mask[w][r]:
                xt2, yt2, _ = training_set(T, I, fac[w][r])
                jobs.append(dict(x=xt2, y=yt2, x_pred=pool.x, std_from="cov"))
                meta.append(("window_refit", r, w, f_pipeline[r] / fac[w][r]))
    tf = time.time()
    fits = RF.fit_jobs(jobs, n_jobs=1, verbose=0, blas_threads=BLAS_THREADS)
    print(f"{len(fits)} GPR fits in {time.time() - tf:.0f} s", flush=True)
    post = [posterior(f) for f in fits]

    inv, rep = [], []
    refit = {w: {} for w in WINDOWS}
    base_fit = None
    for (kind, r, w, k), p in zip(meta, post):
        if kind in ("invariance_base", "pipeline_window"):
            base_fit = p
            rep.append((np.max(np.abs(p["mu"] - pool.Y[r]) / np.abs(pool.S[r])),
                        np.max(np.abs(p["sd"] - np.abs(pool.S[r])) / np.abs(pool.S[r]))))
        elif kind == "invariance_scaled":
            inv.append((np.abs(p["theta"] - base_fit["theta"]).max(), p["lml"] - base_fit["lml"],
                        np.max(np.abs(p["mu"] / k - base_fit["mu"])) / np.max(np.abs(base_fit["mu"])),
                        np.max(np.abs(p["sd"] / abs(k) - base_fit["sd"]) / base_fit["sd"])))
        else:
            refit[w][r] = p
    inv, rep = np.array(inv), np.array(rep)
    C.check("individual GPR invariant to y -> c y", inv[:, 2].max() <= 1e-6 and inv[:, 3].max() <= 1e-5,
            f"{len(inv)} fits; max |d log theta| {inv[:, 0].max():.1e}; max |dLML| {np.abs(inv[:, 1]).max():.1e}; "
            f"max relative d mean {inv[:, 2].max():.1e}; max relative d sd {inv[:, 3].max():.1e}")
    C.check("refits with the pipeline factor reproduce the pipeline posteriors", rep.max() <= 1e-6,
            f"{len(rep)} fits; max |d mean| / sd {rep[:, 0].max():.1e}; max relative d sd {rep[:, 1].max():.1e}")
    n_neg = sum(p["n_negvar"] for v in refit.values() for p in v.values())
    C.check("no negative posterior variance in the window refits", n_neg == 0,
            f"{sum(len(v) for v in refit.values())} refits")

    Yb, Sb, _ = window_curves(pool, f_pipeline / fac[BASE], {}, [])
    b = C.aggregate_arrays(Yb, Sb, fac[BASE], "gprfda")
    d_w = float(np.abs(b.w - w_pipeline).max())
    d_agg = max(float(np.max(np.abs(a - summ[c].to_numpy(float)) / np.abs(summ[c].to_numpy(float))))
                for a, c in ((b.mean_norm, "y_normalised"), (b.band_norm, "Std_normalised"), (b.mean_real, "y_real"),
                             (b.band_real, "Std_real")))
    C.check("window aggregate with the pipeline window = Converged_Weights and Summary_GPR",
            d_w < 1e-10 and d_agg < 1e-8,
            f"weights {d_w:.1e}; mean and band {d_agg:.1e} (relative); {b.iterations} iterations")
    full = C.aggregate_all(pool)
    g = full["gprfda"]
    n_hist = len(C.load_weight_history(POT))
    d_wg = float(np.abs(g.w - w_pipeline).max())
    d_mg = float(np.max(np.abs(g.mean_real / y_ref - 1)))
    d_bg = float(np.max(np.abs(g.band_real / sd_ref - 1)))
    C.check("GPR-FDA aggregate of the pipeline posteriors = Converged_Weights, Summary_GPR, Weight_History length",
            g.iterations == n_hist and d_wg < 1e-12 and d_mg < 1e-10 and d_bg < 1e-10,
            f"weights {d_wg:.1e}; mean {d_mg:.1e}; band {d_bg:.1e} (relative); {g.iterations} iterations "
            f"({n_hist} in Weight_History); point-wise {full['pointwise'].iterations} iterations, converged "
            f"{full['pointwise'].converged}")
    den = g.band_real

    per_curve = pd.DataFrame({"curve": np.arange(R), "record_end_s": tmax, "complete": complete, "contact_loss": cl,
                              "scale_factor_mA": 1e3 * f_pipeline, "weight_pipeline": w_pipeline,
                              f"window_start_{tag(BASE)}_s": span[BASE][:, 0],
                              f"window_end_{tag(BASE)}_s": span[BASE][:, 1]})
    grid = pd.DataFrame({"t_s": tg, "pipeline_mean_mA": 1e3 * y_ref, "pipeline_band_mA": 1e3 * sd_ref})
    summary, shift_rows = [], []
    for w in WINDOWS:
        rows = np.flatnonzero(refit_mask[w])
        Yn, Sn, nref = window_curves(pool, f_pipeline / fac[w], refit[w], rows)
        var = aggregate_methods(Yn, Sn, fac[w])
        ag = var["gprfda"]
        C.check(f"{label(w)}: GPR-FDA weights converged", ag.converged, f"{ag.iterations} iterations")
        d = (ag.mean_real - y_ref) / sd_ref
        ratio = ag.w[~cl] / w_pipeline[~cl]
        row = dict(window_lo=w[0], window_hi=w[1], n_refitted=nref, iterations=ag.iterations,
                   spearman_weights=float(spearmanr(ag.w, w_pipeline).statistic), n_eff=ag.n_eff,
                   n_eff_pipeline=C.n_eff(w_pipeline))
        row.update(shift_stats(d, tg, SHIFT_WINDOWS))
        br = ag.band_real / sd_ref
        row.update(band_ratio_median=float(np.median(br)), band_ratio_min=float(br.min()),
                   band_ratio_max=float(br.max()),
                   max_weight_ratio_up=float(ratio.max()), max_weight_ratio_down=float((1 / ratio).max()),
                   weight_share_truncated=float(ag.w[~complete].sum()),
                   weight_share_truncated_pipeline=float(w_pipeline[~complete].sum()), s_agg_mA=1e3 * ag.s_agg,
                   notes=label(w))
        summary.append(row)
        shift_rows.append(dict(window_lo=w[0], window_hi=w[1], **C.shift_summary(full, var, den),
                               notes=label(w) + f", {nref} curve(s) refitted"))
        t_ = tag(w)
        per_curve[f"scale_factor_ratio_{t_}"] = f_pipeline / fac[w]
        per_curve[f"refitted_{t_}"] = refit_mask[w]
        per_curve[f"weight_{t_}"] = ag.w
        per_curve[f"window_start_{t_}_s"] = span[w][:, 0]
        per_curve[f"window_end_{t_}_s"] = span[w][:, 1]
        grid[f"shift_over_band_{t_}"] = d
        print(f"{label(w)}: {nref} refitted, {ag.iterations} iterations, max |shift| {np.abs(d).max():.4f} band, "
              f"Spearman {row['spearman_weights']:.4f}, N_eff {ag.n_eff:.2f}", flush=True)
    per_curve["notes"] = pool.sids
    tr = np.flatnonzero(~complete)
    examples = (int(np.argmax(np.where(complete, w_pipeline, -1.0))),
                int(tr[np.argmin(np.abs(tmax[tr] - np.median(tmax[tr])))]))
    records = pd.concat([pd.DataFrame({"curve": r, "t_s": raw[r][0], "current_mA": 1e3 * raw[r][1],
                                       "notes": pool.sids[r]}) for r in examples], ignore_index=True)

    sc, w0 = pool.F, g.w
    cur = C.aggregate_arrays(pool.Y * sc[:, None], pool.S * np.abs(sc)[:, None], np.ones(R), "gprfda")
    wr, yr, sdr = cur.w, cur.mean_norm, cur.band_norm
    sh = (yr - g.mean_real) / g.band_real
    tu = C.T_GRID
    shm = 1e3 * yr - 1e3 * y_ref
    C.check("current-unit weights converged", cur.converged, f"{cur.iterations} iterations")
    big, small = np.argsort(np.abs(sc))[-15:], np.argsort(np.abs(sc))[:15]
    cu = shift_stats(sh, tu, UNIT_WINDOWS)
    cu = {k: v for k, v in cu.items() if not k.startswith("rms_shift_over_band_")}
    cu.update(max_rel_change_mean=float(np.abs(yr / g.mean_real - 1).max()),
              band_ratio_median=float(np.median(sdr / g.band_real)),
              band_ratio_min=float((sdr / g.band_real).min()), band_ratio_max=float((sdr / g.band_real).max()),
              n_eff=cur.n_eff, iterations=cur.iterations,
              spearman_weights=float(spearmanr(wr, w0).statistic),
              spearman_weights_scale=float(spearmanr(wr, np.abs(sc)).statistic),
              spearman_weights_pipeline_scale=float(spearmanr(w0, np.abs(sc)).statistic),
              weight_share_truncated=float(wr[~complete].sum()),
              weight_share_truncated_pipeline=float(w0[~complete].sum()),
              weight_share_15_largest_scale=float(wr[big].sum()),
              weight_share_15_largest_scale_pipeline=float(w0[big].sum()),
              weight_share_15_smallest_scale=float(wr[small].sum()),
              weight_share_15_smallest_scale_pipeline=float(w0[small].sum()),
              max_weight_n=float(wr.max() * R), max_weight_n_pipeline=float(w0.max() * R),
              weight_contact_loss=float(wr[np.argsort(np.abs(sc))[:3]].sum()),
              max_abs_shift_mA=float(np.abs(shm).max()),
              t_max_abs_shift_mA_s=float(tu[int(np.argmax(np.abs(shm)))]),
              max_abs_shift_mA_after_10s=float(np.abs(shm[tu >= 10]).max()))
    print(f"current units: max |shift| {cu['max_abs_shift_over_band']:.4f} band, N_eff {cu['n_eff']:.2f}, "
          f"Spearman {cu['spearman_weights']:.3f}", flush=True)
    cu_grid = pd.DataFrame({"t_s": tu, "pipeline_mean_mA": 1e3 * y_ref, "pipeline_band_mA": 1e3 * sd_ref,
                            "mean_mA": 1e3 * yr, "band_mA": 1e3 * sdr, "shift_mA": shm, "shift_over_band": sh})
    cu_curve = pd.DataFrame({"curve": np.arange(R), "record_end_s": tmax, "complete": complete, "contact_loss": cl,
                             "scale_factor_mA": 1e3 * sc, "weight_pipeline": w0, "weight": wr,
                             "notes": pool.sids})

    C.write_tables(__file__, {
        "window_per_curve": per_curve,
        "window_shift_on_grid": grid,
        "window_summary": pd.DataFrame(summary),
        "representative_record": records,
        "aggregate_shift_by_window": pd.DataFrame(shift_rows),
        "current_units_on_grid": cu_grid,
        "current_units_per_curve": cu_curve,
        "current_units_summary": pd.DataFrame([cu]),
    })
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
