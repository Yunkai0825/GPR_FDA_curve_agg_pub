"""Decomposition of the GPR-FDA band variance on the normalized scale into the between-curve term and the within-curve
(posterior) term, the latter split by curve group.

Method
    Curve-wise weights of a pool (GPR-FDA): s2w_r = mean_j S_r(j)^2; w = 1/R; repeat ybar = sum_r w_r Y_r,
    D_r = mean_j (Y_r - ybar)^2, w_new = 1/(D_r + s2w_r + 1e-12) normalized to 1, delta = ||w_new - w|| / sqrt(R),
    w = w_new, until delta < 1e-6. With these weights, the normalized posterior means Y_r and sds S_r on the grid t = 10^x,
    x = linspace(-1, 3.69897, 500):
        var_between(t)     = sum_r w_r (Y_r(t) - sum_q w_q Y_q(t))^2
        var_within_g(t)    = sum_{r in g} w_r S_r(t)^2
        var_total(t)       = var_between(t) + sum_g var_within_g(t) = band(t)^2
    Groups by record end: complete (>= 2990 s), mid (60-2990 s), lt60 (< 60 s). Each posterior variance counts over the
    whole grid, inside and beyond the record end. Window medians of the shares over [0.1, 1), [1, 10), [10, 100),
    [100, 1000), [1000, 5000.5) s.

Pools
    -1.95 V, 154 curves: the pipeline posteriors (_output/.../individual_GPR, order of Converged_Weights); record end =
    last sample of the raw record sorted by T within 1e-4 <= T <= 5000 s.
    -1.95 V, 185 curves: the 154 pipeline posteriors followed by the records removed by the 60 s rule, in the order of
    the start time in their headers, each fitted with the 60 s rule switched off (samples sorted by T, 1e-4 <= T <=
    5000 s, x = log10(T + 1e-9), y = I / s with s the mean current of the samples int(0.5 n) .. int(0.9 n) - 1,
    |y| < 0.01 dropped, at most 500 training points, middle sample of each uniform bin in x; StandardScaler on x and
    y, ConstantKernel(1, (1e-2, 1e4)) * Matern(1, (1e-2, 1e3), nu = 1.5) + WhiteKernel(1, (1e-12, 10)), alpha 0,
    normalize_y, 5 restarts, random_state 0, posterior sd from predict with return_std, one BLAS thread per fit, fits
    in --n-jobs joblib workers; the results do not depend on it); record end of a removed record = its largest sample
    time. Weights iterated on the 185 curves.
    -2.15 V, 12 curves: the pipeline posteriors, record ends as for the 154 curves.

Self-checks
    The weights of the pipeline pools reproduce Converged_Weights and their bands Summary_GPR Std_normalised; the terms
    add up to the squared band; the removed records are shorter than 60 s; the weights of the 185 curves converged.

Output tables (eval_scripts/output/band_decomposition/)
    band_terms_on_grid          potential_v, n_curves, log10_t, t_s, var_total_norm2, var_between_norm2,
                                var_within_norm2, var_within_complete_norm2, var_within_mid_norm2,
                                var_within_lt60_norm2, share_between, share_within_complete, share_within_mid,
                                share_within_lt60, stack_complete_pct, stack_complete_mid_pct,
                                stack_complete_mid_lt60_pct (stacked percentages of the within-curve shares; the
                                between-curve share fills the rest to 100); notes = pool
    band_share_window_medians   potential_v, n_curves, window_lo_s, window_hi_s, n_grid, share_between_median,
                                share_within_total_median, share_within_complete_median, share_within_mid_median,
                                share_within_lt60_median; notes = window
    curve_groups                potential_v, n_curves, max_weight, n_complete, weight_complete, n_mid, weight_mid,
                                n_lt60, weight_lt60, record_end_min_mid_s, record_end_max_mid_s; notes = pool

Run
    python -X utf8 -B eval_scripts/eval_band_decomposition.py [--n-jobs K]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

import _eval_common as C
import _eval_refit as RF

X_COL = "T (Seconds)"
HDR = re.compile(r"Date:\s*(\d{2})-(\d{2})-(\d{4})\s+Time:\s*(\d{2}):(\d{2}):(\d{2})")


def header_start(path):
    """Start date and time in the header of a .cor file."""
    with open(path, "rb") as fh:
        txt = fh.read(6000).decode("latin-1")
    m = HDR.search(txt)
    return datetime(int(m[3]), int(m[1]), int(m[2]), int(m[4]), int(m[5]), int(m[6]))


def records_by_start(pot):
    """Files of one potential (file name) sorted by header start time."""
    idx = C.raw_index()
    files = sorted(f for f in idx if C.record_meta(f)["potential_V"] == float(pot))
    start = {f: header_start(idx[f][0]) for f in files}
    return sorted(files, key=lambda f: start[f])


def band_norm(Y, S, w):
    """GPR-FDA band in normalized current for the weights w."""
    W = w[:, None]
    yb = (W * Y).sum(axis=0)
    return np.sqrt((W * (Y - yb) ** 2).sum(axis=0) + (W * S ** 2).sum(axis=0))


def pool_terms(pot, label, Y, S, w, tend):
    """Band terms on the grid, window medians and group summary of one pool."""
    terms = C.band_terms(Y, S, w, tend)
    sb = band_norm(Y, S, w)
    tot = terms.var_total_norm2.to_numpy()
    C.check(f"{label}: band terms add up to the squared band", np.allclose(tot, sb ** 2, rtol=1e-10, atol=0),
            f"max relative deviation {np.max(np.abs(tot / sb ** 2 - 1)):.1e}")
    head = pd.DataFrame({"potential_v": float(pot), "n_curves": len(w), "log10_t": C.GRID})
    grid = pd.concat([head, terms], axis=1)
    grid["notes"] = label
    med = C.band_window_medians(terms)
    lo = {lab: a for lab, a, b, _ in C.DECADE_WINDOWS}
    hi = {lab: b for lab, a, b, _ in C.DECADE_WINDOWS}
    med.insert(0, "window_hi_s", med.window.map(hi))
    med.insert(0, "window_lo_s", med.window.map(lo))
    med.insert(0, "n_curves", len(w))
    med.insert(0, "potential_v", float(pot))
    med["notes"] = med.pop("window")
    grp = C.record_groups(tend)
    row = dict(potential_v=float(pot), n_curves=len(w), max_weight=float(w.max()))
    for k, m in grp.items():
        row[f"n_{k}"] = int(m.sum())
        row[f"weight_{k}"] = float(w[m].sum())
    row.update(record_end_min_mid_s=float(tend[grp["mid"]].min()), record_end_max_mid_s=float(tend[grp["mid"]].max()),
               notes=label)
    return grid, med, row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="joblib workers for the GPR fits")
    a = ap.parse_args()
    t0 = time.time()
    pools, tends, ws = {}, {}, {}
    for pot in ("-1.95", "-2.15"):
        pools[pot] = C.load_pool(pot)
        ws[pot], nit, _, _ = C.curve_weights(pools[pot].Y, pools[pot].S)
        tends[pot] = C.record_ends(pools[pot].sids)
        w_pipeline = C.load_weights(pot).converged_weight.to_numpy(float)
        band_pipeline = C.load_summary(pot).Std_normalised.to_numpy(float)
        sb = band_norm(pools[pot].Y, pools[pot].S, ws[pot])
        d_w = float(np.abs(ws[pot] - w_pipeline).max())
        d_b = float(np.max(np.abs(sb - band_pipeline) / band_pipeline))
        C.check(f"{pot} V pipeline pool: weights = Converged_Weights, band = Summary_GPR Std_normalised",
                d_w < 1e-10 and d_b < 1e-10, f"weights {d_w:.1e}; band {d_b:.1e} (relative); {nit} iterations")
    print(f"pipeline pools and record ends read in {time.time() - t0:.0f} s", flush=True)

    in_pool = {C.split_sid(s)[0] for s in pools["-1.95"].sids}
    removed = [f for f in records_by_start("-1.95") if f not in in_pool]
    pre, tmax, rng = {}, {}, {}
    for f in removed:
        df, _ = C.load_record(C.find_raw(f))
        pre[f], why = RF.preprocess_record(df, apply_min_range=False)
        tmax[f] = float(df[X_COL].max())
        rng[f] = float(df[X_COL].max() - df[X_COL].min())
        if pre[f] is None:
            C.check(f"preprocessing of {f}", False, why)
    C.check("removed -1.95 V records shorter than 60 s, preprocessed with the 60 s rule off",
            all(pre[f] is not None for f in removed) and max(rng.values()) < C.T_MIN_RECORD,
            f"{len(removed)} records; raw range {min(rng.values()):.2f}-{max(rng.values()):.2f} s")
    jobs = [dict(x=pre[f]["x_train"], y=pre[f]["y_train"], std_from="std") for f in removed]
    tf = time.time()
    fits = RF.fit_jobs(jobs, n_jobs=a.n_jobs, verbose=0, blas_threads=1)
    print(f"{len(fits)} GPR fits in {time.time() - tf:.0f} s", flush=True)
    p154 = pools["-1.95"]
    Y185 = np.vstack([p154.Y, np.array([f.mu for f in fits])])
    S185 = np.vstack([p154.S, np.array([f.sd for f in fits])])
    t185 = np.concatenate([tends["-1.95"], [tmax[f] for f in removed]])
    w185, nit, conv, _ = C.curve_weights(Y185, S185)
    C.check("weights of the 185-curve pool converged", conv, f"{nit} iterations")

    p215 = pools["-2.15"]
    specs = (("-1.95", f"-1.95 V, {p154.m} pipeline curves", p154.Y, p154.S, ws["-1.95"], tends["-1.95"]),
             ("-1.95", f"-1.95 V, {len(w185)} curves with the records shorter than 60 s", Y185, S185, w185, t185),
             ("-2.15", f"-2.15 V, {p215.m} pipeline curves", p215.Y, p215.S, ws["-2.15"], tends["-2.15"]))
    grids, meds, rows = [], [], []
    for pot, label, Y, S, w, tend in specs:
        g, m, r = pool_terms(pot, label, Y, S, w, tend)
        grids.append(g)
        meds.append(m)
        rows.append(r)
        mm = m.set_index("notes").share_within_total_median
        print(f"{label}: median within share 10-100 / 100-1000 / 1000-5000 s {100 * mm['10-100 s']:.1f} / "
              f"{100 * mm['100-1000 s']:.1f} / {100 * mm['1000-5000 s']:.1f} %", flush=True)

    C.write_tables(__file__, {
        "band_terms_on_grid": pd.concat(grids, ignore_index=True),
        "band_share_window_medians": pd.concat(meds, ignore_index=True),
        "curve_groups": pd.DataFrame(rows),
    })
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
