"""Convergence of the iterative weights: curve-wise (GPR-FDA) against point-wise weighting.

Method
  Inputs: the individual GPR posteriors of individual_GPR/Individual_GPR_<sample>.csv (normalized mean Y and sd S
  on the 500-point log10 t grid, scale factor s), in the record order of Converged_Weights of each method, at the
  five potentials with more than one record (-1.75, -1.85, -1.95, -2.05, -2.15 V).
  Curve-wise weights (GPR-FDA): w = 1/R; repeat: ybar = w @ Y, D_r = mean_j (Y_r - ybar)^2,
  w_new = 1/(D_r + mean_j S_r^2 + 1e-12) normalized to sum 1; RMS change ||w_new - w||/sqrt(R); stop below 1e-6.
  Point-wise weights: W = 1/R at every grid point; repeat: Ibar = sum_r W_r Y_r, W = 1/((Y - Ibar)^2 + S^2 + 1e-12)
  normalized over the records at each grid point; RMS change over all R x 500 weights; stop below 1e-6, at most
  10000 updates.
  Aggregate at -1.85 V (4 records): normalized mean sum_r W_r Y_r; current = s_agg x normalized mean with
  s_agg = sum_r wbar_r s_r (wbar = grid-mean weights normalized to sum 1); band sigma = sqrt(sum_r W_r (Y_r - mean)^2
  + sum_r W_r S_r^2) on the current scale with Y_r s_r and S_r |s_r|; 95 % interval = mean -/+ 1.959963984540054
  sigma. The curve-wise weights, iterates and converged aggregate are computed with _eval_common.curve_weights and
  aggregate_arrays. Iterate n is the aggregate with the weights after n updates (n = 0: equal weights).
  Self-checks against the pipeline outputs in batch_testing/<method>/summary_gpr/iterative: curve-wise weights per
  iteration (Weight_History), converged weights, iterates (Curve_History) and the converged aggregate (Summary_GPR,
  including the 95 % interval) of NS_norm__WM_iterative__WS_curve; for NS_norm__WM_iterative__WS_point, whose outputs
  stop after 50 updates, the grid-mean weights per iteration (Weight_History), the grid-mean weights, iterates and
  aggregate after 50 updates (Converged_Weights, Curve_History, Summary_GPR).

Output tables in output/weight_convergence/ (currents in mA on the nominal 1 cm2 electrode area)
  rms_weight_change        potential_v, n_curves, iteration, gprfda_rms_change, pointwise_rms_change (RMS weight
                           change of update n; empty after the last update of a method)
  iterations_to_tolerance  potential_v, n_curves, gprfda_iterations, gprfda_converged, pointwise_iterations,
                           pointwise_converged
  gprfda_iterates          -1.85 V: weight_updates, t_s, log10_t, mean_norm (curve-wise aggregate after n updates)
  pointwise_iterates       -1.85 V: weight_updates, t_s, log10_t, mean_norm (point-wise aggregate after n updates)
  converged_aggregates     -1.85 V: t_s, log10_t, gprfda_mean_norm, gprfda_mean_mA, gprfda_band_mA,
                           gprfda_lower95_mA, gprfda_upper95_mA, pointwise_mean_norm, pointwise_mean_mA,
                           pointwise_band_mA, pointwise_lower95_mA, pointwise_upper95_mA
  individual_posteriors    -1.85 V: curve (position in the Converged_Weights order), t_s, log10_t, mean_norm,
                           mean_mA (posterior mean of each record), notes = sample id

Run
  python -X utf8 -B eval_scripts/eval_weight_convergence.py
"""
import sys

sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd

import _eval_common as C

POTS = C.MULTI_CURVE_POTENTIALS
POT_DETAIL = "-1.85"
TOL = C.TOL
EPS = C.EPS
MAX_UPDATES = C.POINTWISE_MAX_ITER
PIPELINE_POINTWISE_UPDATES = C.PIPELINE_POINTWISE_CAP
Z95 = 1.959963984540054
CHECK_TOL = 1e-10


def curve_history(Y, S):
    """Curve-wise weights; returns final weights, RMS change per update, weights before each update, converged."""
    w, _, conv, hist = C.curve_weights(Y, S, tol=TOL, eps=EPS, history=True)
    start = np.array(hist)
    deltas = np.array([np.linalg.norm(b - a) / np.sqrt(len(w)) for a, b in zip(start, list(start[1:]) + [w])])
    return w, deltas, start, conv


def point_history(Y, S, keep, tol=TOL, max_iter=MAX_UPDATES):
    """Point-wise weights; returns final W, RMS change per update, grid-mean weights per step (initial row first),
    {n: W after n updates} for n in keep, converged."""
    R, N = Y.shape
    W = np.full((R, N), 1.0 / R)
    deltas, means, kept = [], [W.mean(axis=1)], {}
    for it in range(max_iter):
        Wp = W
        Ib = (W * Y).sum(axis=0)
        W = 1.0 / ((Y - Ib) ** 2 + S ** 2 + EPS)
        W = W / W.sum(axis=0, keepdims=True)
        delta = np.linalg.norm(W - Wp) / np.sqrt(W.size)
        deltas.append(delta)
        means.append(W.mean(axis=1))
        if it + 1 in keep:
            kept[it + 1] = W.copy()
        if delta < tol:
            return W, np.array(deltas), np.array(means), kept, True
    return W, np.array(deltas), np.array(means), kept, False


def pw_aggregate(W, Y, S, s):
    """Point-wise aggregate: normalized mean, current mean and band sigma on both scales."""
    m = (W * Y).sum(axis=0)
    wv = W.mean(axis=1)
    wv = wv / wv.sum()
    s_agg = wv @ s
    y = s_agg * m
    Yr, Sr = Y * s[:, None], S * np.abs(s)[:, None]
    band_n = np.sqrt((W * (Y - m) ** 2).sum(axis=0) + (W * S ** 2).sum(axis=0))
    band_r = np.sqrt((W * (Yr - y) ** 2).sum(axis=0) + (W * Sr ** 2).sum(axis=0))
    return dict(m=m, y=y, band_n=band_n, band_r=band_r)


def max_rel(a, b):
    """max |a - b| / max |b|."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def weight_columns(df):
    return df[[c for c in df.columns if c.startswith("weight_")]].to_numpy(float)


def main():
    t_start = time.time()
    rms_rows, it_rows = [], []
    detail = {}
    for pot in POTS:
        pc = C.load_pool(pot, "gprfda")
        w, dc, start, convc = curve_history(pc.Y, pc.S)
        wh_c = weight_columns(C.load_weight_history(pot, "gprfda"))
        wst_c = C.load_weights(pot, "gprfda").converged_weight.to_numpy(float)
        C.check(f"{pot} V curve-wise weights per iteration = Weight_History",
                start.shape == wh_c.shape and np.abs(start - wh_c).max() <= CHECK_TOL,
                f"{len(dc)} iterations, {len(wh_c)} rows")
        C.check(f"{pot} V curve-wise converged weights = Converged_Weights", np.abs(w - wst_c).max() <= CHECK_TOL,
                f"max abs {np.abs(w - wst_c).max():.2g}")

        pp = C.load_pool(pot, "pointwise")
        keep = set(range(1, PIPELINE_POINTWISE_UPDATES + 1))
        if pot == POT_DETAIL:
            keep = set(range(1, MAX_UPDATES + 1))
        Wu, dp, means, kept, convp = point_history(pp.Y, pp.S, keep)
        nit = len(dp)
        Wcap = kept.get(PIPELINE_POINTWISE_UPDATES, Wu)
        wh_p = weight_columns(C.load_weight_history(pot, "pointwise"))
        sm_p = C.load_summary(pot, "pointwise")
        A_cap = pw_aggregate(Wcap, pp.Y, pp.S, pp.F)
        d_sum = max(max_rel(A_cap["m"], sm_p.y_normalised), max_rel(A_cap["y"], sm_p.y_real),
                    max_rel(A_cap["band_n"], sm_p.Std_normalised), max_rel(A_cap["band_r"], sm_p.Std_real))
        ch_p = pd.read_csv(C.PW / f"Curve_History_potential_{pot}.csv", comment="#")
        n_ch = sum(c.startswith("iter_") for c in ch_p.columns)
        d_ch = max(max_rel(pw_aggregate(np.full(pp.Y.shape, 1.0 / pp.m) if n == 1 else kept[n - 1],
                                        pp.Y, pp.S, pp.F)["m"], ch_p[f"iter_{n}"])
                   for n in range(1, min(n_ch, nit + 1) + 1))
        C.check(f"{pot} V point-wise converged within {MAX_UPDATES} updates", convp, f"{nit} updates")
        C.check(f"{pot} V point-wise grid-mean weights per iteration = Weight_History",
                len(wh_p) <= len(means) and np.abs(means[:len(wh_p)] - wh_p).max() <= CHECK_TOL, f"{len(wh_p)} rows")
        C.check(f"{pot} V point-wise after {min(nit, PIPELINE_POINTWISE_UPDATES)} updates = Converged_Weights and "
                "Summary_GPR",
                np.abs(Wcap.mean(axis=1)
                       - C.load_weights(pot, "pointwise").converged_weight.to_numpy(float)).max() <= CHECK_TOL
                and d_sum <= CHECK_TOL, f"Summary_GPR max rel {d_sum:.2g}")
        C.check(f"{pot} V point-wise iterates = Curve_History iter_1..iter_{min(n_ch, nit + 1)}", d_ch <= CHECK_TOL,
                f"max rel {d_ch:.2g}")

        for i in range(max(len(dc), nit)):
            rms_rows.append(dict(potential_v=float(pot), n_curves=pc.m, iteration=i + 1,
                                 gprfda_rms_change=dc[i] if i < len(dc) else np.nan,
                                 pointwise_rms_change=dp[i] if i < nit else np.nan))
        it_rows.append(dict(potential_v=float(pot), n_curves=pc.m, gprfda_iterations=len(dc), gprfda_converged=convc,
                            pointwise_iterations=nit, pointwise_converged=convp))
        if pot == POT_DETAIL:
            detail = dict(pc=pc, pp=pp, Wu=Wu, kept=kept, nit=nit)
        print(f"{pot} V: {pc.m} records, curve-wise {len(dc)}, point-wise {nit} updates "
              f"({time.time() - t_start:.0f} s)", flush=True)

    pc, pp, Wu, kept, nit = (detail[k] for k in ("pc", "pp", "Wu", "kept", "nit"))
    t, x = pc.t, pc.x
    C.check("both methods use the same records and grid at -1.85 V",
            pc.sids == pp.sids and np.array_equal(pc.t, pp.t) and np.array_equal(pc.Y, pp.Y))

    hist_c = C.curve_weights(pc.Y, pc.S, history=True)[3]
    g = C.aggregate_arrays(pc.Y, pc.S, pc.F, "gprfda")
    iter_c = [(h[:, None] * pc.Y).sum(axis=0) for h in hist_c] + [g.mean_norm]
    ch_c = pd.read_csv(C.MAIN / f"Curve_History_potential_{POT_DETAIL}.csv", comment="#")
    n_chc = sum(c.startswith("iter_") for c in ch_c.columns)
    C.check("-1.85 V curve-wise iterates = Curve_History", n_chc == len(hist_c) and
            max(max_rel(iter_c[n - 1], ch_c[f"iter_{n}"]) for n in range(1, n_chc + 1)) <= CHECK_TOL,
            f"iter_1..iter_{n_chc}")
    sm_c = C.load_summary(POT_DETAIL, "gprfda")
    g_lo, g_hi = g.mean_real - Z95 * g.band_real, g.mean_real + Z95 * g.band_real
    d_c = max(max_rel(g.mean_norm, sm_c.y_normalised), max_rel(g.mean_real, sm_c.y_real),
              max_rel(g.band_real, sm_c.Std_real), max_rel(g_lo, sm_c.Lower_CI_real), max_rel(g_hi, sm_c.Upper_CI_real))
    C.check("-1.85 V curve-wise converged aggregate and 95 % interval = Summary_GPR", d_c <= CHECK_TOL,
            f"max rel {d_c:.2g}")

    A = pw_aggregate(Wu, pp.Y, pp.S, pp.F)
    p_lo, p_hi = A["y"] - Z95 * A["band_r"], A["y"] + Z95 * A["band_r"]

    gi = pd.concat([pd.DataFrame(dict(weight_updates=n, t_s=t, log10_t=x, mean_norm=m)) for n, m in enumerate(iter_c)],
                   ignore_index=True)
    starts = [np.full(pp.Y.shape, 1.0 / pp.m)] + [kept[n] for n in range(1, nit + 1)]
    pi = pd.concat([pd.DataFrame(dict(weight_updates=n, t_s=t, log10_t=x, mean_norm=pw_aggregate(W_, pp.Y, pp.S, pp.F)["m"]))
                    for n, W_ in enumerate(starts)], ignore_index=True)
    C.check("last point-wise iterate = converged point-wise aggregate",
            np.array_equal(pi[pi.weight_updates == nit].mean_norm.to_numpy(), A["m"]))
    agg = pd.DataFrame(dict(t_s=t, log10_t=x, gprfda_mean_norm=g.mean_norm, gprfda_mean_mA=g.mean_real * 1e3,
                            gprfda_band_mA=g.band_real * 1e3, gprfda_lower95_mA=g_lo * 1e3, gprfda_upper95_mA=g_hi * 1e3,
                            pointwise_mean_norm=A["m"], pointwise_mean_mA=A["y"] * 1e3,
                            pointwise_band_mA=A["band_r"] * 1e3, pointwise_lower95_mA=p_lo * 1e3,
                            pointwise_upper95_mA=p_hi * 1e3))
    ind = pd.concat([pd.DataFrame(dict(curve=i, t_s=t, log10_t=x, mean_norm=pc.Y[i], mean_mA=pc.Y[i] * pc.F[i] * 1e3,
                                       notes=pc.sids[i])) for i in range(pc.m)], ignore_index=True)

    C.write_tables(__file__, dict(rms_weight_change=pd.DataFrame(rms_rows), iterations_to_tolerance=pd.DataFrame(it_rows),
                                  gprfda_iterates=gi, pointwise_iterates=pi, converged_aggregates=agg,
                                  individual_posteriors=ind))
    print(f"done in {time.time() - t_start:.0f} s; {C.checks_failed()} failed checks", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
