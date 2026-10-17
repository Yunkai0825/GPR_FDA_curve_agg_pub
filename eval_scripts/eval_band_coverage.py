"""Coverage of the curves by the GPR-FDA band: weighted and unweighted, in sample and leave one batch out, and the
unweighted head counts of all curve points at the multi-curve potentials.

Band. For curves r with normalized posterior means y_r(t), posterior sd s_r(t) and converged curve weights w_r (curve-wise
iterative weighting in normalized current, tolerance 1e-6, epsilon 1e-12), the aggregate mean is
ybar(t) = sum_r w_r y_r(t) and the band is sigma(t) = sqrt(sum_r w_r (y_r(t) - ybar(t))^2 + sum_r w_r s_r(t)^2).
A curve point is inside the +-k sigma band when |y_r(t) - ybar(t)| <= k sigma(t), k = 1, 2.

Coverage at -1.95 V (154 curves, pipeline posteriors in _output/CCNF_CTAB_PT_output/individual_GPR, normalized current,
500-point grid log10 t = linspace(-1, 3.69897, 500)):
  in sample             every curve against the band and mean of the aggregate of all 154 curves;
  leave one batch out   (lobo) each of the 15 batches (catholyte batch, 'acidB(<batch>)' in the file name) is removed in
                        turn, the curve weights are re-iterated on the curves of the other batches, and the removed curves
                        are counted against the band and mean of that aggregate (every curve is held out once).
  weighted share at t   sum_r w_r 1{inside} with the converged weights w_r of the 154-curve aggregate;
  unweighted share at t the plain share of the 154 curves;
  Chebyshev floor       max(0, 1 - B(t)/(k^2 sigma(t)^2)) with B(t) = sum_r w_r (y_r(t) - ybar(t))^2, a lower bound of the
                        weighted in-sample share;
  grid means            per curve, c_r = share of the 500 grid points inside; weighted mean = sum_r w_r c_r with the
                        converged weights of the pipeline file Converged_Weights_potential_-1.95.csv normalized to sum 1;
                        unweighted mean = mean_r c_r.
  Gaussian reference    100 erf(k / sqrt(2)): share of equally weighted Gaussian scatter inside +-k sigma.

Head counts. For each multi-curve potential (-1.75, -1.85, -1.95, -2.05, -2.15 V) the pipeline sigma-calibration files
(batch_testing/NS_norm__WM_iterative__WS_curve/summary_gpr/iterative/Sigma_Calibration_potential_<P>.csv for GPR-FDA,
batch_testing/NS_norm__AM_student_t/summary_gpr/student_t/StudentT_Sigma_Calibration_potential_<P>.csv for the
hierarchical Student-t model with its predictive sd) hold every curve on the grid with the mean and sd of the model.
Head count = number of (curve, grid point) pairs with -k sd(t) <= y_r(t) - mean(t) <= k sd(t), counted once per pair, in
normalized current and in current (A cm^-2 on the nominal 1 cm2 area). The deviations y_r(t) - mean(t) and the sd of both
models are tabulated in normalized current at -2.15 V and -1.95 V.

Nothing is refitted; inputs are the pipeline outputs in _output/CCNF_CTAB_PT_output.

Output tables in output/band_coverage/ (coverage in %):
  coverage_by_grid_point   t_s, log10_t, and for k = 1, 2 (<k> = 1sigma, 2sigma): in_sample_weighted_<k>_pct,
                           in_sample_unweighted_<k>_pct, lobo_weighted_<k>_pct, lobo_unweighted_<k>_pct,
                           chebyshev_floor_<k>_pct; -1.95 V, normalized current
  coverage_summary         k, gaussian_reference_pct, in_sample_weighted_pct, in_sample_unweighted_pct, lobo_weighted_pct,
                           lobo_unweighted_pct (grid means), in_sample_weighted_min_pct, lobo_weighted_min_pct (minimum
                           over the grid), n_curves, n_batches
  headcount_by_potential   potential_v, student_t (0 = GPR-FDA, 1 = hierarchical Student-t), normalized_current
                           (1 = normalized current, 0 = current), n_curves, n_points (curves x grid points), inside_1sigma,
                           inside_2sigma (counts), inside_1sigma_pct, inside_2sigma_pct, notes = model and scale
  calibration_band         potential_v, t_s, log10_t, gprfda_sd_norm, student_t_sd_norm (sd of the band, normalized current;
                           -2.15 V and -1.95 V)
  calibration_deviation    potential_v, curve (position in the Converged_Weights order), t_s, log10_t,
                           gprfda_deviation_norm, student_t_deviation_norm (y_r(t) - mean(t) of each model, normalized
                           current; -2.15 V and -1.95 V)

Run
    python -X utf8 -B eval_scripts/eval_band_coverage.py
"""
import sys

sys.dont_write_bytecode = True

import math
import re
import time

import numpy as np
import pandas as pd

import _eval_common as C

POT = "-1.95"
KS = (1, 2)
N_BATCHES = 15
CALIB_POTENTIALS = ("-2.15", "-1.95")
ST_DIR = C.BT / "NS_norm__AM_student_t" / "summary_gpr" / "student_t"
MODELS = (("GPR-FDA", 0, C.MAIN, "Sigma_Calibration", "mean", "std"),
          ("hierarchical Student-t", 1, ST_DIR, "StudentT_Sigma_Calibration", "student_t_mean", "student_t_std_pred"))
SCALES = (("normalized current", 1, "norm"), ("current", 0, "real"))


def inside_band(Y, mean, band, k):
    """(curves x grid) boolean |y_r(t) - mean(t)| <= k band(t)."""
    return np.abs(Y - mean) <= k * band


def calibration_file(folder, prefix, pot):
    """Pipeline sigma-calibration file of one potential, read at full precision."""
    return pd.read_csv(folder / f"{prefix}_potential_{C.pkey(pot)}.csv", float_precision="round_trip")


def samples(d, space):
    """(curves x grid) curve values sample_<i>_<space> in column order i = 0, 1, ..."""
    cols = [c for c in d.columns if re.fullmatch(rf"sample_\d+_{space}", c)]
    if cols != [f"sample_{i}_{space}" for i in range(len(cols))]:
        raise ValueError("unexpected sample column order")
    return d[cols].to_numpy(float).T


def head_count(Y, mean, sd, k):
    """Number of curve points with -k sd <= y - mean <= k sd, and their share of all points."""
    dev = Y - mean
    ins = (dev >= -k * sd) & (dev <= k * sd)
    return int(ins.sum()), float(np.mean(ins))


def main():
    t_start = time.time()
    pool = C.load_pool(POT)
    R, N = pool.m, C.N_GRID
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)
    summ = C.load_summary(POT)

    g = C.aggregate(pool, "gprfda")
    w = g.w
    C.check("full-pool weights reproduce Converged_Weights", np.abs(w - w_pipeline).max() < 1e-12,
            f"max |dw| {np.abs(w - w_pipeline).max():.2g}")
    rel_band = np.abs(g.band_norm / summ.Std_normalised.to_numpy(float) - 1).max()
    rel_mean = np.abs(g.mean_norm / summ.y_normalised.to_numpy(float) - 1).max()
    C.check("full-pool band and mean reproduce Summary_GPR (normalized)", max(rel_band, rel_mean) < 1e-10,
            f"max rel diff band {rel_band:.2g}, mean {rel_mean:.2g}")

    batch = C.pool_meta(pool.sids).batch.to_numpy()
    batches = sorted(np.unique(batch))
    C.check("batch label in every file name", not (batch == "?").any() and len(batches) == N_BATCHES,
            f"{len(batches)} batches")

    ins_in = {k: inside_band(pool.Y, g.mean_norm, g.band_norm, k) for k in KS}
    ins_lobo = {k: np.zeros((R, N), bool) for k in KS}
    held = np.zeros(R, int)
    converged = []
    for b in batches:
        out_i = np.flatnonzero(batch == b)
        r = C.aggregate(pool, "gprfda", np.flatnonzero(batch != b))
        converged.append(r.converged)
        held[out_i] += 1
        for k in KS:
            ins_lobo[k][out_i] = inside_band(pool.Y[out_i], r.mean_norm, r.band_norm, k)
    C.check("every leave-one-batch-out aggregate converged", all(converged), f"{len(converged)} aggregates")
    C.check("every curve held out exactly once", bool((held == 1).all()))

    between = (w[:, None] * (pool.Y - g.mean_norm) ** 2).sum(0)
    share_between = between / g.band_norm ** 2
    grid = pd.DataFrame(dict(t_s=C.T_GRID, log10_t=C.GRID))
    wn_pipeline = w_pipeline / w_pipeline.sum()
    srows = []
    for k in KS:
        floor = np.maximum(0.0, 1.0 - share_between / k ** 2)
        in_w, lobo_w = w @ ins_in[k], w @ ins_lobo[k]
        grid[f"in_sample_weighted_{k}sigma_pct"] = 100 * in_w
        grid[f"in_sample_unweighted_{k}sigma_pct"] = 100 * ins_in[k].mean(0)
        grid[f"lobo_weighted_{k}sigma_pct"] = 100 * lobo_w
        grid[f"lobo_unweighted_{k}sigma_pct"] = 100 * ins_lobo[k].mean(0)
        grid[f"chebyshev_floor_{k}sigma_pct"] = 100 * floor
        C.check(f"weighted in-sample share >= Chebyshev floor at every grid point, k = {k}",
                bool((in_w >= floor - 1e-12).all()), f"min margin {(in_w - floor).min():.3g}")
        c_in = np.array([ins_in[k][i].mean() for i in range(R)])
        c_lobo = np.array([ins_lobo[k][i].mean() for i in range(R)])
        srows.append(dict(k=k, gaussian_reference_pct=100 * math.erf(k / math.sqrt(2)),
                          in_sample_weighted_pct=100 * np.sum(wn_pipeline * c_in), in_sample_unweighted_pct=100 * c_in.mean(),
                          lobo_weighted_pct=100 * np.sum(wn_pipeline * c_lobo), lobo_unweighted_pct=100 * c_lobo.mean(),
                          in_sample_weighted_min_pct=100 * in_w.min(), lobo_weighted_min_pct=100 * lobo_w.min(),
                          n_curves=R, n_batches=len(batches)))
    summary = pd.DataFrame(srows)

    hrows, band_rows, dev_rows = [], [], []
    for pot in C.MULTI_CURVE_POTENTIALS:
        files = {code: calibration_file(folder, prefix, pot) for _, code, folder, prefix, _, _ in MODELS}
        Ys = {(code, sp): samples(files[code], sp) for _, code, *_ in MODELS for _, _, sp in SCALES}
        for mlabel, code, *_ in MODELS:
            x = files[code].x_real.to_numpy(float)
            C.check(f"{mlabel} calibration grid at {pot} V is the shared grid",
                    np.abs(x / C.T_GRID - 1).max() < 1e-12, f"max rel diff {np.abs(x / C.T_GRID - 1).max():.2g}")
        for _, _, sp in SCALES:
            a, b = Ys[(0, sp)], Ys[(1, sp)]
            d = np.abs(b - a).max() / np.abs(a).max()
            C.check(f"both models hold the same curves in the same order at {pot} V ({sp})", d < 1e-12,
                    f"max rel diff {d:.2g}")
        p = pool if pot == POT else C.load_pool(pot)
        d = np.abs(Ys[(0, "norm")] - p.Y).max(axis=1) / np.abs(p.Y).max(axis=1)
        C.check(f"calibration curves at {pot} V are the pipeline posteriors in Converged_Weights order", d.max() < 1e-9,
                f"max rel diff per curve {d.max():.2g}")
        for mlabel, code, _, _, mcol, scol in MODELS:
            fr = files[code]
            for slabel, nflag, sp in SCALES:
                Y = Ys[(code, sp)]
                mean, sd = fr[f"{mcol}_{sp}"].to_numpy(float), fr[f"{scol}_{sp}"].to_numpy(float)
                (n1, f1), (n2, f2) = head_count(Y, mean, sd, 1), head_count(Y, mean, sd, 2)
                hrows.append(dict(potential_v=float(pot), student_t=code, normalized_current=nflag, n_curves=Y.shape[0],
                                  n_points=Y.size, inside_1sigma=n1, inside_2sigma=n2, inside_1sigma_pct=100 * f1,
                                  inside_2sigma_pct=100 * f2, notes=f"{mlabel}, {slabel}"))
                if pot == POT and code == 0 and sp == "norm":
                    for k, n in ((1, n1), (2, n2)):
                        C.check(f"head count equals the unweighted in-sample count at {POT} V, k = {k}",
                                n == int(ins_in[k].sum()), f"{n} of {Y.size}")
                    rb = np.abs(sd / g.band_norm - 1).max()
                    C.check(f"calibration sd at {POT} V is the band of the 154-curve aggregate", rb < 1e-9,
                            f"max rel diff {rb:.2g}")
        if pot in CALIB_POTENTIALS:
            dev = {code: Ys[(code, "norm")] - files[code][f"{mcol}_norm"].to_numpy(float)
                   for _, code, _, _, mcol, _ in MODELS}
            band_rows.append(pd.DataFrame(dict(potential_v=float(pot), t_s=C.T_GRID, log10_t=C.GRID,
                                               gprfda_sd_norm=files[0]["std_norm"].to_numpy(float),
                                               student_t_sd_norm=files[1]["student_t_std_pred_norm"].to_numpy(float))))
            m = dev[0].shape[0]
            dev_rows.append(pd.DataFrame(dict(potential_v=float(pot), curve=np.repeat(np.arange(m), N),
                                              t_s=np.tile(C.T_GRID, m), log10_t=np.tile(C.GRID, m),
                                              gprfda_deviation_norm=dev[0].ravel(),
                                              student_t_deviation_norm=dev[1].ravel())))
    heads = pd.DataFrame(hrows)
    band = pd.concat(band_rows, ignore_index=True)
    deviation = pd.concat(dev_rows, ignore_index=True)

    C.write_tables(__file__, dict(coverage_by_grid_point=grid, coverage_summary=summary, headcount_by_potential=heads,
                                  calibration_band=band, calibration_deviation=deviation))
    print(f"done in {time.time() - t_start:.0f} s; {C.checks_failed()} failed checks", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
