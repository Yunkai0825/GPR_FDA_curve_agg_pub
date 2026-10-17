"""Controlled contamination of the aggregate at -1.95 V: flagged records removed, and copies of recorded curves added.

Pool
    The 154 pipeline posteriors at -1.95 V (_output/CCNF_CTAB_PT_output/individual_GPR: normalized posterior mean and
    sd on the grid log10 t = linspace(-1, 3.69897, 500), t = 10^x - 1e-9 s, and the scale factor s_r of each curve).
    Nothing is refitted and no curve is simulated: contamination is made of extra copies of recorded curves.

Record classes
    contact loss       the 3 records with the smallest |scale factor|; they are the records with pipeline GPR-FDA
                       weight < 1e-8
    complete           record end >= 2990 s; the record end is the last sample of the raw record (_input/CCNF_CTAB_PT)
                       within 1e-4 <= t <= 5000 s
    magnitude outlier  complete, no contact loss, and |modified z| > 3.5, with modified z = 0.6745 (s_r - median) / MAD
                       of the scale factors of the 105 complete records without contact loss

Aggregators
    Each pool is aggregated from equal starting weights by GPR-FDA (curve-wise iterative weights, tolerance 1e-6,
    eps 1e-12), point-wise iterative weights (run to convergence, at most 10000 iterations), the direct average of the
    currents and the direct average of the normalized currents. The current-scale mean of the normalized-space methods
    is the normalized mean times the weighted scale factor sum_r w_r s_r; the band in current is
    sqrt(sum_r w_r (Y_r s_r - mean)^2 + sum_r w_r (S_r s_r)^2). Every shift is (mean of a pool - mean of the same method
    on its reference pool) in current, divided point by point by a GPR-FDA band; max and RMS run over the 500 grid
    points.

Tests
    contact-loss copies    the 151 curves without contact loss plus k copies of each contact-loss curve, k = 1..22
                           (copy share 3k / (151 + 3k)); reference = the 151 curves; denominator = GPR-FDA band of the
                           151 curves. Point-wise weighting is run once more from the point-wise solution of the 151
                           curves (copies at 1e-12, renormalized per grid point) as clean start. Weight on the copies:
                           sum of the curve weights (GPR-FDA, direct averages) or, for point-wise weights, the sum over
                           the copies at each grid point, averaged over and maximized over the grid.
    removal                154 vs 151 (contact loss removed), 154 vs 106 (records ending before 2990 s removed) and 151
                           vs 105 (the same records removed from the 151 curves); the point-wise run on the 154 curves is
                           also started from the solution of the 151 curves.
    outlier copies         the 150 other curves without contact loss plus round(p 150 / (1 - p)) copies of the magnitude
                           outlier, p = 2, 5, 10, 20, 30 % (3, 8, 17, 38 and 64 copies); reference = the 150 curves.
                           Removal and outlier shifts are divided by the band of the pipeline GPR-FDA aggregate of the
                           154 curves (Summary_GPR Std_real).
    contamination checks   the removal of the contact-loss records, the removal of the records ending before 2990 s, 22
                           copies of each contact-loss curve and 64 copies of the outlier, each against its reference
                           pool, all divided by the GPR-FDA band of the 154 curves.

Output tables (eval_scripts/output/controlled_contamination/; currents in mA = A cm^-2 x 1e3 on the nominal 1 cm2 area)
    record_classes                       one row per curve: curve (position in Converged_Weights), scale_factor_mA,
                                         record_end_s, contact_loss, complete (0/1), scale_factor_over_median (over the
                                         median of the 105 complete records without contact loss), modified_z,
                                         magnitude_outlier (0/1), notes = sample id
    aggregate_with_contact_loss_copies   t_s, log10_t, reference_mean_mA, reference_band_mA (pipeline GPR-FDA aggregate
                                         of the 154 curves, Summary_GPR), <method>_mean_mA (151 curves + 13 copies of each
                                         contact-loss curve), <method>_abs_shift_over_band (|mean - mean of the 151
                                         curves| over reference_band_mA); method = gprfda, pointwise, direct_obs,
                                         direct_norm
    shift_by_contact_loss_copies         copies_per_curve, n_copies, copy_share_pct, gprfda_max, gprfda_rms, pointwise_max,
                                         pointwise_rms, direct_obs_max, direct_obs_rms, direct_norm_max,
                                         direct_norm_rms (over the GPR-FDA band of the 151 curves),
                                         pointwise_iterations, pointwise_converged, n_reference, n_variant, notes
    copy_weight_by_contact_loss_copies   copies_per_curve, n_copies, copy_share_pct, gprfda_copy_weight,
                                         pointwise_copy_weight_mean, pointwise_copy_weight_max (over the grid),
                                         direct_copy_weight
    clean_start_by_contact_loss_copies   copies_per_curve, n_copies, copy_share_pct, pointwise_max (over the GPR-FDA band
                                         of the 151 curves), pointwise_copy_weight_mean, pointwise_iterations,
                                         pointwise_converged
    shift_by_removal_and_outlier_copies  variant, n_copies, copy_share_pct, the eight shift columns (over the pipeline
                                         GPR-FDA band of the 154 curves), pointwise_iterations, pointwise_converged,
                                         n_reference, n_variant, notes
    clean_start_after_removal            pointwise_max_over_clean_band (154 curves started from the 151-curve
                                         solution, over the GPR-FDA band of the 151 curves),
                                         clean_band_ratio_deviation_max (max |band of the 151 curves / band of the 154
                                         curves - 1|), pointwise_max_over_band_bound (the product
                                         pointwise_max_over_clean_band (1 + clean_band_ratio_deviation_max), an upper
                                         bound of the shift over the 154-curve band), pointwise_max_over_band (the same
                                         shift over the 154-curve band), pointwise_iterations, pointwise_converged,
                                         n_reference, n_variant, notes
    shift_by_contamination_check         variant, n_copies, copy_share_pct, the eight shift columns (over the GPR-FDA
                                         band of the 154 curves), pointwise_iterations, pointwise_converged,
                                         n_reference, n_variant, notes

Run
    python -X utf8 -B eval_scripts/eval_controlled_contamination.py
"""
import sys

sys.dont_write_bytecode = True

import argparse
import time

import numpy as np
import pandas as pd

import _eval_common as C

POT = "-1.95"
K_SCAN = tuple(range(1, 23))
K_TIME = 13
K_CHECK = 22
OUTLIER_SHARES = (0.02, 0.05, 0.10, 0.20, 0.30)
OUTLIER_CHECK = 0.30
MZ_CUT = 3.5
FLOOR = 1e-12


def clean_start(pool, idx, rows, W_clean):
    """Point-wise weighting of the curves idx started from W_clean on `rows` and 1e-12 elsewhere, renormalized."""
    W0 = np.full((len(idx), C.N_GRID), FLOOR)
    W0[rows] = W_clean
    W0 /= W0.sum(axis=0, keepdims=True)
    return C.aggregate(pool, "pointwise", idx, cap=C.POINTWISE_MAX_ITER, W0=W0)


def copy_weight(a, mask):
    """Weight on the copy rows: (sum, sum) of curve weights, or (mean, max) over the grid of the point-wise sums."""
    if a.W.ndim == 1:
        s = float(a.W[mask].sum())
        return s, s
    p = a.W[mask].sum(axis=0)
    return float(p.mean()), float(p.max())


def max_rel(a, b):
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1)))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    ALL = np.arange(R)
    st = C.load_summary(POT)
    m154, s154 = st.y_real.to_numpy(float), st.Std_real.to_numpy(float)
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)

    res_all = C.aggregate_all(pool, ALL)
    d = 0.0
    for m in ("gprfda", "direct_obs", "direct_norm"):
        sm = C.load_summary(POT, m)
        d = max([d] + [max_rel(getattr(res_all[m], k), sm[c]) for c, k in
                       (("y_real", "mean_real"), ("Std_real", "band_real"), ("y_normalised", "mean_norm"),
                        ("Std_normalised", "band_norm"))])
    C.check("GPR-FDA and direct averages of the 154 curves reproduce Summary_GPR", d <= 1e-10, f"max relative {d:.2g}")
    pc50 = C.aggregate(pool, "pointwise", cap=C.PIPELINE_POINTWISE_CAP)
    sp = C.load_summary(POT, "pointwise")
    d = max(max_rel(getattr(pc50, k), sp[c]) for c, k in (("y_real", "mean_real"), ("Std_real", "band_real"),
                                                          ("y_normalised", "mean_norm"), ("Std_normalised", "band_norm")))
    C.check("point-wise weighting capped at 50 iterations reproduces the point-wise Summary_GPR", d <= 1e-10,
            f"max relative {d:.2g}")

    CL = C.contact_loss_indices(pool)
    C.check("contact-loss records are the records with pipeline weight < 1e-8",
            np.array_equal(CL, np.flatnonzero(w_pipeline < 1e-8)), f"positions {CL.tolist()}")
    tend = C.record_ends(pool.sids)
    TR = np.flatnonzero(tend < C.T_COMPLETE)
    CLEAN = np.setdiff1d(ALL, CL)
    COMPLETE = np.setdiff1d(ALL, TR)
    COMPLETE_CLEAN = np.setdiff1d(CLEAN, TR)
    C.check("record classes", len(TR) == 48 and len(COMPLETE) == 106 and len(CLEAN) == 151 and len(COMPLETE_CLEAN) == 105,
            f"{len(TR)} records end before 2990 s; pools {len(COMPLETE)}, {len(CLEAN)}, {len(COMPLETE_CLEAN)}")
    fc = pool.F[COMPLETE_CLEAN]
    med = np.median(fc)
    mad = np.median(np.abs(fc - med))
    mz = 0.6745 * (pool.F - med) / mad
    OUTL = np.array([i for i in COMPLETE_CLEAN if abs(mz[i]) > MZ_CUT])
    C.check("one magnitude outlier", len(OUTL) == 1,
            f"positions {OUTL.tolist()}, modified z {[round(float(mz[i]), 3) for i in OUTL]}, "
            f"scale factor / median {[round(float(pool.F[i] / med), 3) for i in OUTL]}")
    BASE = np.setdiff1d(CLEAN, OUTL)
    classes = pd.DataFrame(dict(curve=ALL, scale_factor_mA=pool.F * 1e3, record_end_s=tend, contact_loss=np.isin(ALL, CL),
                                complete=tend >= C.T_COMPLETE, scale_factor_over_median=pool.F / med, modified_z=mz,
                                magnitude_outlier=np.isin(ALL, OUTL), notes=pool.sids))

    res_clean = C.aggregate_all(pool, CLEAN)
    res_complete = C.aggregate_all(pool, COMPLETE)
    res_cc = C.aggregate_all(pool, COMPLETE_CLEAN)
    res_base = C.aggregate_all(pool, BASE)
    band151 = res_clean["gprfda"].band_real
    W_clean = res_clean["pointwise"].W
    converged = [res_all["pointwise"].converged, res_clean["pointwise"].converged, res_complete["pointwise"].converged,
                 res_cc["pointwise"].converged, res_base["pointwise"].converged]
    print(f"reference pools done ({time.time() - t0:.0f} s)", flush=True)

    scan, weights, starts = [], [], []
    res_k = {}
    for k in K_SCAN:
        idx = np.concatenate([CLEAN, np.tile(CL, k)])
        mask = np.r_[np.zeros(len(CLEAN), bool), np.ones(3 * k, bool)]
        frac = 3 * k / len(idx)
        r = C.aggregate_all(pool, idx)
        cs = clean_start(pool, idx, slice(0, len(CLEAN)), W_clean)
        converged.append(r["pointwise"].converged)
        par = dict(copies_per_curve=k, n_copies=3 * k, copy_share_pct=100 * frac)
        scan.append(dict(par, **C.shift_summary(res_clean, r, band151),
                         notes=f"151 curves + {k} copies of each contact-loss curve"))
        pw_mean, pw_max = copy_weight(r["pointwise"], mask)
        weights.append(dict(par, gprfda_copy_weight=copy_weight(r["gprfda"], mask)[0], pointwise_copy_weight_mean=pw_mean,
                            pointwise_copy_weight_max=pw_max, direct_copy_weight=copy_weight(r["direct_obs"], mask)[0]))
        z = (cs.mean_real - res_clean["pointwise"].mean_real) / band151
        starts.append(dict(par, pointwise_max=float(np.abs(z).max()), pointwise_copy_weight_mean=copy_weight(cs, mask)[0],
                           pointwise_iterations=cs.iterations, pointwise_converged=cs.converged))
        if k in (1, K_TIME, K_CHECK):
            res_k[k] = r
        print(f"  k = {k:2d} ({100 * frac:5.2f} %): point-wise {r['pointwise'].iterations} iterations, clean start "
              f"{cs.iterations} ({time.time() - t0:.0f} s)", flush=True)
    scan, weights, starts = pd.DataFrame(scan), pd.DataFrame(weights), pd.DataFrame(starts)
    d1 = max(abs(C.shift_summary(res_clean, res_k[1], band151)[f"{m}_max"] /
                 C.shift_summary(res_clean, res_all, band151)[f"{m}_max"] - 1) for m in C.METHODS)
    C.check("151 curves + 1 copy of each contact-loss curve = the 154 curves in another order", d1 < 1e-6,
            f"max relative deviation of the max shifts {d1:.2g}")
    C.check("clean-start runs converged", bool(starts.pointwise_converged.all()),
            f"iterations {sorted(set(starts.pointwise_iterations))}")

    rk = res_k[K_TIME]
    grid = pd.DataFrame(dict(t_s=pool.t, log10_t=C.GRID, reference_mean_mA=1e3 * m154, reference_band_mA=1e3 * s154))
    for m in C.METHODS:
        grid[f"{m}_mean_mA"] = 1e3 * rk[m].mean_real
    for m in C.METHODS:
        grid[f"{m}_abs_shift_over_band"] = np.abs(rk[m].mean_real - res_clean[m].mean_real) / s154

    other, res_mo = [], None
    for lab, ref, var in (("3 contact-loss records removed (154 vs 151)", res_clean, res_all),
                          ("48 records ending before 2990 s removed (154 vs 106)", res_complete, res_all),
                          ("46 records ending before 2990 s removed from the 151 curves (151 vs 105)", res_cc, res_clean)):
        other.append(dict(variant=len(other), n_copies=0, copy_share_pct=0.0, **C.shift_summary(ref, var, s154), notes=lab))
    n_copies = []
    for p in OUTLIER_SHARES:
        ncopy = int(round(p * len(BASE) / (1 - p)))
        idx = np.concatenate([BASE, np.tile(OUTL, int(np.ceil(ncopy / len(OUTL))))[:ncopy]])
        frac = ncopy / len(idx)
        r = C.aggregate_all(pool, idx)
        converged.append(r["pointwise"].converged)
        n_copies.append(ncopy)
        other.append(dict(variant=len(other), n_copies=ncopy, copy_share_pct=100 * frac,
                          **C.shift_summary(res_base, r, s154),
                          notes=f"150 curves + {ncopy} copies of the magnitude outlier ({100 * frac:.1f} %)"))
        if p == OUTLIER_CHECK:
            res_mo, frac_mo = r, frac
    other = pd.DataFrame(other)
    C.check("outlier copies", n_copies == [3, 8, 17, 38, 64], f"{n_copies}")
    C.check("every point-wise run converged", all(converged), f"{len(converged)} runs")

    W0 = np.full((R, C.N_GRID), FLOOR)
    W0[CLEAN] = W_clean
    W0 /= W0.sum(axis=0, keepdims=True)
    cs154 = C.aggregate(pool, "pointwise", cap=C.POINTWISE_MAX_ITER, W0=W0)
    dz = cs154.mean_real - res_clean["pointwise"].mean_real
    v_clean = float(np.abs(dz / band151).max())
    rel = max_rel(band151, s154)
    removal_start = pd.DataFrame([dict(pointwise_max_over_clean_band=v_clean, clean_band_ratio_deviation_max=rel,
                                       pointwise_max_over_band_bound=v_clean * (1 + rel),
                                       pointwise_max_over_band=float(np.abs(dz / s154).max()),
                                       pointwise_iterations=cs154.iterations, pointwise_converged=cs154.converged,
                                       n_reference=len(CLEAN), n_variant=R,
                                       notes="154 curves, point-wise weighting started from the 151-curve solution")])

    den = res_all["gprfda"].band_real
    dw = float(np.abs(res_all["gprfda"].w - w_pipeline).max())
    C.check("GPR-FDA weights of the 154 curves reproduce Converged_Weights", dw <= 1e-12, f"max abs deviation {dw:.2g}")
    checks = []
    for lab, nc, share, ref, var in (
            ("records ending before 2990 s removed (154 vs 106)", 0, 0.0, res_complete, res_all),
            ("3 contact-loss records removed (154 vs 151)", 0, 0.0, res_clean, res_all),
            (f"151 curves + {K_CHECK} copies of each contact-loss curve", 3 * K_CHECK,
             100 * (3 * K_CHECK / (len(CLEAN) + 3 * K_CHECK)), res_clean, res_k[K_CHECK]),
            (f"150 curves + {n_copies[-1]} copies of the magnitude outlier", n_copies[-1], 100 * frac_mo, res_base,
             res_mo)):
        checks.append(dict(variant=len(checks), n_copies=nc, copy_share_pct=share, **C.shift_summary(ref, var, den),
                           notes=lab))
    checks = pd.DataFrame(checks)

    print(scan[["copies_per_curve", "copy_share_pct", "gprfda_max", "pointwise_max", "direct_obs_max", "direct_norm_max",
                "pointwise_iterations"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(other[["variant", "gprfda_max", "pointwise_max", "direct_obs_max", "direct_norm_max", "notes"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print(checks[["variant", "gprfda_max", "pointwise_max", "direct_obs_max", "direct_norm_max", "notes"]].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    C.write_tables(__file__, dict(record_classes=classes, aggregate_with_contact_loss_copies=grid,
                                  shift_by_contact_loss_copies=scan, copy_weight_by_contact_loss_copies=weights,
                                  clean_start_by_contact_loss_copies=starts, shift_by_removal_and_outlier_copies=other,
                                  clean_start_after_removal=removal_start, shift_by_contamination_check=checks))
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
