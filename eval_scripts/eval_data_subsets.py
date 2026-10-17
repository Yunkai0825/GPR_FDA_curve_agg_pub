"""Data subsets left out of the aggregate at -1.95 V: each subset is removed before initialization and the curve
weights are iterated again on the remaining curves.

Pool
    The 154 pipeline posteriors at -1.95 V (_output/CCNF_CTAB_PT_output/individual_GPR: normalized posterior mean and
    sd on the grid log10 t = linspace(-1, 3.69897, 500), and the scale factor s_r of each curve). Nothing is refitted.

Subsets
    batch            each of the 15 batches in turn ('acidB(<batch>)' in the file name; a batch is the set of tests run in
                     the same catholyte), batches in sorted order
    Au               the 6 Au records ('_Au-<n>_' in the file name); 148 records are kept
    best5/10/30      the 5, 10 and 30 records with the largest pipeline GPR-FDA weight (Converged_Weights)
    initial5/10/30   the 5, 10 and 30 records with the smallest D_init = ((Y_r - ybar)^2 at t = 0.1 s + (Y_r - ybar)^2
                     at t = 1 s) / 2 in normalized current, ybar = Summary_GPR y_normalised; the 1 s values are
                     interpolated linearly in log10 t between the grid points 0.9958 s and 1.0176 s

GPR-FDA aggregate
    Curve-wise iterative weights (tolerance 1e-6, eps 1e-12) from equal weights; mean_norm = sum_r w_r Y_r,
    mean = mean_norm sum_r w_r s_r, band_norm = sqrt(sum_r w_r (Y_r - mean_norm)^2 + sum_r w_r S_r^2), band in current
    = sqrt(sum_r w_r (Y_r s_r - mean)^2 + sum_r w_r (S_r s_r)^2). The reference is the aggregate of the 154 curves. Shifts of a subset are
    (subset mean - reference mean) over the reference band of the same scale, on t = 10^(log10 t) s.

Four-method shifts
    The same subsets aggregated by GPR-FDA (as above), point-wise iterative weights (run to convergence, at most 10000
    iterations), the direct average of the currents and the direct average of the normalized currents, each compared in
    current with the same method on the 154 curves and divided by the GPR-FDA band in current of the 154 curves.

Output tables (eval_scripts/output/data_subsets/; currents in mA = A cm^-2 x 1e3 on the nominal 1 cm2 area)
    subset_selection        one row per curve: curve (position in Converged_Weights), batch (index of the batch in
                            sorted order), au (0/1), weight (Converged_Weights), weight_rank (1 = largest), d_init
                            (normalized current squared), initial_rank (1 = smallest d_init), best5, best10, best30,
                            initial5, initial10, initial30 (0/1, 1 = removed in that subset), notes = sample id
    reference_aggregate     t_s, log10_t, mean_norm, band_norm, mean_mA, band_mA (GPR-FDA aggregate of the 154 curves)
    aggregate_by_subset     variant, t_s, log10_t, mean_norm, band_norm, shift_norm (mean_norm - reference mean_norm),
                            mean_mA, band_mA, shift_mA (mean_mA - reference mean_mA), notes = subset
    gprfda_shift_by_subset  variant, n_removed, n_kept, weight_removed (sum of the reference weights of the removed
                            curves), norm_max_over_band, norm_t_max_s, current_max_over_band, current_t_max_s (max over
                            the grid of |shift| / reference band in normalized current and in current, and its time),
                            notes = subset
    shift_by_subset         variant, n_removed, gprfda_max, gprfda_rms, pointwise_max, pointwise_rms, direct_obs_max,
                            direct_obs_rms, direct_norm_max, direct_norm_rms, pointwise_iterations,
                            pointwise_converged, n_reference, n_variant, notes = subset
    variant: 0-14 = batches in sorted order, 15 = Au, 16-21 = best5, best10, best30, initial5, initial10, initial30

Run
    python -X utf8 -B eval_scripts/eval_data_subsets.py
"""
import sys

sys.dont_write_bytecode = True

import argparse
import time

import numpy as np
import pandas as pd

import _eval_common as C

POT = "-1.95"
SCORE = ("best5", "best10", "best30", "initial5", "initial10", "initial30")
LABEL = {**{f"best{n}": f"without the {n} highest-weight curves" for n in (5, 10, 30)},
         **{f"initial{n}": f"without the {n} best initial-stage curves" for n in (5, 10, 30)}}


def max_rel(a, b):
    return float(np.max(np.abs(np.asarray(a) / np.asarray(b) - 1)))


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    t0 = time.time()
    pool = C.load_pool(POT)
    R = pool.m
    ALL = np.arange(R)
    st = C.load_summary(POT)
    w_pipeline = C.load_weights(POT).converged_weight.to_numpy(float)
    T = C.T10_GRID

    ref = C.aggregate(pool, "gprfda", ALL)
    dw = float(np.abs(ref.w - w_pipeline).max())
    d = max(max_rel(ref.mean_real, st.y_real), max_rel(ref.band_real, st.Std_real),
            max_rel(ref.band_norm, st.Std_normalised))
    dn = float(np.abs(ref.mean_norm - st.y_normalised.to_numpy(float)).max())
    C.check("reference aggregate reproduces Converged_Weights and Summary_GPR", dw <= 1e-12 and d <= 1e-10 and dn <= 1e-10,
            f"{ref.iterations} iterations; max abs weight deviation {dw:.2g}; max relative {d:.2g}; normalized mean "
            f"max abs {dn:.2g}")
    dt = max_rel(pool.t, T)
    C.check("grid times", dt <= 1e-6, f"max relative deviation from 10^(log10 t) {dt:.2g}")

    meta = C.pool_meta(pool.sids)
    batch = meta.batch.to_numpy()
    batches = sorted(np.unique(batch))
    au = meta.substrate.to_numpy() == "Au"
    C.check("batches and substrates", len(batches) == 15 and "?" not in batches and int(au.sum()) == 6,
            f"{len(batches)} batches; {int(au.sum())} Au, {int((~au).sum())} other records")
    masks, sel = C.select_subsets(pool, w_pipeline, st.y_normalised.to_numpy(float))
    C.check("subset sizes", all(int(masks[k].sum()) == C.SUBSET_SIZES[k] for k in SCORE),
            ", ".join(f"{k} {int(masks[k].sum())}" for k in SCORE))

    subsets = [(f"without batch {b} ({int((batch == b).sum())} curves)", batch == b) for b in batches]
    subsets += [(f"without Au ({int(au.sum())} curves)", au)]
    subsets += [(LABEL[k], masks[k]) for k in SCORE]

    selection = pd.DataFrame(dict(curve=ALL, batch=np.searchsorted(batches, batch), au=au, weight=w_pipeline,
                                  weight_rank=sel.weight_rank.to_numpy(),
                                  d_init=sel.d_init.to_numpy(float),
                                  initial_rank=sel.initial_rank.to_numpy(),
                                  **{k: masks[k] for k in SCORE}, notes=pool.sids))
    reference = pd.DataFrame(dict(t_s=T, log10_t=C.GRID, mean_norm=ref.mean_norm, band_norm=ref.band_norm,
                                  mean_mA=1e3 * ref.mean_real, band_mA=1e3 * ref.band_real))

    curves, summary = [], []
    for v, (lab, drop) in enumerate(subsets):
        res = C.aggregate(pool, "gprfda", np.flatnonzero(~drop))
        dn_ = res.mean_norm - ref.mean_norm
        dr_ = res.mean_real - ref.mean_real
        zn, zr = dn_ / ref.band_norm, dr_ / ref.band_real
        curves.append(pd.DataFrame(dict(variant=v, t_s=T, log10_t=C.GRID, mean_norm=res.mean_norm, band_norm=res.band_norm,
                                        shift_norm=dn_, mean_mA=1e3 * res.mean_real, band_mA=1e3 * res.band_real,
                                        shift_mA=1e3 * dr_, notes=lab)))
        summary.append(dict(variant=v, n_removed=int(drop.sum()), n_kept=int((~drop).sum()),
                            weight_removed=float(ref.w[drop].sum()),
                            norm_max_over_band=float(np.abs(zn).max()), norm_t_max_s=float(T[np.argmax(np.abs(zn))]),
                            current_max_over_band=float(np.abs(zr).max()),
                            current_t_max_s=float(T[np.argmax(np.abs(zr))]), notes=lab))
    curves = pd.concat(curves, ignore_index=True)
    summary = pd.DataFrame(summary)
    print(summary.drop(columns="notes").assign(subset=summary.notes).to_string(
        index=False, float_format=lambda x: f"{x:.4g}"), flush=True)

    rows = []
    full = C.aggregate_all(pool, ALL)
    den = full["gprfda"].band_real
    for v, (lab, drop) in enumerate(subsets):
        var = C.aggregate_all(pool, np.flatnonzero(~drop))
        rows.append(dict(variant=v, n_removed=int(drop.sum()), **C.shift_summary(full, var, den), notes=lab))
        print(f"  {lab}: point-wise {var['pointwise'].iterations} iterations ({time.time() - t0:.0f} s)", flush=True)
    shift = pd.DataFrame(rows)
    C.check("every point-wise run converged", bool(full["pointwise"].converged) and bool(shift.pointwise_converged.all()),
            f"iterations {full['pointwise'].iterations} (154 curves), {shift.pointwise_iterations.min()}-"
            f"{shift.pointwise_iterations.max()} (subsets)")
    print(shift[["variant", "gprfda_max", "pointwise_max", "direct_obs_max", "direct_norm_max", "notes"]].to_string(
        index=False, float_format=lambda x: f"{x:.4g}"), flush=True)

    C.write_tables(__file__, dict(subset_selection=selection, reference_aggregate=reference,
                                  aggregate_by_subset=curves, gprfda_shift_by_subset=summary, shift_by_subset=shift))
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
