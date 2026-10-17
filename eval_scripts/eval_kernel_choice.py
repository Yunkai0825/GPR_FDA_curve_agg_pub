"""Kernel family of the individual GPR fits at -1.95 V.

Every record is refitted on its training rows of individual_GPR/Diagnostic_Data_<sample>.csv with
ConstantKernel(1.0, (1e-2, 1e4)) * k(length_scale=1.0, (1e-2, 1e3)) + WhiteKernel(1.0, (1e-12, 10)),
k = Matern nu 0.5, 1.5, 2.5 or RBF, StandardScaler on x = log10(t + 1e-9) and on y, L-BFGS-B with 5 restarts and
random_state 0.

Likelihood comparison: the 151 records with GPR-FDA weight > 1e-6, in the order numpy.random.default_rng(0).choice
over the sorted file names; alpha 1e-7, normalize_y False, one fit after another with 6 BLAS threads. Per record and
kernel: log marginal likelihood of the standardized training data, and the wall time of the fit plus the prediction
(return_std) at the validation rows of the record.

Aggregate comparison: all 154 records; alpha 0, normalize_y True, prediction (return_std) on the grid
log10 t = linspace(-1, 3.69897, 500); two worker processes with 3 BLAS threads each. GPR-FDA, point-wise weighting
(to convergence, at most 10000 iterations), the direct average of the currents and the direct average of the normalized
currents are recomputed on the refitted posteriors of each kernel with the pipeline scale factors. Shifts are taken
against the same method on the Matern 1.5 refit and divided by the band of the Matern 1.5 GPR-FDA aggregate.

Representative record: among the records with weight > 1e-6 that end at or after 2990 s, sorted by w N
(N = number of records), the one at position len // 2.

Output tables in output/kernel_choice/ (<k> = matern05, matern15, matern25, rbf; currents in mA on the nominal
1 cm2 electrode area):
  lml_per_curve             curve (position in the pool), n_train, n_validation, lml_<k>, dlml_<k> = lml_<k> -
                            lml_matern15, best_nu (inf = RBF), fit_predict_s_<k> (wall time of the fit plus the
                            prediction at the validation rows), notes = sample id
  lml_summary               one row per kernel: nu, n_curves, n_highest_lml, dlml_q10, dlml_q25, dlml_q50, dlml_q75,
                            dlml_q90, dlml_mean, fit_predict_s_median, fit_predict_s_min, fit_predict_s_max,
                            notes = kernel
  aggregate_by_kernel       t_s, log10_t, gprfda_mean_mA_<k>, gprfda_band_mA_<k>; shift_mA_<a>, shift_over_band_<a>
                            for a = matern05, matern25, rbf (difference from Matern 1.5)
  shift_by_kernel           one row per kernel a: nu, gprfda_max, gprfda_rms, pointwise_max, pointwise_rms, direct_obs_max, direct_obs_rms,
                            direct_norm_max, direct_norm_rms, pointwise_iterations, pointwise_converged, n_reference,
                            n_variant, notes = kernel
  representative_record     curve, weight_n, record_end_s, scale_factor_mA, n_train, n_validation, notes = sample id
  representative_posterior  t_s, log10_t, mean_norm_<k>, sd_norm_<k>, mean_mA_<k>, sd_mA_<k>; diff_norm_<a>,
                            diff_mA_<a> (difference from Matern 1.5)
  representative_points     t_s, log10_t, current_norm, current_mA, resid_norm, resid_mA (residual from the Matern 1.5
                            posterior mean, interpolated in log10 t)

Run
    python -X utf8 -B eval_scripts/eval_kernel_choice.py
"""
import sys

sys.dont_write_bytecode = True

import time

import numpy as np
import pandas as pd

import _eval_common as C
import _eval_refit as RF

POT = "-1.95"
KERNELS = ("M0.5", "M1.5", "M2.5", "RBF")
REF = "M1.5"
ALT = ("M0.5", "M2.5", "RBF")
COL = {"M0.5": "matern05", "M1.5": "matern15", "M2.5": "matern25", "RBF": "rbf"}
NAME = {"M0.5": "Matern 0.5", "M1.5": "Matern 1.5", "M2.5": "Matern 2.5", "RBF": "RBF"}
NU = {"M0.5": 0.5, "M1.5": 1.5, "M2.5": 2.5, "RBF": np.inf}
AGG_ORDER = ("M1.5", "M0.5", "M2.5", "RBF")
LML_THREADS = 6
AGG_WORKERS = 2
AGG_THREADS = 3
WEIGHT_MIN = 1e-6
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def diagnostic_rows(sid):
    """Training and validation rows (x = log10(t + 1e-9), y = normalized current) of a Diagnostic_Data file."""
    d = pd.read_csv(C.diagnostic_path(sid), comment="#")
    for c in ("x", "y_observed"):
        d[c] = pd.to_numeric(d[c].astype(str).str.strip().str.strip("[]"), errors="coerce")
    d = d.dropna(subset=["x", "y_observed"])
    kind = d.data_type.astype(str).str.lower()
    tr, va = d[kind.str.startswith("train")], d[kind.str.startswith("valid")]
    return dict(x_train=tr.x.to_numpy(), y_train=tr.y_observed.to_numpy(), x_val=va.x.to_numpy(),
                y_val=va.y_observed.to_numpy())


def likelihood_fit(item):
    """One fit of the likelihood comparison: log marginal likelihood and wall time of fit plus validation prediction."""
    rows, kernel = item
    f = RF.fit_curve(rows["x_train"], rows["y_train"], kernel, x_pred=rows["x_val"], std_from="std", alpha=1e-7,
                     normalize_y=False)
    return dict(lml=f.lml_fit, fit_predict_s=f.fit_s + f.predict_s, n_train=f.n_train)


def grid_fits(item):
    """Grid posteriors (normalized mean and sd) of one record for every kernel."""
    x, y = item
    out = {}
    for k in AGG_ORDER:
        f = RF.fit_curve(x, y, k, x_pred=C.GRID, std_from="std")
        out[k] = (f.mu, f.sd)
    return out


def representative_record(w, tend):
    """Position of the record at len // 2 among the records with weight > 1e-6 ending at or after 2990 s, sorted by
    w N."""
    wn = w * len(w)
    ok = (w > WEIGHT_MIN) & (tend >= C.T_COMPLETE)
    cand = pd.DataFrame(dict(pos=np.flatnonzero(ok), w_n=wn[ok])).sort_values("w_n")
    return int(cand.iloc[len(cand) // 2].pos), len(cand)


def main():
    t_start = time.time()
    pool = C.load_pool(POT)
    w = C.load_weights(POT).converged_weight.to_numpy(float)
    R = pool.m
    clean = w > WEIGHT_MIN
    C.check("records with weight <= 1e-6 are the three smallest scale factors",
            set(np.flatnonzero(~clean).tolist()) == set(C.contact_loss_indices(pool).tolist()),
            f"{int((~clean).sum())} records")

    print(f"reading {R} Diagnostic_Data files ...", flush=True)
    rows = {sid: diagnostic_rows(sid) for sid in pool.sids}

    names = sorted(p.name for p in C.IND.glob(f"Diagnostic_Data_*{POT}V*.csv"))
    good = {pool.sids[i] for i in np.flatnonzero(clean)}
    lml_sids = [n[len("Diagnostic_Data_"):-len(".csv")] for n in names]
    lml_sids = [s for s in lml_sids if s in good]
    pick = np.random.default_rng(0).choice(len(lml_sids), size=len(lml_sids), replace=False)
    order = [lml_sids[i] for i in pick]

    print(f"likelihood comparison: {len(order)} records x {len(KERNELS)} kernels, serial, {LML_THREADS} BLAS threads",
          flush=True)
    lik = {}
    for j, sid in enumerate(order):
        for k in KERNELS:
            lik[(sid, k)] = RF.pmap(likelihood_fit, [(rows[sid], k)], n_jobs=1, blas_threads=LML_THREADS)[0]
        if (j + 1) % 10 == 0 or j + 1 == len(order):
            print(f"  {j + 1}/{len(order)} records, {time.time() - t_start:.0f} s", flush=True)

    print(f"aggregate comparison: {R} records x {len(AGG_ORDER)} kernels, {AGG_WORKERS} workers x {AGG_THREADS} BLAS "
          "threads", flush=True)
    fits = RF.pmap(grid_fits, [(rows[s]["x_train"], rows[s]["y_train"]) for s in pool.sids], n_jobs=AGG_WORKERS,
                   blas_threads=AGG_THREADS)
    Y = {k: np.array([f[k][0] for f in fits]) for k in KERNELS}
    S = {k: np.array([f[k][1] for f in fits]) for k in KERNELS}

    agg = {k: C.aggregate_all(C.Pool(POT, pool.sids, pool.x, pool.t, Y[k], S[k], pool.F)) for k in KERNELS}
    agg_pipe = C.aggregate_all(pool)
    summ = C.load_summary(POT)
    C.check("GPR-FDA aggregate of the pipeline posteriors reproduces Summary_GPR",
            np.abs(agg_pipe["gprfda"].mean_real / summ.y_real.to_numpy(float) - 1).max() < 1e-10)
    ref = agg[REF]
    band = ref["gprfda"].band_real
    d_ref = np.abs(ref["gprfda"].mean_real - agg_pipe["gprfda"].mean_real) / band
    C.check("Matern 1.5 refit reproduces the pipeline GPR-FDA aggregate", d_ref.max() < 1e-5, f"max {d_ref.max():.2g} band")

    lml = pd.DataFrame([{k: lik[(s, k)]["lml"] for k in KERNELS} for s in order], index=order)
    tfit = pd.DataFrame([{k: lik[(s, k)]["fit_predict_s"] for k in KERNELS} for s in order], index=order)
    pos = [pool.index(s) for s in pool.sids if s in good]
    sids_c = [pool.sids[i] for i in pos]
    lml, tfit = lml.loc[sids_c], tfit.loc[sids_c]
    dlml = lml.sub(lml[REF], axis=0)
    best = lml[list(KERNELS)].idxmax(axis=1)
    per = pd.DataFrame(dict(curve=pos, n_train=[lik[(s, REF)]["n_train"] for s in sids_c],
                            n_validation=[len(rows[s]["x_val"]) for s in sids_c]))
    for k in KERNELS:
        per[f"lml_{COL[k]}"] = lml[k].to_numpy()
    for k in KERNELS:
        per[f"dlml_{COL[k]}"] = dlml[k].to_numpy()
    per["best_nu"] = best.map(NU).to_numpy(float)
    for k in KERNELS:
        per[f"fit_predict_s_{COL[k]}"] = tfit[k].to_numpy()
    per["notes"] = sids_c

    summary = []
    for k in KERNELS:
        v = dlml[k]
        summary.append(dict(nu=NU[k], n_curves=len(v), n_highest_lml=int((best == k).sum()),
                            **{f"dlml_q{int(round(q * 100))}": float(v.quantile(q)) for q in QUANTILES},
                            dlml_mean=float(v.mean()), fit_predict_s_median=float(tfit[k].median()),
                            fit_predict_s_min=float(tfit[k].min()), fit_predict_s_max=float(tfit[k].max()),
                            notes=NAME[k]))
    summary = pd.DataFrame(summary)
    C.check("highest-likelihood counts cover every record", int(summary.n_highest_lml.sum()) == len(sids_c))

    grid = pd.DataFrame(dict(t_s=C.T_GRID, log10_t=C.GRID))
    for k in KERNELS:
        grid[f"gprfda_mean_mA_{COL[k]}"] = agg[k]["gprfda"].mean_real * 1e3
        grid[f"gprfda_band_mA_{COL[k]}"] = agg[k]["gprfda"].band_real * 1e3
    for k in ALT:
        dz = agg[k]["gprfda"].mean_real - ref["gprfda"].mean_real
        grid[f"shift_mA_{COL[k]}"] = dz * 1e3
        grid[f"shift_over_band_{COL[k]}"] = dz / band

    shift = pd.DataFrame([dict(nu=NU[k], **C.shift_summary(ref, agg[k], band), notes=NAME[k]) for k in ALT])

    print("reading record ends ...", flush=True)
    tend = C.record_ends(pool.sids)
    ex, n_cand = representative_record(w, tend)
    sid = pool.sids[ex]
    F = float(pool.F[ex])
    print(f"representative record: {sid} ({n_cand} candidates)", flush=True)
    rec = pd.DataFrame([dict(curve=ex, weight_n=float(w[ex] * R), record_end_s=float(tend[ex]), scale_factor_mA=F * 1e3,
                             n_train=len(rows[sid]["x_train"]), n_validation=len(rows[sid]["x_val"]), notes=sid)])
    post = pd.DataFrame(dict(t_s=C.T_GRID, log10_t=C.GRID))
    for k in KERNELS:
        post[f"mean_norm_{COL[k]}"] = Y[k][ex]
        post[f"sd_norm_{COL[k]}"] = S[k][ex]
    for k in KERNELS:
        post[f"mean_mA_{COL[k]}"] = Y[k][ex] * F * 1e3
        post[f"sd_mA_{COL[k]}"] = np.abs(S[k][ex] * F) * 1e3
    for k in ALT:
        post[f"diff_norm_{COL[k]}"] = Y[k][ex] - Y[REF][ex]
        post[f"diff_mA_{COL[k]}"] = (Y[k][ex] - Y[REF][ex]) * F * 1e3
    xv, yv = rows[sid]["x_val"], rows[sid]["y_val"]
    res = yv - np.interp(xv, C.GRID, Y[REF][ex])
    pts = pd.DataFrame(dict(t_s=10 ** xv - C.X_SHIFT, log10_t=xv, current_norm=yv, current_mA=yv * F * 1e3,
                            resid_norm=res, resid_mA=res * F * 1e3))

    C.write_tables(__file__, dict(lml_per_curve=per, lml_summary=summary, aggregate_by_kernel=grid,
                                  shift_by_kernel=shift, representative_record=rec, representative_posterior=post,
                                  representative_points=pts))
    print(f"done in {(time.time() - t_start) / 60:.1f} min; {C.checks_failed()} failed checks", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
