"""Residual diagnostics of the individual GPR fits at -1.95 V by log-time window: coverage of held-out raw points by
the +-2 sigma posterior interval, RMS standardized residual, residual scatter against the posterior sigma, and lag-1
autocorrelation of the standardized residuals per window and over the whole record.

Method
    Curves: the 154 rows of Converged_Weights (pipeline run NS_norm__WM_iterative__WS_curve). For each curve the
    pipeline Diagnostic_Data holds training rows (the GPR at its <= 500 training points) and validation rows (the same
    GPR at every retained raw sample; these rows include the training points). x is read as text and converted with
    pd.to_numeric after stripping '[' and ']'; the rows of each data type are sorted by x (pandas sort_values).
    Held-out raw points = validation rows whose x is not within 5e-8 of a training x (nearest validation row of each
    training x). Windows on t: [0, 1), [1, 10), [10, 100), [100, 1000), [1000, 5000.5) s, assigned on
    x = log10(t + 1e-9) with a 1e-7 tolerance below each inner edge. Per curve, window and point set (held-out raw,
    training), with z = standardized_residual of the pipeline file = (y_observed - y_pred) / (y_std + 1e-10): n_points,
    n_in_2sig (|z| <= 2), frac_in_2sig, rms_stdres = sqrt(mean z^2), lag1_stdres = sum(c[1:] c[:-1]) / sum(c^2),
    c = z - mean(z), in x order (at least 8 points), rms_resid_norm = sqrt(mean (y_observed - y_pred)^2) (normalized
    current). Posterior sigma (y_std_normalized of Individual_GPR on the 500-point grid) per window: median over the
    grid points with x <= log10(t_end + 1e-9) (within the record) and with x > log10(t_end + 1e-9) (beyond the
    record end), t_end = last raw sample time. Raw records (_input/CCNF_CTAB_PT, first match of the file name in the
    sorted list of <folder>/*.cor) are read from the line after 'End Comments' (latin-1) up to the first empty line
    or line containing 'End', sorted by T (stable).
    Window statistics over curves: box statistics of matplotlib.cbook.boxplot_stats(v, whis=(5, 95)) (quartiles from
    np.percentile, whisker ends = most extreme values inside the 5 and 95 % percentiles) of 100 frac_in_2sig,
    rms_stdres and lag1_stdres (finite values; curves with n_points > 0); pooled coverage = 100 (sum n_in_2sig /
    sum n_points); median and quartiles (pandas) of rms_resid_norm and of the median posterior sigma; posterior
    sigma beyond the record end over the records ending before 2990 s.
    Whole record: for each data type (training rows, all validation rows) of the pipeline Diagnostic_Data, rows sorted
    by the x column as read (text; pandas sort_values, string order of the x text), z with non-finite values
    dropped, lag-1 as above when at least 10 values.
    Record classes: contact_loss = pipeline weight < 1e-8 (the 3 smallest |s|); complete = t_end >= 2990 s;
    truncated = the rest.
    Self-checks: pipeline standardized residuals recomputed; every training point found among the validation rows with
    the same prediction; header true_pct_within_2sig reproduced; every validation row matched to a raw sample
    (y_observed s = raw I, log t equal), the raw samples absent from the validation rows have |I/s| < 0.01; s = mean I
    over the samples 50-90 % of the record (t > 0); grid of Individual_GPR = the shared grid; record classes.

Inputs
    _output/CCNF_CTAB_PT_output/individual_GPR (Diagnostic_Data_<sid>.csv, Individual_GPR_<sid>.csv);
    batch_testing/NS_norm__WM_iterative__WS_curve/summary_gpr/iterative (Converged_Weights); _input/CCNF_CTAB_PT.

Output tables (eval_scripts/output/residual_diagnostics/)
    residuals_by_window          one row per curve, window and point set: curve (index in Converged_Weights),
                                 complete, contact_loss, record_end_s (t_end), window_lo_s, window_hi_s, heldout
                                 (1 = held-out raw points, 0 = training points), n_points, n_in_2sig, frac_in_2sig,
                                 rms_stdres, lag1_stdres, rms_resid_norm; notes = sample id
    posterior_sigma_by_window    one row per curve, window and grid part: curve, complete, contact_loss,
                                 record_end_s, window_lo_s, window_hi_s, beyond_record_end (0/1), n_points,
                                 median_sigma_norm; notes = sample id
    residuals_window_summary     one row per window and point set: window_lo_s, window_hi_s, heldout, n_curves,
                                 n_points_total, n_points_per_curve_median, cov2_pooled_pct, box statistics
                                 <q>_n_curves, <q>_whisker_low, <q>_q1, <q>_median, <q>_q3, <q>_whisker_high for
                                 q = cov2_pct, rms_stdres, lag1, and rms_resid_norm_median, rms_resid_norm_q25,
                                 rms_resid_norm_q75; notes = window and point set
    posterior_sigma_window_summary
                                 one row per window and grid part: window_lo_s, window_hi_s, beyond_record_end
                                 (records ending before 2990 s only), n_curves, n_points_total,
                                 median_sigma_norm_median, median_sigma_norm_q25, median_sigma_norm_q75; notes
    whole_record_lag1            one row per curve: curve, complete, contact_loss, training_n, training_lag1,
                                 validation_n, validation_lag1 (all validation rows); notes = sample id
    whole_record_lag1_summary    one row per data type: training (0/1), n_curves, whisker_low, q1, median, q3,
                                 whisker_high (box statistics as above); notes
    records                      one row per curve: curve, complete, contact_loss, record_end_s, n_training,
                                 n_validation, n_heldout, first_heldout_t_s; notes = sample id

Run
    python -X utf8 -B eval_scripts/eval_residual_diagnostics.py [--n-jobs K]
    --n-jobs: worker processes reading the curves (the results do not depend on it).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import glob
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats

import _eval_common as C

POT = "-1.95"
WIN_EDGES = [0.0, 1.0, 10.0, 100.0, 1000.0, 5000.5]
WIN_LABELS = ["0-1 s", "1-10 s", "10-100 s", "100-1000 s", "1000-5000 s"]
LOG_EDGES_INNER = np.log10(np.array(WIN_EDGES[1:-1]))
MIN_N_LAG = 8
MIN_N_LAG_WHOLE = 10
X_MATCH_TOL = 5e-8
CL_WEIGHT = 1e-8


def win_index_x(x):
    """Window index 0..4 for x = log10(t + 1e-9); [lo, hi) with a 1e-7 tolerance below each edge."""
    return np.searchsorted(LOG_EDGES_INNER - 1e-7, np.asarray(x, float), side="right")


def lag1(r):
    r = np.asarray(r, float)
    if len(r) < MIN_N_LAG:
        return np.nan
    rc = r - r.mean()
    den = np.sum(rc ** 2)
    return float(np.sum(rc[1:] * rc[:-1]) / den) if den > 0 else np.nan


def read_cor(path):
    """Raw record: rows after 'End Comments' up to the first empty line or 'End' line -> DataFrame E, I, T by T."""
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    start = None
    for k, line in enumerate(lines):
        if line.strip() == "End Comments":
            start = k + 1
            break
    if start is None:
        raise ValueError(f"no 'End Comments' in {path}")
    rows = []
    for line in lines[start:]:
        s = line.strip()
        if not s or "End" in s:
            break
        v = re.split(r"[\t\s]+", s)
        if len(v) >= 3:
            rows.append((float(v[0]), float(v[1]), float(v[2])))
    return pd.DataFrame(rows, columns=["E", "I", "T"]).sort_values("T", kind="stable").reset_index(drop=True)


def whole_record(d):
    """Lag-1 of z per data type over the whole record, rows sorted by the x text -> {kind_n, kind_lag1}."""
    out = {}
    for kind in d["data_type"].unique():
        s = d[d["data_type"] == kind].sort_values("x")
        r = s["standardized_residual"].to_numpy()
        r = r[np.isfinite(r)]
        if len(r) < MIN_N_LAG_WHOLE:
            continue
        rc = r - r.mean()
        out[f"{kind}_n"] = len(r)
        out[f"{kind}_lag1"] = np.sum(rc[1:] * rc[:-1]) / np.sum(rc ** 2)
    return out


def one_curve(args):
    """Window diagnostics, checks and whole-record lag-1 of one curve."""
    sid, raw_path = args
    dpath = C.diagnostic_path(sid)
    ipath = C.IND / f"Individual_GPR_{sid}.csv"
    meta = C.read_header(dpath)
    d = pd.read_csv(dpath, comment="#")
    whole = whole_record(d)
    d["x"] = pd.to_numeric(d["x"].astype(str).str.strip().str.strip("[]"))
    tr = d[d.data_type == "training"].sort_values("x").reset_index(drop=True)
    va = d[d.data_type == "validation"].sort_values("x").reset_index(drop=True)

    chk = {}
    for name, s in (("tr", tr), ("va", va)):
        z = (s.y_observed - s.y_pred) / (s.y_std + 1e-10)
        chk[f"stdres_{name}"] = float(np.max(np.abs(z - s.standardized_residual)))
    vx = va.x.to_numpy()
    i = np.clip(np.searchsorted(vx, tr.x.to_numpy()), 1, len(vx) - 1)
    dl = np.abs(vx[i - 1] - tr.x.to_numpy())
    dr = np.abs(vx[i] - tr.x.to_numpy())
    j = np.where(dr < dl, i, i - 1)
    matched = np.minimum(dl, dr) < X_MATCH_TOL
    chk["n_training"] = len(tr)
    chk["n_validation"] = len(va)
    chk["n_training_found"] = int(matched.sum())
    chk["train_vs_val_pred"] = float(np.max(np.abs(va.y_pred.to_numpy()[j[matched]] - tr.y_pred.to_numpy()[matched])))
    is_train = np.zeros(len(va), bool)
    is_train[j[matched]] = True
    chk["hdr_pct_2sig"] = float(meta.get("true_pct_within_2sig", "nan"))
    chk["recomp_pct_2sig"] = float(100 * np.mean(np.abs(va.standardized_residual.to_numpy()) <= 2))

    scale = float(C.read_header(ipath)["physical_scale_factor"])
    raw = read_cor(raw_path)
    rp = raw[raw["T"] > 0].reset_index(drop=True)
    lt = np.log10(rp["T"].to_numpy() + 1e-9)
    k = np.clip(np.searchsorted(vx, lt), 1, len(vx) - 1)
    kk = np.where(np.abs(vx[k] - lt) < np.abs(vx[k - 1] - lt), k, k - 1)
    present = np.abs(vx[kk] - lt) < X_MATCH_TOL
    ynorm = rp.I.to_numpy() / scale
    chk["n_absent"] = int((~present).sum())
    chk["absent_max_abs_ynorm"] = float(np.max(np.abs(ynorm[~present]))) if (~present).any() else np.nan
    chk["present_min_abs_ynorm"] = float(np.min(np.abs(ynorm[present])))
    chk["n_validation_matched"] = int(len(np.unique(kk[present])))
    chk["val_vs_raw_I"] = float(np.max(np.abs(va.y_observed.to_numpy()[kk[present]] * scale
                                              - rp.I.to_numpy()[present])) / np.max(np.abs(rp.I)))
    chk["val_vs_raw_logt"] = float(np.max(np.abs(vx[kk[present]] - lt[present])))
    m = len(rp)
    chk["scale_vs_midavg"] = float(abs(rp.I.iloc[int(.5 * m):int(.9 * m)].mean() / scale - 1))
    t_end = float(raw["T"].max())
    g = pd.read_csv(ipath, comment="#")
    chk["grid"] = float(np.max(np.abs(g.iloc[:, 0].to_numpy() - C.GRID)))
    gs = g["y_std_normalized"].to_numpy()
    x_end = np.log10(t_end + 1e-9)

    rows, grid_rows = [], []
    sets = {1: va[~is_train], 0: tr}
    set_widx = {key: win_index_x(v.x.to_numpy()) for key, v in sets.items()}
    gwidx = win_index_x(C.GRID)
    chk["first_heldout_t_s"] = float(10 ** va[~is_train].x.min() - 1e-9) if (~is_train).any() else np.nan
    for wi in range(len(WIN_LABELS)):
        lo, hi = WIN_EDGES[wi], WIN_EDGES[wi + 1]
        for heldout, s in sets.items():
            sel = set_widx[heldout] == wi
            ss = s[sel]
            r = dict(record_end_s=t_end, window_lo_s=lo, window_hi_s=hi, heldout=heldout, n_points=int(sel.sum()),
                     n_in_2sig=0)
            if sel.sum() > 0:
                z = ss.standardized_residual.to_numpy()
                res = (ss.y_observed - ss.y_pred).to_numpy()
                r.update(n_in_2sig=int(np.sum(np.abs(z) <= 2)), frac_in_2sig=float(np.mean(np.abs(z) <= 2)),
                         rms_stdres=float(np.sqrt(np.mean(z ** 2))), lag1_stdres=lag1(z),
                         rms_resid_norm=float(np.sqrt(np.mean(res ** 2))))
            rows.append(r)
        selg = gwidx == wi
        for beyond, sg in ((0, selg & (C.GRID <= x_end)), (1, selg & (C.GRID > x_end))):
            r = dict(record_end_s=t_end, window_lo_s=lo, window_hi_s=hi, beyond_record_end=beyond,
                     n_points=int(sg.sum()))
            if sg.any():
                r.update(median_sigma_norm=float(np.median(gs[sg])))
            grid_rows.append(r)
    chk.update(record_end_s=t_end, n_heldout=int((~is_train).sum()))
    return rows, grid_rows, chk, whole


def box(v):
    """boxplot_stats(v, whis=(5, 95)) of the finite values -> dict(n_curves, whisker_low, q1, median, q3,
    whisker_high)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n_curves=0, whisker_low=np.nan, q1=np.nan, median=np.nan, q3=np.nan, whisker_high=np.nan)
    s = boxplot_stats(v, whis=(5, 95))[0]
    return dict(n_curves=len(v), whisker_low=float(s["whislo"]), q1=float(s["q1"]), median=float(s["med"]),
                q3=float(s["q3"]), whisker_high=float(s["whishi"]))


def quartiles(v):
    v = v.dropna()
    if not len(v):
        return np.nan, np.nan, np.nan
    return float(v.median()), float(v.quantile(.25)), float(v.quantile(.75))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="worker processes reading the curves")
    a = ap.parse_args()
    t0 = time.time()
    wdf = C.load_weights(POT)
    sids = wdf.sample_id.tolist()
    w_pipeline = wdf.converged_weight.to_numpy(float)
    R = len(sids)
    diag = sorted(glob.glob(str(C.IND / f"Diagnostic_Data_*{POT}V*.csv")))
    names = {Path(f).name[len("Diagnostic_Data_"):-4] for f in diag}
    C.check("Diagnostic_Data files at -1.95 V = the Converged_Weights curves", names == set(sids),
            f"{len(diag)} files, {R} curves")
    by_name = {}
    for p in sorted(glob.glob(str(C.RAW / "*" / "*.cor"))):
        by_name.setdefault(Path(p).name, []).append(p)
    args = [(sid, by_name[re.sub(r"_curve_\d+$", "", sid)][0]) for sid in sids]
    if a.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
            res = list(ex.map(one_curve, args, chunksize=4))
    else:
        res = [one_curve(x) for x in args]
    print(f"read {R} curves in {time.time() - t0:.0f} s", flush=True)
    chk = pd.DataFrame([c for _, _, c, _ in res])
    tend = chk.record_end_s.to_numpy()
    cl = w_pipeline < CL_WEIGHT
    complete = tend >= C.T_COMPLETE
    sc = np.array([float(C.read_header(C.IND / f"Individual_GPR_{sid}.csv")["physical_scale_factor"]) for sid in sids])
    n_cls = (int((complete & ~cl).sum()), int((~complete & ~cl).sum()), int(cl.sum()))
    C.check("record classes complete / truncated / contact loss = 105 / 46 / 3; contact loss = 3 smallest |s|",
            n_cls == (105, 46, 3) and np.array_equal(np.sort(np.argsort(np.abs(sc))[:3]), np.flatnonzero(cl)),
            f"{n_cls[0]} / {n_cls[1]} / {n_cls[2]}; records ending before 2990 s: {int((~complete).sum())}")
    C.check("pipeline standardized residuals recomputed (training / validation)",
            chk.stdres_tr.max() < 1e-12 and chk.stdres_va.max() < 1e-12,
            f"max |diff| {chk.stdres_tr.max():.1e} / {chk.stdres_va.max():.1e}")
    C.check("every training point found among the validation rows with the same prediction",
            (chk.n_training_found == chk.n_training).all() and chk.train_vs_val_pred.max() < 1e-6,
            f"{int(chk.n_training_found.sum())} of {int(chk.n_training.sum())}; max |d pred| "
            f"{chk.train_vs_val_pred.max():.1e}")
    dh = float(np.nanmax(np.abs(chk.hdr_pct_2sig - chk.recomp_pct_2sig)))
    C.check("header true_pct_within_2sig reproduced from all validation rows", dh < 1e-9, f"max |diff| {dh:.1e}")
    ab = chk[chk.n_absent > 0]
    C.check("validation rows = raw samples (t > 0) with |I/s| >= 0.01",
            (chk.n_validation_matched == chk.n_validation).all()
            and (not len(ab) or ab.absent_max_abs_ynorm.max() < 0.01) and chk.present_min_abs_ynorm.min() >= 0.01
            and chk.val_vs_raw_I.max() < 1e-9 and chk.val_vs_raw_logt.max() < X_MATCH_TOL,
            f"{int(ab.n_absent.sum())} absent samples in {len(ab)} curves; y_observed s vs I "
            f"{chk.val_vs_raw_I.max():.1e}; log t {chk.val_vs_raw_logt.max():.1e}")
    C.check("s = mean I over the samples 50-90 % of the record (t > 0)", chk.scale_vs_midavg.max() < 1e-12,
            f"max relative difference {chk.scale_vs_midavg.max():.1e}")
    C.check("grid of Individual_GPR = shared grid", chk.grid.max() < 1e-12, f"max |dx| {chk.grid.max():.1e}")

    ident = pd.DataFrame({"curve": np.arange(R), "complete": complete, "contact_loss": cl})
    cols = ["record_end_s", "window_lo_s", "window_hi_s", "heldout", "n_points", "n_in_2sig", "frac_in_2sig",
            "rms_stdres", "lag1_stdres", "rms_resid_norm"]
    pc = pd.concat([pd.DataFrame(rows).reindex(columns=cols).assign(_r=r) for r, (rows, _, _, _) in enumerate(res)],
                   ignore_index=True)
    pc = ident.iloc[pc._r].reset_index(drop=True).join(pc.drop(columns="_r"))
    pc["notes"] = [sids[r] for r in pc.curve]
    gcols = ["record_end_s", "window_lo_s", "window_hi_s", "beyond_record_end", "n_points", "median_sigma_norm"]
    pg = pd.concat([pd.DataFrame(g).reindex(columns=gcols).assign(_r=r) for r, (_, g, _, _) in enumerate(res)],
                   ignore_index=True)
    pg = ident.iloc[pg._r].reset_index(drop=True).join(pg.drop(columns="_r"))
    pg["notes"] = [sids[r] for r in pg.curve]

    summ = []
    for wi, lab in enumerate(WIN_LABELS):
        lo, hi = WIN_EDGES[wi], WIN_EDGES[wi + 1]
        for heldout in (1, 0):
            g = pc[(pc.heldout == heldout) & (pc.window_lo_s == lo) & (pc.n_points > 0)]
            row = dict(window_lo_s=lo, window_hi_s=hi, heldout=heldout, n_curves=len(g),
                       n_points_total=int(g.n_points.sum()),
                       n_points_per_curve_median=float(g.n_points.median()) if len(g) else np.nan,
                       cov2_pooled_pct=100 * float(g.n_in_2sig.sum() / g.n_points.sum()) if len(g) else np.nan)
            for q, v in (("cov2_pct", 100 * g.frac_in_2sig), ("rms_stdres", g.rms_stdres),
                         ("lag1", g.lag1_stdres[g.lag1_stdres.notna()])):
                row.update({f"{q}_{k}": x for k, x in box(v).items()})
            md, q1, q3 = quartiles(g.rms_resid_norm)
            row.update(rms_resid_norm_median=md, rms_resid_norm_q25=q1, rms_resid_norm_q75=q3,
                       notes=f"{lab}, {'held-out raw' if heldout else 'training'} points"
                       + ("" if len(g) else ", no points"))
            summ.append(row)
    gsum = []
    for wi, lab in enumerate(WIN_LABELS):
        lo, hi = WIN_EDGES[wi], WIN_EDGES[wi + 1]
        for beyond in (0, 1):
            g = pg[(pg.beyond_record_end == beyond) & (pg.window_lo_s == lo) & (pg.n_points > 0)]
            if beyond:
                g = g[g.record_end_s < C.T_COMPLETE]
            md, q1, q3 = quartiles(g.median_sigma_norm)
            gsum.append(dict(window_lo_s=lo, window_hi_s=hi, beyond_record_end=beyond, n_curves=len(g),
                             n_points_total=int(g.n_points.sum()), median_sigma_norm_median=md,
                             median_sigma_norm_q25=q1, median_sigma_norm_q75=q3,
                             notes=f"{lab}, " + ("beyond the record end, records ending before 2990 s" if beyond
                                                 else "within the record, all curves")))
    wh = pd.DataFrame([x for _, _, _, x in res])
    whole = ident.assign(training_n=wh.training_n, training_lag1=wh.training_lag1.astype(float),
                         validation_n=wh.validation_n, validation_lag1=wh.validation_lag1.astype(float), notes=sids)
    wsum = [dict(training=1, **box(whole.training_lag1), notes="training rows"),
            dict(training=0, **box(whole.validation_lag1), notes="all validation rows")]
    recs = ident.assign(record_end_s=tend, n_training=chk.n_training, n_validation=chk.n_validation,
                        n_heldout=chk.n_heldout, first_heldout_t_s=chk.first_heldout_t_s, notes=sids)

    S = pd.DataFrame(summ)
    ho = S[(S.heldout == 1) & (S.n_curves > 0)]
    print("pooled held-out coverage (%): " + " / ".join(f"{v:.1f}" for v in ho.cov2_pooled_pct)
          + "; held-out median RMS standardized residual: " + " / ".join(f"{v:.2f}" for v in ho.rms_stdres_median)
          + f"; whole-record lag-1 medians: all validation rows {wsum[1]['median']:.4f}, training rows "
          f"{wsum[0]['median']:.4f}", flush=True)
    C.write_tables(__file__, {
        "residuals_by_window": pc,
        "posterior_sigma_by_window": pg,
        "residuals_window_summary": S,
        "posterior_sigma_window_summary": pd.DataFrame(gsum),
        "whole_record_lag1": whole,
        "whole_record_lag1_summary": pd.DataFrame(wsum),
        "records": recs,
    })
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
