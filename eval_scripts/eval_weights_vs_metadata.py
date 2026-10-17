"""Converged GPR-FDA curve weights at -1.95 V against the record metadata: test chronology and sittings, batch,
record length, period and record-length class, for the 154 pipeline curves and for all 185 records at -1.95 V.

Records
    The 226 .cor files under _input/CCNF_CTAB_PT. File-name metadata: label = batch = 'acidB(<label>)', potential,
    substrate (Ru | Au from '_<Ru|Au>-<run><suffix>_CTAB'), file date (yyyymmdd before '.cor'); folder label = the
    parent folder name after 'acidB-'. Start time = header 'Date: MM-DD-YYYY Time: hh:mm:ss' (instrument clock; only
    intervals are used); record end = largest sample time T of the record (all samples); end time = start + record
    end, truncated to whole seconds. Retained = an Individual_GPR file exists for the record. Records are ordered by
    start time; days since study start = (start - first start) / 86400 s. Sittings: within a label, a new sitting
    after a gap > 6 h between consecutive starts; sittings ordered by their first start; hours since sitting start of
    a record start or end; sitting duration = latest end - first start; gap before a label's second sitting = the
    largest gap between consecutive starts of the label.

Weights
    154 pipeline curves: the pipeline Converged_Weights (order of that file); w_r N = w_r 154; complete = record
    end >= 2990 s; contact loss = pipeline weight < 1e-8; period June = file date >= 20230601. The weight iteration
    below, run on the pipeline posteriors, is checked against the pipeline weights.
    185 curves: the 154 pipeline posteriors followed by the 31 records removed by the 60 s rule, in start order, each
    fitted with the 60 s rule switched off (samples sorted by T, 1e-4 <= T <= 5000 s, x = log10(T + 1e-9), y = I / s
    with s the mean current of the samples int(0.5 n) .. int(0.9 n) - 1, |y| < 0.01 dropped, at most 500 training
    points, middle sample of each uniform bin in x; StandardScaler on x and y, ConstantKernel(1, (1e-2, 1e4)) *
    Matern(1, (1e-2, 1e3), nu = 1.5) + WhiteKernel(1, (1e-12, 10)), alpha 0, normalize_y, 5 restarts,
    random_state 0, posterior sd from predict with return_std, one BLAS thread per fit, fits in --n-jobs joblib
    workers; the results do not depend on it). Curve-wise weights: s2w_r = mean_j S_r(j)^2; w = 1/R; repeat
    ybar = sum_r w_r Y_r, D_r = mean_j (Y_r - ybar)^2, w_new = 1/(D_r + s2w_r + 1e-12) normalized to 1,
    delta = ||w_new - w|| / sqrt(R), w = w_new, until delta < 1e-6; w_r N = w_r 185.

Statistics
    Batches ordered by their first file date. Per batch: n, weight share (sum of w_r), median of w_r N, quartiles of
    w_r N (linear interpolation). Kruskal-Wallis over the 15 batches (154 weights); Spearman of record end against
    log10 w_r (154); medians of w_r N for record end < 2990 s and >= 2990 s. Period x record-length class groups
    (record >= 2990 s, 60-2990 s, < 60 s; Apr-May, June): n, contact-loss count, median and quartiles of w_r N and
    weight share = sum of w_r N / N, for the 154-curve and the 185-curve pools. N_eff = 1 / sum w_r^2.

Self-checks
    226 records, 185 retained, 41 removed, every removed record shorter than 60 s and every retained one at least
    60 s; no overlapping recordings; 17 sittings of 15 labels; weight iteration = Converged_Weights; contact loss =
    the 3 smallest |scale factor|; record classes 105 complete + 46 truncated + 3 contact loss; 185-curve weights
    converged.

Output tables (eval_scripts/output/weights_vs_metadata/)
    records_timeline     record (start order), potential_v, file_date, substrate_au, retained, in_pool_195, complete,
                         contact_loss, june, sitting_row, start_since_study_d, start_since_sitting_h,
                         end_since_sitting_h, record_end_s, weight_n_pipeline (w_r N of the 154 curves);
                         notes = label, folder, file
    sittings             sitting_row, sitting_of_label, n_sittings_of_label, n_records, start_since_study_d,
                         duration_h, june, folder_differs_from_label, gap_before_sitting_h (start of the sitting
                         minus the start of the previous record of the label; empty for the first sitting of a
                         label); notes = label (sitting), file dates, folders, substrates
    weights_by_curve     curve, reincluded, batch_position, file_date, record_end_s, complete, contact_loss,
                         substrate_au, june, weight_pipeline, weight_n_pipeline (154 curves), weight_all,
                         weight_n_all (185 curves);
                         notes = batch, substrate note, file
    batch_summary        batch_position, first_file_date, june, n_curves, weight_share, median_weight_n, q1_weight_n,
                         q3_weight_n, n_complete, n_contact_loss; notes = batch
    period_class_groups  n_pool, record_end_lo_s, record_end_hi_s, june, n_curves, n_contact_loss, median_weight_n,
                         q1_weight_n, q3_weight_n, weight_share; notes = group
    weights_summary      n_records, n_retained, n_removed, n_other_potentials, n_sittings, n_labels, n_curves,
                         n_complete_class, n_truncated_class, n_contact_loss, kruskal_statistic, kruskal_p,
                         spearman_rho, spearman_p, n_short, median_weight_n_short, n_long, median_weight_n_long,
                         n_june, weight_share_june, n_au, n_curves_all, n_reincluded, n_au_all,
                         weight_share_reincluded, median_weight_n_reincluded, n_eff_pipeline, n_eff_all

Run
    python -X utf8 -B eval_scripts/eval_weights_vs_metadata.py [--n-jobs K]
"""
import sys

sys.dont_write_bytecode = True

import argparse
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

import _eval_common as C
import _eval_refit as RF

POT = "-1.95"
X_COL = "T (Seconds)"
SESSION_GAP_H = 6.0
JUNE = 20230601
HDR = re.compile(r"Date:\s*(\d{2})-(\d{2})-(\d{4})\s+Time:\s*(\d{2}):(\d{2}):(\d{2})")
GROUPS = (("complete", "Apr-May"), ("complete", "June"), ("truncated", "Apr-May"), ("truncated", "June"),
          ("reincluded", "Apr-May"), ("reincluded", "June"))
CLASS_SPAN = {"complete": (C.T_COMPLETE, np.inf), "truncated": (C.T_MIN_RECORD, C.T_COMPLETE),
              "reincluded": (0.0, C.T_MIN_RECORD)}
CLASS_LABEL = {"complete": "record >= 2990 s", "truncated": "record 60-2990 s", "reincluded": "record < 60 s"}


def header_start(path):
    """Start date and time in the header of a .cor file."""
    with open(path, "rb") as fh:
        txt = fh.read(6000).decode("latin-1")
    m = HDR.search(txt)
    return datetime(int(m[3]), int(m[1]), int(m[2]), int(m[4]), int(m[5]), int(m[6]))


def parse_name(fn):
    """Label, potential, substrate, substrate note and file date from a .cor file name."""
    lab = re.search(r"acidB\(([^)]*)\)", fn)[1]
    pot = float(re.match(r"(-?\d\.\d+)V", fn)[1])
    m = re.search(r"_(Ru|Au)-(\d+)([^_]*)_CTAB", fn)
    note = re.search(r"\(([^)]*)\)", m[3])
    return dict(label=lab, potential_V=pot, substrate=m[1], substrate_note=(note[1] if note else ""),
                filename_date=int(re.search(r"_(\d{8})\.cor$", fn)[1]))


def group_table(df, wcol, n_total):
    """n, contact-loss count, median and quartiles of w_r N and weight share per (length class, period) group."""
    rows = []
    for cls, per in GROUPS:
        m = (df.length_class == cls) & (df.period == per)
        v = df.loc[m, wcol]
        rows.append(dict(n_pool=n_total, record_end_lo_s=CLASS_SPAN[cls][0], record_end_hi_s=CLASS_SPAN[cls][1],
                         june=int(per == "June"), n_curves=int(m.sum()),
                         n_contact_loss=int(df.loc[m, "contact_loss"].astype(bool).sum()),
                         median_weight_n=v.median() if m.any() else np.nan,
                         q1_weight_n=v.quantile(.25) if m.any() else np.nan,
                         q3_weight_n=v.quantile(.75) if m.any() else np.nan, weight_share=v.sum() / n_total,
                         notes=f"{CLASS_LABEL[cls]}, {per}, {n_total} curves"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-jobs", type=int, default=3, help="joblib workers for the GPR fits")
    a = ap.parse_args()
    t0 = time.time()

    idx = C.raw_index()
    fitted = {re.sub(r"_curve_\d+\.csv$", "", p.name[len("Individual_GPR_"):])
              for p in C.IND.glob("Individual_GPR_*.csv")}
    rows, raw = [], {}
    for name in sorted(idx):
        path = C.find_raw(name)
        df, _ = C.load_record(path)
        r = dict(file=name, folder_label=path.parent.name.split("acidB-")[-1])
        r.update(parse_name(name))
        r.update(start_dt=header_start(path), tmax_s=float(df[X_COL].max()), retained=name in fitted)
        rows.append(r)
        if not r["retained"] and r["potential_V"] == float(POT):
            raw[name] = df
    T = pd.DataFrame(rows)
    T["start_dt"] = pd.to_datetime(T.start_dt)
    T["end_dt"] = T.start_dt + pd.to_timedelta(T.tmax_s, unit="s")
    T = T.sort_values("start_dt").reset_index(drop=True)
    print(f"{len(T)} raw records read in {time.time() - t0:.0f} s", flush=True)
    gap = (T.start_dt - T.end_dt.shift(1)).dt.total_seconds()
    T["days_since_study_start"] = (T.start_dt - T.start_dt.min()).dt.total_seconds() / 86400
    T["sitting"] = T.label + "_s" + T.groupby("label").start_dt.transform(
        lambda s: (s.diff().dt.total_seconds() / 3600 > SESSION_GAP_H).cumsum()).astype(int).astype(str)
    T["complete"] = T.tmax_s >= C.T_COMPLETE
    T["period"] = np.where(T.filename_date >= JUNE, "June", "Apr-May")
    T["end_s"] = T.end_dt.dt.floor("s")
    rm = ~T.retained
    C.check("226 records, 185 retained, 41 removed; removed < 60 s <= retained",
            len(T) == 226 and int(T.retained.sum()) == 185
            and T.tmax_s[rm].max() < C.T_MIN_RECORD <= T.tmax_s[~rm].min(),
            f"{len(T)} / {int(T.retained.sum())} / {int(rm.sum())}; removed max {T.tmax_s[rm].max():.2f} s, retained "
            f"min {T.tmax_s[~rm].min():.2f} s")
    C.check("no overlapping recordings (> 2 s)", int((gap < -2).sum()) == 0, f"min gap {gap.min():.1f} s")

    T["gap_before_h"] = T.groupby("label").start_dt.diff().dt.total_seconds() / 3600
    sit = (T.groupby("sitting").agg(label=("label", "first"), start=("start_dt", "min"), end=("end_s", "max"),
                                    gap_h=("gap_before_h", lambda s: s.iloc[0]),
                                    day0=("days_since_study_start", "min"), n=("file", "size"),
                                    fdate=("filename_date", lambda s: "/".join(sorted({str(v)[4:6] + "-" + str(v)[6:8]
                                                                                         for v in s}))),
                                    folders=("folder_label", lambda s: "/".join(sorted(set(s)))),
                                    substrate=("substrate", lambda s: "/".join(sorted(set(s)))),
                                    period=("period", "first"))
           .sort_values("start"))
    sit["k"] = sit.groupby("label").cumcount() + 1
    sit["n_sit_label"] = sit.groupby("label").label.transform("size")
    sit["row"] = np.arange(len(sit))
    T["sitting_start"] = T.sitting.map(sit.start)
    T["h_sit"] = (T.start_dt - T.sitting_start).dt.total_seconds() / 3600
    T["h_sit_end"] = (T.end_s - T.sitting_start).dt.total_seconds() / 3600
    T["row"] = T.sitting.map(sit.row)
    sit["dur_h"] = T.groupby("sitting").h_sit_end.max().reindex(sit.index)
    C.check("17 sittings of 15 labels, one period per sitting",
            len(sit) == 17 and sit.label.nunique() == 15 and (T.groupby("sitting").period.nunique() == 1).all(),
            f"{len(sit)} sittings, {sit.label.nunique()} labels")

    pool = C.load_pool(POT)
    R = pool.m
    cw = C.load_weights(POT)
    w_it, nit, _, _ = C.curve_weights(pool.Y, pool.S)
    d_w = float(np.abs(w_it - cw.converged_weight.to_numpy()).max())
    C.check("weight iteration on the pipeline posteriors = Converged_Weights", d_w < 1e-10,
            f"max |dw| {d_w:.1e}; {nit} iterations")
    Ti = T.set_index("file")
    files = [C.split_sid(s)[0] for s in cw.sample_id]
    d = pd.DataFrame({"file": files, "batch": Ti.label.reindex(files).to_numpy(),
                      "file_date": Ti.filename_date.reindex(files).to_numpy(),
                      "substrate": Ti.substrate.reindex(files).to_numpy(),
                      "substrate_note": Ti.substrate_note.reindex(files).to_numpy(),
                      "tmax_s": Ti.tmax_s.reindex(files).to_numpy(), "weight": cw.converged_weight.to_numpy()})
    d["w_rN"] = d.weight * R
    d["complete"] = d.tmax_s >= C.T_COMPLETE
    d["contact_loss"] = d.weight < 1e-8
    d["period"] = np.where(d.file_date >= JUNE, "June", "Apr-May")
    d["record_class"] = np.select([d.contact_loss, d.complete], ["contact-loss", "complete"], "truncated")
    cnt = d.record_class.value_counts()
    cl3 = set(np.argsort(np.abs(pool.F))[:3])
    C.check("contact loss: pipeline weight < 1e-8 = 3 smallest |scale factor|; classes 105 + 46 + 3",
            set(np.flatnonzero(d.contact_loss)) == cl3 and (cnt.get("complete"), cnt.get("truncated"),
                                                           cnt.get("contact-loss")) == (105, 46, 3),
            str(cnt.to_dict()))
    T = T.merge(d[["file", "w_rN", "contact_loss"]], on="file", how="left")
    T["contact_loss"] = T.contact_loss.astype("boolean").fillna(False).astype(bool)
    T["in_pool"] = T.retained & np.isclose(T.potential_V, float(POT))

    first = d.groupby("batch").file_date.min()
    order = first.sort_values().index.tolist()
    bs = (d.groupby("batch").agg(n=("weight", "size"), weight_share=("weight", "sum"), median_w_rN=("w_rN", "median"),
                                 n_complete=("complete", "sum"), n_contact_loss=("contact_loss", "sum"))
          .loc[order].reset_index())
    qb = np.array([np.percentile(d.w_rN[d.batch == b].to_numpy(), [25, 75]) for b in order])
    kw = kruskal(*[d.weight[d.batch == b].values for b in order])
    rho, p_rho = spearmanr(d.tmax_s, np.log10(d.weight))
    med_short, med_long = d.w_rN[~d.complete].median(), d.w_rN[d.complete].median()

    removed = [f for f in T.file[rm & np.isclose(T.potential_V, float(POT))]]
    pre = {}
    for f in removed:
        pre[f], why = RF.preprocess_record(raw[f], apply_min_range=False)
        if pre[f] is None:
            C.check(f"preprocessing of {f}", False, why)
    jobs = [dict(x=pre[f]["x_train"], y=pre[f]["y_train"], std_from="std") for f in removed]
    tf = time.time()
    fits = RF.fit_jobs(jobs, n_jobs=a.n_jobs, verbose=0, blas_threads=1)
    print(f"{len(fits)} GPR fits in {time.time() - tf:.0f} s", flush=True)
    Ya = np.vstack([pool.Y, np.array([f.mu for f in fits])])
    Sa = np.vstack([pool.S, np.array([f.sd for f in fits])])
    w_all, nit_all, conv_all, _ = C.curve_weights(Ya, Sa)
    N_ALL = len(w_all)
    C.check("weights of the 185-curve pool converged", N_ALL == 185 and conv_all, f"{nit_all} iterations")
    ac = pd.DataFrame({"file": files + removed, "reincluded": [False] * R + [True] * len(removed),
                       "tmax_s": list(d.tmax_s) + [Ti.tmax_s[f] for f in removed],
                       "contact_loss": list(d.contact_loss) + [False] * len(removed), "w_rN_all": w_all * N_ALL})
    ac["period"] = Ti.period.reindex(ac.file).to_numpy()
    ac["substrate"] = Ti.substrate.reindex(ac.file).to_numpy()
    ac["length_class"] = np.select([ac.tmax_s < C.T_MIN_RECORD, ac.tmax_s >= C.T_COMPLETE],
                                   ["reincluded", "complete"], "truncated")
    C.check("length class < 60 s = the refitted records", ((ac.length_class == "reincluded") == ac.reincluded).all(),
            f"{int(ac.reincluded.sum())} records")
    d154 = d.assign(length_class=np.where(d.complete, "complete", "truncated"))
    gt = pd.concat([group_table(d154, "w_rN", R), group_table(ac, "w_rN_all", N_ALL)], ignore_index=True)
    rei = ac[ac.reincluded]

    bpos = {b: i for i, b in enumerate(order)}
    timeline = pd.DataFrame({
        "record": np.arange(len(T)), "potential_v": T.potential_V, "file_date": T.filename_date,
        "substrate_au": (T.substrate == "Au").astype(int), "retained": T.retained, "in_pool_195": T.in_pool,
        "complete": T.complete, "contact_loss": T.contact_loss, "june": (T.period == "June").astype(int),
        "sitting_row": T.row, "start_since_study_d": T.days_since_study_start, "start_since_sitting_h": T.h_sit,
        "end_since_sitting_h": T.h_sit_end, "record_end_s": T.tmax_s, "weight_n_pipeline": T.w_rN,
        "notes": T.label + "; folder " + T.folder_label + "; " + T.file})
    sittings = pd.DataFrame({
        "sitting_row": sit.row, "sitting_of_label": sit.k, "n_sittings_of_label": sit.n_sit_label,
        "n_records": sit.n, "start_since_study_d": sit.day0, "duration_h": sit.dur_h,
        "june": (sit.period == "June").astype(int), "folder_differs_from_label": (sit.folders != sit.label).astype(int),
        "gap_before_sitting_h": sit.gap_h,
        "notes": [f"{r.label}" + (f" ({r.k})" if r.n_sit_label > 1 else "") + f"; {r.fdate}; folders {r.folders}; "
                  f"{r.substrate}" for r in sit.itertuples()]})
    by_curve = pd.DataFrame({
        "curve": np.arange(N_ALL), "reincluded": ac.reincluded,
        "batch_position": [bpos.get(b, np.nan) for b in Ti.label.reindex(ac.file)],
        "file_date": Ti.filename_date.reindex(ac.file).to_numpy(), "record_end_s": ac.tmax_s,
        "complete": ac.tmax_s >= C.T_COMPLETE, "contact_loss": ac.contact_loss,
        "substrate_au": (ac.substrate == "Au").astype(int), "june": (ac.period == "June").astype(int),
        "weight_pipeline": list(d.weight) + [np.nan] * len(removed),
        "weight_n_pipeline": list(d.w_rN) + [np.nan] * len(removed),
        "weight_all": w_all, "weight_n_all": ac.w_rN_all,
        "notes": ["; ".join(p for p in (Ti.label[f], Ti.substrate_note[f], f) if p) for f in ac.file]})
    batches = pd.DataFrame({
        "batch_position": np.arange(len(order)), "first_file_date": [first[b] for b in order],
        "june": [int(first[b] >= JUNE) for b in order], "n_curves": bs.n, "weight_share": bs.weight_share,
        "median_weight_n": bs.median_w_rN, "q1_weight_n": qb[:, 0], "q3_weight_n": qb[:, 1],
        "n_complete": bs.n_complete.astype(int), "n_contact_loss": bs.n_contact_loss.astype(int), "notes": order})
    summary = pd.DataFrame([dict(
        n_records=len(T), n_retained=int(T.retained.sum()), n_removed=int(rm.sum()),
        n_other_potentials=int((T.retained & ~np.isclose(T.potential_V, float(POT))).sum()), n_sittings=len(sit),
        n_labels=sit.label.nunique(), n_curves=R, n_complete_class=int(cnt["complete"]),
        n_truncated_class=int(cnt["truncated"]), n_contact_loss=int(cnt["contact-loss"]),
        kruskal_statistic=float(kw.statistic), kruskal_p=float(kw.pvalue), spearman_rho=float(rho),
        spearman_p=float(p_rho), n_short=int((~d.complete).sum()),
        median_weight_n_short=float(med_short), n_long=int(d.complete.sum()), median_weight_n_long=float(med_long),
        n_june=int((d.period == "June").sum()), weight_share_june=float(d.weight[d.period == "June"].sum()),
        n_au=int((d.substrate == "Au").sum()), n_curves_all=N_ALL, n_reincluded=len(rei),
        n_au_all=int((ac.substrate == "Au").sum()), weight_share_reincluded=float(rei.w_rN_all.sum() / N_ALL),
        median_weight_n_reincluded=float(rei.w_rN_all.median()),
        n_eff_pipeline=float(1.0 / np.sum((d.w_rN.to_numpy() / R) ** 2)),
        n_eff_all=float(1.0 / np.sum((ac.w_rN_all.to_numpy() / N_ALL) ** 2)))])
    s = summary.iloc[0]
    print(f"Kruskal-Wallis p {s.kruskal_p:.2e}; Spearman rho {s.spearman_rho:.3f} (p {s.spearman_p:.1e}); medians "
          f"{s.median_weight_n_short:.3f} / {s.median_weight_n_long:.3f}; June {s.n_june} curves, "
          f"{100 * s.weight_share_june:.1f} % of the weight; re-included {100 * s.weight_share_reincluded:.2f} %; "
          f"N_eff {s.n_eff_pipeline:.2f} -> {s.n_eff_all:.2f}", flush=True)

    C.write_tables(__file__, {
        "records_timeline": timeline,
        "sittings": sittings,
        "weights_by_curve": by_curve,
        "batch_summary": batches,
        "period_class_groups": gt,
        "weights_summary": summary,
    })
    print(f"{len(C.CHECKS)} checks, {C.checks_failed()} failed; runtime {time.time() - t0:.0f} s", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
