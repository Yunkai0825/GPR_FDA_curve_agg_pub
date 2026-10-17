"""Shared helpers of the evaluation scripts (numpy and pandas only).

Contents
  * paths of the released inputs (_input/CCNF_CTAB_PT) and pipeline outputs (_output/CCNF_CTAB_PT_output)
  * loaders of the pipeline outputs: Individual_GPR posteriors, Converged_Weights, Summary_GPR, Weight_History,
    Validation_Results and the Diagnostic_Data training rows
  * the .cor parser of the pipeline, record ends and file-name metadata (batch, substrate, run, date)
  * one aggregator per method: GPR-FDA (curve-wise iterative weights), point-wise iterative weights with an explicit
    iteration cap, direct average of the currents and direct average of the normalized currents; each reproduces the
    corresponding pipeline Summary_GPR, Converged_Weights and Weight_History
  * band decomposition into between-curve and within-curve variance, score subsets, shift summaries
  * self-checks and the CSV writer of the result tables

Units
  t            time in s; the shared grid is x = log10(t + 1e-9) = GRID and t = 10**GRID - 1e-9 = T_GRID
  normalized   current divided by the scale factor of the record (dimensionless), the GPR data units
  real         current in A cm^-2 as recorded ("Surface Area: 1", numerically A on the nominal 1 cm2 area)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_scripts"
OUTPUT_DIR = EVAL / "output"
INPUT = REPO / "_input"
RAW = INPUT / "CCNF_CTAB_PT"
RUN = REPO / "_output" / "CCNF_CTAB_PT_output"
IND = RUN / "individual_GPR"
BT = RUN / "batch_testing"

VARIANT = {
    "gprfda": "NS_norm__WM_iterative__WS_curve",
    "pointwise": "NS_norm__WM_iterative__WS_point",
    "direct_obs": "NS_real__WM_equal__WS_curve",
    "direct_norm": "NS_norm__WM_equal__WS_curve",
}
METHODS = ("gprfda", "pointwise", "direct_obs", "direct_norm")


def summary_dir(method: str) -> Path:
    """batch_testing/<variant>/summary_gpr/iterative of a method name of VARIANT (or of a variant name)."""
    v = VARIANT.get(method, method)
    return BT / v / "summary_gpr" / "iterative"


MAIN = summary_dir("gprfda")
PW = summary_dir("pointwise")

POTENTIALS = ("-1.75", "-1.85", "-1.95", "-2.05", "-2.15", "-2.25", "-2.35")
MULTI_CURVE_POTENTIALS = ("-1.75", "-1.85", "-1.95", "-2.05", "-2.15")
N_CURVES = {"-1.75": 4, "-1.85": 4, "-1.95": 154, "-2.05": 9, "-2.15": 12, "-2.25": 1, "-2.35": 1}

EPS = 1e-12
TOL = 1e-6
POINTWISE_MAX_ITER = 10000
PIPELINE_POINTWISE_CAP = 50
X_SHIFT = 1e-9
X_CAPS = (1e-4, 5000.0)
GRID = np.linspace(-1.0, 3.69897, 500)
T_GRID = 10 ** GRID - X_SHIFT
T10_GRID = 10 ** GRID
N_GRID = len(GRID)
T_COMPLETE = 2990.0
T_MIN_RECORD = 60.0

DECADE_WINDOWS = (("0.1-1 s", 0.1, 1.0, "[)"), ("1-10 s", 1.0, 10.0, "[)"), ("10-100 s", 10.0, 100.0, "[)"),
                  ("100-1000 s", 100.0, 1000.0, "[)"), ("1000-5000 s", 1000.0, 5000.5, "[)"))


def window_masks(windows, t: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """{label: boolean mask over t} for windows (label, lo, hi, rule); rule '(]' = lo < t <= hi, '[)' = lo <= t < hi,
    None = open end. t defaults to T_GRID."""
    t = T_GRID if t is None else np.asarray(t, float)
    out = {}
    for lab, lo, hi, rule in windows:
        m = np.ones(t.shape, bool)
        if rule == "(]":
            if lo is not None:
                m &= t > lo
            if hi is not None:
                m &= t <= hi
        else:
            if lo is not None:
                m &= t >= lo
            if hi is not None:
                m &= t < hi
        out[lab] = m
    return out


def pkey(pot: Union[str, float]) -> str:
    """Potential key of the file names: -1.95, '-1.95' or '-1.95V' -> '-1.95'."""
    return f"{float(str(pot).rstrip('Vv')):.2f}"


def rel(p: Union[str, Path]) -> str:
    """Repository-relative POSIX path."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def read_header(path: Union[str, Path]) -> Dict[str, str]:
    """'# key: value' header lines at the top of a pipeline CSV -> {key: value}."""
    meta = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            s = line.lstrip("#").strip()
            if ":" in s:
                k, v = s.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta


def load_weights(pot, method: str = "gprfda") -> pd.DataFrame:
    """Converged_Weights_potential_<P>.csv: sample_id, index_id, converged_weight (the curve order used everywhere)."""
    return pd.read_csv(summary_dir(method) / f"Converged_Weights_potential_{pkey(pot)}.csv")


def load_summary(pot, method: str = "gprfda") -> pd.DataFrame:
    """Summary_GPR_potential_<P>.csv: x_transformed, x_real (s), y_real (A cm^-2), y_normalised, Std_real,
    Upper/Lower_CI_real, Std_normalised, Upper/Lower_CI_normalised."""
    return pd.read_csv(summary_dir(method) / f"Summary_GPR_potential_{pkey(pot)}.csv", comment="#")


def load_weight_history(pot, method: str = "gprfda") -> pd.DataFrame:
    """Weight_History_potential_<P>.csv. Curve-wise: one row per iteration with the weights at the start of that
    iteration. Point-wise: the initial uniform row and one row per iteration (curve means of the point weights)."""
    return pd.read_csv(summary_dir(method) / f"Weight_History_potential_{pkey(pot)}.csv")


def load_individual(sid: str) -> Dict[str, Any]:
    """Individual_GPR_<sid>.csv -> dict(mean, sd: normalized posterior on the grid, scale: scale factor (A cm^-2),
    x: x_pred_transformed, t: x_pred_original (s), header)."""
    f = IND / f"Individual_GPR_{sid}.csv"
    hdr = read_header(f)
    fac = float(json.loads(hdr["y_scaling_params"])["factor"])
    if fac != float(hdr["physical_scale_factor"]):
        raise ValueError(f"scale factor mismatch in {f.name}")
    d = pd.read_csv(f, comment="#")
    return dict(mean=d.y_pred_normalized.to_numpy(float), sd=d.y_std_normalized.to_numpy(float), scale=fac,
                x=d.x_pred_transformed.to_numpy(float), t=d.x_pred_original.to_numpy(float), header=hdr)


@dataclass
class Pool:
    """Posteriors of a set of curves on the shared grid. Y, S: m x 500 normalized posterior mean and sd;
    F: m scale factors (A cm^-2)."""
    pot: str
    sids: List[str]
    x: np.ndarray
    t: np.ndarray
    Y: np.ndarray
    S: np.ndarray
    F: np.ndarray

    @property
    def m(self) -> int:
        return len(self.sids)

    def index(self, sid: str) -> int:
        return self.sids.index(sid)


def load_pool(pot, order_method: str = "gprfda", sids: Optional[Sequence[str]] = None) -> Pool:
    """Pipeline posteriors of one potential in the order of Converged_Weights (the same order for every method)."""
    pot = pkey(pot)
    if sids is None:
        sids = load_weights(pot, order_method).sample_id.tolist()
    Y, S, F, x0, t0 = [], [], [], None, None
    for sid in sids:
        d = load_individual(sid)
        if x0 is None:
            x0, t0 = d["x"], d["t"]
            if np.abs(x0 - GRID).max() > 1e-12:
                raise ValueError("posterior grid differs from GRID")
        elif not np.array_equal(x0, d["x"]):
            raise ValueError(f"grid of {sid} differs")
        Y.append(d["mean"]); S.append(d["sd"]); F.append(d["scale"])
    return Pool(pot, list(sids), x0, t0, np.array(Y), np.array(S), np.array(F, float))


def _bounds(v):
    return v if v == "fixed" else tuple(float(b) for b in ast.literal_eval(str(v)))


def load_validation_results(pot) -> pd.DataFrame:
    """individual_GPR/Validation_Results_potential_<P>.csv with the kernel columns renamed to C_fit, l_fit, nu,
    noise_fit, lml_fit and their bounds as (lo, hi) tuples (StandardScaler fit space)."""
    d = pd.read_csv(IND / f"Validation_Results_potential_{pkey(pot)}.csv")
    ren = {"kernel1_kernel1_constant_value": "C_fit", "kernel1_kernel2_length_scale": "l_fit",
           "kernel1_kernel2_nu": "nu", "kernel2_noise_level": "noise_fit", "log_marginal_likelihood": "lml_fit"}
    d = d.rename(columns=ren)
    for c, b in (("C_fit", "kernel1_kernel1_constant_value_bounds"), ("l_fit", "kernel1_kernel2_length_scale_bounds"),
                 ("noise_fit", "kernel2_noise_level_bounds")):
        d[c + "_bounds"] = d[b].map(_bounds)
    return d


def diagnostic_path(sid: str) -> Path:
    return IND / f"Diagnostic_Data_{sid}.csv"


def pipeline_training_set(sid: str) -> Tuple[np.ndarray, np.ndarray]:
    """Training points of the pipeline fit (Diagnostic_Data rows with data_type 'training'):
    x = log10(t + 1e-9) and y = normalized current."""
    f = diagnostic_path(sid)
    n = int(read_header(f)["n_training"])
    d = pd.read_csv(f, comment="#", nrows=n)
    if not (d.data_type.astype(str) == "training").all() or len(d) != n:
        raise ValueError(f"unexpected Diagnostic_Data layout in {f.name}")
    x = pd.to_numeric(d.x.astype(str).str.strip().str.strip("[]")).to_numpy(float)
    return x, d.y_observed.to_numpy(float)


_COR: Dict[str, List[Path]] = {}


def raw_index() -> Dict[str, List[Path]]:
    """{file name: [paths]} of every .cor file under _input/CCNF_CTAB_PT."""
    if not _COR:
        for dp, _, fn in os.walk(RAW):
            for f in fn:
                if f.lower().endswith(".cor"):
                    _COR.setdefault(f, []).append(Path(dp) / f)
    return _COR


def split_sid(sid: str) -> Tuple[str, Optional[int]]:
    """'<file>.cor_curve_<k>' -> ('<file>.cor', k); '<file>.cor' -> ('<file>.cor', None)."""
    m = re.match(r"(.*\.cor)(?:_curve_(\d+))?$", sid)
    if not m:
        raise ValueError(f"not a .cor sample id: {sid}")
    return m.group(1), (int(m.group(2)) if m.group(2) else None)


def find_raw(sid: str) -> Path:
    name, _ = split_sid(sid)
    paths = raw_index().get(name, [])
    if len(paths) != 1:
        raise FileNotFoundError(f"{name}: {len(paths)} matches under {rel(RAW)}")
    return paths[0]


def parse_cor(path: Union[str, Path]) -> List[dict]:
    """Parse a .cor file with the rules of the pipeline loader (py_FDA_GPR_modules/pip0_dataloading): one dict per
    potentiostatic block with sample_id ('<file>_curve_<line index of Begin Experiment>' unless a 'Pstat Title'
    follows the block start), start_line, potential (V) and data_points in file order. Read as latin-1."""
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.readlines()
    curves, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if "Begin Experiment:" in line and "Potentiostatic" in line:
            start = i
            potential, sample_id, data_start, j = None, None, None, i
            for j in range(i + 1, min(i + 100, len(lines))):
                lj = lines[j].strip()
                if "Potential:" in lj:
                    try:
                        potential = float(lj.split(":", 1)[1].strip())
                    except ValueError:
                        potential = None
                elif "Pstat Title:" in lj:
                    sample_id = lj.split(":", 1)[1].strip()
                elif lj.startswith("Data Points:"):
                    for k in range(j + 1, len(lines)):
                        if lines[k].strip() == "End Comments":
                            data_start = k + 1
                            break
                    break
            if data_start is None:
                i = j
                continue
            pts, k = [], data_start
            for k in range(data_start, len(lines)):
                dl = lines[k].strip()
                if dl == "" or "End" in dl:
                    break
                v = re.split(r"[\t\s]+", dl)
                if len(v) >= 3:
                    try:
                        pts.append({"E (Volts)": float(v[0]), "I (A/cm2)": float(v[1]), "T (Seconds)": float(v[2])})
                    except ValueError:
                        continue
            if potential is not None and pts:
                curves.append(dict(file_path=Path(path), start_line=start, potential=potential,
                                   sample_id=sample_id if sample_id else f"{Path(path).name}_curve_{start}",
                                   data_points=pts))
            i = k
        else:
            i += 1
    return curves


def load_record(sid_or_path: Union[str, Path]) -> Tuple[pd.DataFrame, dict]:
    """Raw record of a sample id ('<file>.cor_curve_<k>') or of a .cor path -> (DataFrame with the columns
    'E (Volts)', 'I (A/cm2)', 'T (Seconds)' in file order, block dict of parse_cor)."""
    p = Path(sid_or_path)
    if p.suffix.lower() == ".cor" and p.exists():
        path, k = p, None
    else:
        path, k = find_raw(str(sid_or_path)), split_sid(str(sid_or_path))[1]
    curves = parse_cor(path)
    if k is not None:
        curves = [c for c in curves if c["start_line"] == k]
    if len(curves) != 1:
        raise RuntimeError(f"{path.name}: {len(curves)} matching potentiostatic blocks")
    return pd.DataFrame(curves[0]["data_points"]), curves[0]


def record_end(sid: str) -> float:
    """Record end (s): the last sample time of the raw record within 1e-4 <= t <= 5000 s."""
    df, _ = load_record(sid)
    T = np.sort(df["T (Seconds)"].to_numpy(float))
    T = T[(T >= X_CAPS[0]) & (T <= X_CAPS[1])]
    return float(T.max())


def record_ends(sids: Sequence[str]) -> np.ndarray:
    return np.array([record_end(s) for s in sids], float)


def record_meta(sid: str) -> Dict[str, Any]:
    """File-name metadata: potential_V, pH, batch ('acidB(<batch>)'), substrate (Ru or Au), run, retry ('', 'R',
    'R2', ...), substrate_note (text in parentheses after the run, e.g. 'bareSi'), date (yyyymmdd)."""
    name = split_sid(sid)[0] if ".cor" in sid else sid
    m = re.search(r"acidB\(([^)]*)\)", name)
    m2 = re.search(r"_(Ru|Au)-(\d+)([Rr]\d*)?(?:\(([^)]*)\))?_", name)
    d = re.search(r"(\d{8})\.cor", name)
    p = re.match(r"^(-?\d+\.\d+)V", name)
    ph = re.search(r"_pH(\d+\.\d+)_", name)
    return dict(sample_id=sid, file=name, potential_V=float(p.group(1)) if p else np.nan,
                pH=float(ph.group(1)) if ph else np.nan, batch=m.group(1) if m else "?",
                substrate=m2.group(1) if m2 else "?", run=int(m2.group(2)) if m2 else np.nan,
                retry=(m2.group(3) or "") if m2 else "", substrate_note=(m2.group(4) or "") if m2 else "",
                date=d.group(1) if d else "?")


def pool_meta(sids: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame([record_meta(s) for s in sids])


def contact_loss_indices(pool: Pool, n: int = 3) -> np.ndarray:
    """Sorted positions of the n records with the smallest |scale factor|. At -1.95 V these are the three
    contact-loss records, the records with GPR-FDA weight < 1e-8."""
    return np.sort(np.argsort(np.abs(pool.F))[:n])


def curve_weights(Y: np.ndarray, S: np.ndarray, *, tol: float = TOL, eps: float = EPS,
                  max_iter: Optional[int] = None, history: bool = False):
    """Curve-wise iterative weights (GPR-FDA): s2w_r = mean_j S_r(j)^2; w = 1/R; repeat ybar = sum_r w_r Y_r,
    D_r = mean_j (Y_r - ybar)^2, w_new = 1/(D_r + s2w_r + eps) normalized, delta = ||w_new - w|| / sqrt(R);
    stop when delta < tol. Returns (w, iterations, converged, history); history = the weights at the start of each
    iteration (as Weight_History) when history=True, else None."""
    Y = np.asarray(Y, float); S = np.asarray(S, float)
    R, N = Y.shape
    s2w = np.sum(S ** 2, axis=1) / N
    w = np.ones(R) / R
    hist = [] if history else None
    it, conv = 0, False
    cap = 1_000_000 if max_iter is None else int(max_iter)
    while True:
        it += 1
        yb = (w[:, None] * Y).sum(axis=0)
        if history:
            hist.append(w.copy())
        D = np.sum((Y - yb) ** 2, axis=1) / N
        wn = 1.0 / (D + s2w + eps)
        wn /= np.sum(wn)
        delta = np.linalg.norm(wn - w) / np.sqrt(wn.size)
        if delta < tol:
            w, conv = wn, True
            break
        if it >= cap:
            w = wn
            break
        w = wn.copy()
    return w, it, conv, hist


def point_weights(Y: np.ndarray, S: np.ndarray, cap: int, *, tol: float = TOL, eps: float = EPS,
                  W0: Optional[np.ndarray] = None):
    """Point-wise iterative weights with an explicit iteration cap: W = 1/m; repeat Ibar = sum_r W_r Y_r per grid
    point, W = 1/((Y - Ibar)^2 + S^2 + eps) normalized over the curves per grid point, delta = ||W - W_prev|| /
    sqrt(m N); stop when delta < tol or after `cap` iterations. cap = PIPELINE_POINTWISE_CAP (50) gives the pipeline
    point-wise run; cap = POINTWISE_MAX_ITER runs to convergence. W0: start weights. Returns (W, iterations,
    converged)."""
    if cap is None:
        raise ValueError("point-wise aggregation needs an explicit iteration cap")
    Y = np.asarray(Y, float); S = np.asarray(S, float)
    m, n = Y.shape
    W = np.full((m, n), 1.0 / m) if W0 is None else np.array(W0, float, copy=True)
    it, conv = 0, False
    for it in range(1, int(cap) + 1):
        Wp = W.copy()
        Ib = (W * Y).sum(axis=0, keepdims=True)
        W = 1.0 / ((Y - Ib) ** 2 + S ** 2 + eps)
        W /= W.sum(axis=0, keepdims=True)
        delta = np.linalg.norm(W - Wp) / np.sqrt(W.size)
        if delta < tol:
            conv = True
            break
    return W, it, conv


@dataclass
class Aggregate:
    """Aggregate on the shared grid. mean_norm, band_norm: normalized current; mean_real, band_real: A cm^-2;
    band^2 = weighted between-curve variance + weighted posterior variance in each scale. W: raw weights, (m,)
    curve-wise and direct or (m, N) point-wise; w: curve weights normalized to 1 (W.mean(1) for point-wise);
    s_agg = w . F (A cm^-2)."""
    method: str
    mean_norm: np.ndarray
    mean_real: np.ndarray
    band_norm: np.ndarray
    band_real: np.ndarray
    W: np.ndarray
    w: np.ndarray
    s_agg: float
    iterations: int
    converged: bool
    n_eff: float
    n: int


def aggregate_arrays(Y: np.ndarray, S: np.ndarray, F: np.ndarray, method: str, *, cap: Optional[int] = None,
                     tol: float = TOL, eps: float = EPS, W0: Optional[np.ndarray] = None) -> Aggregate:
    """Aggregate of the curves Y, S (normalized posterior mean and sd) with scale factors F (A cm^-2):
      gprfda       curve-wise weights in normalized space; mean_norm = sum W Y; mean_real = mean_norm s_agg
      pointwise    point-wise weights in normalized space with the explicit iteration cap `cap`
      direct_obs   equal weights on the real-scale curves; mean_norm = mean_real / s_agg
      direct_norm  equal weights in normalized space
    Band in each scale: sqrt(sum W (Y - mean)^2 + sum W S^2) with Yr = Y F and Sr = |S F| for the real scale."""
    Y = np.asarray(Y, float); S = np.asarray(S, float); F = np.asarray(F, float)
    m = len(Y)
    if method == "gprfda":
        W, nit, conv, _ = curve_weights(Y, S, tol=tol, eps=eps)
        space = "norm"
    elif method == "pointwise":
        W, nit, conv = point_weights(Y, S, cap, tol=tol, eps=eps, W0=W0)
        space = "norm"
    elif method in ("direct_obs", "direct_norm"):
        W, nit, conv = np.full(m, 1.0 / m), 0, True
        space = "real" if method == "direct_obs" else "norm"
    else:
        raise ValueError(f"unknown method {method}")
    W2 = W[:, None] if W.ndim == 1 else W
    wvec = W if W.ndim == 1 else W.mean(axis=1)
    wvec = wvec / wvec.sum()
    s_agg = float(np.dot(wvec, F))
    Yr = Y * F[:, None]
    Sr = np.abs(S * F[:, None])
    if space == "norm":
        mn = (W2 * Y).sum(axis=0); mr = mn * s_agg
    else:
        mr = (W2 * Yr).sum(axis=0); mn = mr / s_agg
    bn = (W2 * (Y - mn) ** 2).sum(axis=0); wn_ = (W2 * S ** 2).sum(axis=0)
    br = (W2 * (Yr - mr) ** 2).sum(axis=0); wr_ = (W2 * Sr ** 2).sum(axis=0)
    return Aggregate(method=method, mean_norm=mn, mean_real=mr, band_norm=np.sqrt(bn + wn_), band_real=np.sqrt(br + wr_),
                     W=W, w=wvec, s_agg=s_agg, iterations=int(nit), converged=bool(conv),
                     n_eff=float(1.0 / np.sum(wvec ** 2)), n=m)


def aggregate(pool: Pool, method: str, idx=None, *, cap: Optional[int] = None, tol: float = TOL, eps: float = EPS,
              W0: Optional[np.ndarray] = None) -> Aggregate:
    """aggregate_arrays on a Pool, optionally on the positions or boolean mask idx."""
    if idx is None:
        idx = np.arange(pool.m)
    idx = np.asarray(idx)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    return aggregate_arrays(pool.Y[idx], pool.S[idx], pool.F[idx], method, cap=cap, tol=tol, eps=eps, W0=W0)


def aggregate_all(pool: Pool, idx=None, *, pointwise_cap: int = POINTWISE_MAX_ITER,
                  methods: Sequence[str] = METHODS) -> Dict[str, Aggregate]:
    """{method: Aggregate} for the four methods; point-wise runs to convergence unless pointwise_cap is given."""
    return {m: aggregate(pool, m, idx, cap=pointwise_cap if m == "pointwise" else None) for m in methods}


def n_eff(w: np.ndarray) -> float:
    """Effective number of curves 1 / sum(w^2) of a weight vector normalized to 1."""
    w = np.asarray(w, float)
    w = w / w.sum()
    return float(1.0 / np.sum(w ** 2))


GROUPS = ("complete", "mid", "lt60")


def record_groups(tend: np.ndarray, t_complete: float = T_COMPLETE, t_min: float = T_MIN_RECORD) -> Dict[str, np.ndarray]:
    """Curve groups by record end: complete (>= 2990 s), mid (60 to 2990 s) and lt60 (< 60 s)."""
    tend = np.asarray(tend, float)
    return {"complete": tend >= t_complete, "mid": (tend >= t_min) & (tend < t_complete), "lt60": tend < t_min}


def band_terms(Y: np.ndarray, S: np.ndarray, w: np.ndarray, tend: np.ndarray) -> pd.DataFrame:
    """Band variance in normalized current split into the between-curve term sum_r w_r (Y_r - ybar)^2 and the
    within-curve term sum_{r in g} w_r S_r^2 of each record group g (complete, mid, lt60), each record counted as a
    whole curve, on the 500-point grid. Returns t_s (= 10**GRID), the variance terms ((normalized current)^2), the
    shares of the band variance and the stacked percentages of the within-curve shares (complete; complete + mid;
    complete + mid + lt60)."""
    Y = np.asarray(Y, float); S = np.asarray(S, float); w = np.asarray(w, float)
    W = w[:, None]
    yb = (W * Y).sum(axis=0)
    btw = (W * (Y - yb) ** 2).sum(axis=0)
    grp = record_groups(tend)
    wth = {g: (W[m] * S[m] ** 2).sum(axis=0) for g, m in grp.items()}
    tot = btw + sum(wth.values())
    out = pd.DataFrame({"t_s": T10_GRID, "var_total_norm2": tot, "var_between_norm2": btw,
                        "var_within_norm2": tot - btw})
    for g in GROUPS:
        out[f"var_within_{g}_norm2"] = wth[g]
    out["share_between"] = btw / tot
    for g in GROUPS:
        out[f"share_within_{g}"] = wth[g] / tot
    c = np.zeros(N_GRID)
    for g, name in zip(GROUPS, ("stack_complete_pct", "stack_complete_mid_pct", "stack_complete_mid_lt60_pct")):
        c = c + 100 * out[f"share_within_{g}"].to_numpy()
        out[name] = c
    return out


def band_window_medians(terms: pd.DataFrame, windows=DECADE_WINDOWS) -> pd.DataFrame:
    """Median shares of the band variance per time window ([lo, hi) on t = 10**GRID)."""
    masks = window_masks(windows, terms.t_s.to_numpy(float))
    rows = []
    for lab, m in masks.items():
        d = dict(window=lab, n_grid=int(m.sum()), share_between_median=float(np.median(terms.share_between[m])),
                 share_within_total_median=float(np.median(1 - terms.share_between[m])))
        for g in GROUPS:
            d[f"share_within_{g}_median"] = float(np.median(terms[f"share_within_{g}"][m]))
        rows.append(d)
    return pd.DataFrame(rows)


SUBSET_SIZES = {"best5": 5, "best10": 10, "best30": 30, "initial5": 5, "initial10": 10, "initial30": 30}
T_INIT = (0.1, 1.0)


def select_subsets(pool: Pool, w: np.ndarray, ybar_norm: np.ndarray,
                   sizes: Dict[str, int] = SUBSET_SIZES) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Score subsets of a pool with GPR-FDA weights w and aggregate ybar_norm (normalized current):
      bestN     the N curves with the largest weights
      initialN  the N curves with the smallest d_init = mean over t = 0.1 s (grid point 0) and t = 1 s of
                (Y_r(t) - ybar(t))^2; the 1 s value is interpolated linearly in log10 t between the neighbouring
                grid points
    Each cut is checked for ties. Returns ({key: boolean mask of the subset}, per-curve table)."""
    w = np.asarray(w, float); ybar = np.asarray(ybar_norm, float); Y = pool.Y; x = pool.x
    if np.abs(w @ Y - ybar).max() >= 1e-10:
        raise AssertionError("aggregate not reproduced by w @ Y")
    if abs(x[0] - np.log10(T_INIT[0])) >= 1e-12:
        raise AssertionError("first grid point is not 0.1 s")
    lx = np.log10(T_INIT[1])
    j = int(np.searchsorted(x, lx))
    a = (lx - x[j - 1]) / (x[j] - x[j - 1])
    y1 = (1 - a) * Y[:, j - 1] + a * Y[:, j]
    ybar1 = (1 - a) * ybar[j - 1] + a * ybar[j]
    t = pool_meta(pool.sids)[["sample_id", "batch", "substrate", "run", "retry", "date"]].copy()
    t["weight"] = w
    t["weight_n"] = w * len(w)
    t["weight_rank"] = pd.Series(w).rank(ascending=False, method="first").astype(int).to_numpy()
    t["y_norm_t0p1"] = Y[:, 0]
    t["y_norm_t1"] = y1
    t["aggregate_norm_t0p1"] = ybar[0]
    t["aggregate_norm_t1"] = ybar1
    t["deviation_norm_t0p1"] = Y[:, 0] - ybar[0]
    t["deviation_norm_t1"] = y1 - ybar1
    D = (t["deviation_norm_t0p1"] ** 2 + t["deviation_norm_t1"] ** 2) / 2
    t["d_init"] = D
    t["initial_rank"] = D.rank(ascending=True, method="first").astype(int)
    ws = np.sort(w)[::-1]; ds = np.sort(D.to_numpy())
    masks = {}
    for k, n in sizes.items():
        if k.startswith("best"):
            if not ws[n - 1] > ws[n]:
                raise AssertionError(f"tie at the weight cut of {k}")
            masks[k] = t.weight_rank.to_numpy() <= n
        else:
            if not ds[n - 1] < ds[n]:
                raise AssertionError(f"tie at the initial-stage cut of {k}")
            masks[k] = t.initial_rank.to_numpy() <= n
        t[k] = masks[k]
    return masks, t


def at_bound(value, bounds, rtol_log: float = 1e-3) -> bool:
    """True when |ln(value / bound)| < 1e-3 for either bound (fit space)."""
    lo, hi = bounds
    return bool(abs(np.log(value / lo)) < rtol_log or abs(np.log(value / hi)) < rtol_log)


def shift_summary(ref: Dict[str, Aggregate], var: Dict[str, Aggregate], band: np.ndarray,
                  sel: Optional[np.ndarray] = None, x_var: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Shift of each method's aggregate mean (current) from its reference, divided by `band` per grid point:
    <method>_max = max |z| and <method>_rms = sqrt(mean z^2) over the grid points `sel` (default: all), plus the
    point-wise iteration count and convergence of the variant and the numbers of records. x_var: grid (log10 t) of
    the variant when it differs from GRID; the variant mean is then interpolated onto GRID[sel]."""
    sel = np.ones(N_GRID, bool) if sel is None else np.asarray(sel, bool)
    band = np.asarray(band, float)
    out: Dict[str, Any] = {}
    for m in METHODS:
        b = var[m].mean_real[sel] if x_var is None else np.interp(GRID[sel], x_var, var[m].mean_real)
        z = (b - ref[m].mean_real[sel]) / band[sel]
        out[f"{m}_max"] = float(np.abs(z).max())
        out[f"{m}_rms"] = float(np.sqrt(np.mean(z ** 2)))
    out.update(pointwise_iterations=int(var["pointwise"].iterations),
               pointwise_converged=int(bool(var["pointwise"].converged)), n_reference=int(ref["gprfda"].n),
               n_variant=int(var["gprfda"].n))
    return out


CHECKS: List[Dict[str, Any]] = []


def check(name: str, passed: bool, detail: str = "") -> bool:
    """Record and print a self-check of the computation."""
    CHECKS.append(dict(name=name, passed=bool(passed), detail=str(detail)))
    print(f"check {'passed' if passed else 'FAILED'}: {name}" + (f" ({detail})" if detail else ""), flush=True)
    return bool(passed)


def checks_failed() -> int:
    return sum(not c["passed"] for c in CHECKS)


_TABLE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_COLUMN_NAME = re.compile(r"^[a-z][a-z0-9]*(?:_(?:[a-z0-9]+|mA))*$")


def output_dir(script_file: Union[str, Path]) -> Path:
    """eval_scripts/output/<test name>, where the test name is the script name without the 'eval_' prefix."""
    stem = Path(script_file).stem
    return OUTPUT_DIR / (stem[len("eval_"):] if stem.startswith("eval_") else stem)


def strict_table(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a result table: lower-case snake_case column names (the unit token 'mA' allowed), numeric columns
    (booleans become 0/1), and text only in an optional last column 'notes'."""
    df = pd.DataFrame(df).reset_index(drop=True).copy()
    cols = [str(c) for c in df.columns]
    if not cols:
        raise ValueError("empty table")
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names: {cols}")
    for c in cols:
        if not _COLUMN_NAME.match(c):
            raise ValueError(f"column name {c!r} is not lower-case snake_case")
    if "notes" in cols and cols[-1] != "notes":
        raise ValueError("'notes' must be the last column")
    df.columns = cols
    for c in cols:
        s = df[c]
        if c == "notes":
            df[c] = s.map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        elif pd.api.types.is_bool_dtype(s):
            df[c] = s.astype(int)
        elif not pd.api.types.is_numeric_dtype(s):
            raise TypeError(f"column {c!r} is not numeric (text belongs in 'notes')")
    return df


def write_tables(script_file: Union[str, Path], tables: Dict[str, pd.DataFrame]) -> List[Path]:
    """Write each table to eval_scripts/output/<test name>/<table name>.csv: one header row followed by the data rows,
    comma separated, UTF-8, no index. CSV files of earlier runs that are not in `tables` are removed."""
    frames = {}
    for name, df in tables.items():
        if not _TABLE_NAME.match(name):
            raise ValueError(f"table name {name!r} is not lower-case snake_case")
        frames[name] = strict_table(df)
    out = output_dir(script_file)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.csv"):
        if old.stem not in frames:
            old.unlink()
    paths = []
    for name, df in frames.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False, lineterminator="\n", encoding="utf-8")
        paths.append(p)
        print(f"wrote {rel(p)} ({len(df)} rows x {df.shape[1]} columns)", flush=True)
    return paths
