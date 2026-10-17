"""Refit helpers of the evaluation scripts (scikit-learn, joblib, threadpoolctl).

Settings of the individual GPR fits (the pipeline settings unless a test passes others):
  preprocessing  record sorted by T; keep 1e-4 <= T <= 5000 s; scale factor s_r = mean of the raw current over the
                 samples int(0.5 n) .. int(0.9 n) - 1 of the capped record (by index); x = log10(T + 1e-9);
                 y = I / s_r; drop |y| < 0.01; at most 500 training points, the middle sample of each bin of a uniform
                 grid in x (downsample_uniform_bins of the pipeline); records shorter than 60 s are skipped unless
                 apply_min_range=False
  regression     StandardScaler on x and on y (ddof 0); GaussianProcessRegressor(normalize_y=True, alpha=0, the
                 WhiteKernel carrying the noise), kernel ConstantKernel(1.0, (1e-2, 1e4)) * Matern(1.0, (1e-2, 1e3), nu)
                 + WhiteKernel(1.0, (1e-12, 10)), L-BFGS-B with 5 restarts, random_state 0
  prediction     training-only GPR on the shared grid GRID (500 points, log10 t from -1 to 3.69897); posterior sd from
                 predict(return_cov=True) (std_from 'cov') or predict(return_std=True) (std_from 'std');
                 normalized mean = mu~ y_std + y_mean, sd = sd~ y_std
The fitted hyperparameters and the log marginal likelihood are returned in StandardScaler fit space.

fit_jobs() and pmap() run every call with a fixed number of BLAS threads (threadpoolctl). The thread count changes the
round-off of the Cholesky factorizations, which on a flat likelihood can move the optimizer to another optimum, so each
test fixes it. Random downsampling uses the global numpy random state and runs in the main process.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from _eval_common import GRID, X_CAPS, X_SHIFT, T_MIN_RECORD

KERNELS = ("M0.5", "M1.5", "M2.5", "RBF")
N_RESTARTS = 5
RANDOM_STATE = 0
MAX_POINTS = 500
NORM_WINDOW = (0.5, 0.9)
Y_THRESHOLD = 0.01
X_COL, Y_COL = "T (Seconds)", "I (A/cm2)"
COV_MAX_POINTS = 5000


def downsample_uniform_bins(df: pd.DataFrame, max_points: int, x_col: str = "x_transformed",
                            selection: str = "middle") -> Tuple[pd.DataFrame, np.ndarray]:
    """Downsampling of the pipeline (pip1_datapreprocessing/preprc_downsampling): num_bins = min(max_points, len);
    bin edges np.linspace(min x, max x, num_bins + 1); bin = digitize - 1 clipped to [0, num_bins - 1]; one sample per
    non-empty bin: 'middle' = x.iloc[len // 2] or 'random' = x.iloc[np.random.randint(len)] (global numpy random
    state)."""
    df = df.copy()
    num_bins = min(max_points, len(df))
    if num_bins >= len(df):
        return df, df.index.to_numpy()
    x_values = np.asarray(df[x_col].values)
    bin_edges = np.linspace(x_values.min(), x_values.max(), num_bins + 1)
    df["_bin_ds"] = np.digitize(x_values, bin_edges) - 1
    df["_bin_ds"] = np.clip(df["_bin_ds"], 0, num_bins - 1)
    if selection == "middle":
        selector = lambda x: x.iloc[len(x) // 2].name
    elif selection == "random":
        selector = lambda x: x.iloc[np.random.randint(len(x))].name
    else:
        raise ValueError(f"Unknown selection method: {selection}")
    try:
        idx = df.groupby("_bin_ds").apply(selector, include_groups=False)
    except TypeError:
        idx = df.groupby("_bin_ds").apply(selector)
    out = df.loc[idx.values].drop(columns=["_bin_ds"])
    return out, idx.values


def preprocess_record(df: pd.DataFrame, *, window: Tuple[float, float] = NORM_WINDOW, max_points: int = MAX_POINTS,
                      x_axis: str = "log10", selection: str = "middle", edge_points: bool = False,
                      apply_min_range: bool = True, y_threshold: float = Y_THRESHOLD) -> Tuple[Optional[dict], Optional[str]]:
    """Preprocessing of one raw record (columns 'E (Volts)', 'I (A/cm2)', 'T (Seconds)', e.g. from load_record).
    window: normalization window as sample-index fractions (a, b), s_r = mean I[int(a n):int(b n)];
    max_points: number of uniform bins; x_axis: 'log10' (x = log10(T + 1e-9)) or 'linear' (x = T in s);
    selection: 'middle' or 'random' (global np.random state, seeded by the caller); edge_points: add the first and
    last samples of the filtered record; apply_min_range: skip records whose raw T range is below 60 s.
    Returns (dict, None) or (None, reason). dict: x_train, y_train, t_train, x_full, y_full, t_full (filtered record),
    scale (s_r, A cm^-2), n_raw, n_capped, n_after_threshold, n_train, n_removed_threshold, x_range_s, tmax_s, tmin_s."""
    d = df.sort_values(by=X_COL).reset_index(drop=True)
    x_range = float(d[X_COL].max() - d[X_COL].min())
    if apply_min_range and x_range < T_MIN_RECORD:
        return None, f"X range {x_range:.2f} < min_curve_range ({T_MIN_RECORD})"
    d = d[(d[X_COL] >= X_CAPS[0]) & (d[X_COL] <= X_CAPS[1])].copy()
    if d.empty:
        return None, "No data after applying x caps"
    x0 = d[X_COL].to_numpy(float); y0 = d[Y_COL].to_numpy(float)
    n = len(y0); i0, i1 = int(window[0] * n), int(window[1] * n)
    if i0 >= i1:
        return None, "Y scaling failed (middle window empty)"
    s = float(np.mean(y0[i0:i1]))
    if s == 0 or np.isnan(s):
        return None, "Y scaling failed (zero middle average)"
    if x_axis == "log10":
        d["x_transformed"] = np.log10(x0 + X_SHIFT)
    elif x_axis == "linear":
        d["x_transformed"] = x0
    else:
        raise ValueError(f"x_axis must be 'log10' or 'linear', not {x_axis!r}")
    d["y_transformed"] = y0 / s
    d["t_s"] = x0
    dn = d[d["y_transformed"].abs() >= y_threshold].copy()
    if dn.empty:
        return None, "All data points filtered out after normalization"
    dds, _ = downsample_uniform_bins(dn, max_points, "x_transformed", selection)
    if edge_points:
        add = [i for i in (dn.index[0], dn.index[-1]) if i not in dds.index]
        dds = pd.concat([dds, dn.loc[add, dds.columns]]).sort_values("t_s")
    return dict(x_train=dds["x_transformed"].to_numpy(float), y_train=dds["y_transformed"].to_numpy(float),
                t_train=dds["t_s"].to_numpy(float), x_full=dn["x_transformed"].to_numpy(float),
                y_full=dn["y_transformed"].to_numpy(float), t_full=dn["t_s"].to_numpy(float), scale=s,
                n_raw=len(df), n_capped=n, n_after_threshold=len(dn), n_train=len(dds), n_removed_threshold=n - len(dn),
                x_range_s=x_range, tmax_s=float(x0.max()), tmin_s=float(x0.min()), window=tuple(window),
                max_points=max_points, x_axis=x_axis, selection=selection, edge_points=edge_points), None


def _kind(kernel: Union[str, float]) -> str:
    k = str(kernel).upper().replace("MATERN", "M").replace(" ", "")
    if k in ("RBF", "SE"):
        return "RBF"
    k = k.lstrip("M")
    if k in ("0.5", "1.5", "2.5"):
        return "M" + k
    raise ValueError(f"unknown kernel {kernel!r}; use one of {KERNELS}")


def make_kernel(kernel: Union[str, float] = "M1.5"):
    """ConstantKernel(1.0, (1e-2, 1e4)) * Matern(1.0, (1e-2, 1e3), nu) + WhiteKernel(1.0, (1e-12, 10));
    kernel = 'M0.5' | 'M1.5' | 'M2.5' (also '1.5', 1.5, 'Matern1.5') or 'RBF' (RBF(1.0, (1e-2, 1e3)) in place of the
    Matern, same constant and white terms)."""
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
    k = _kind(kernel)
    c = ConstantKernel(1.0, (1e-2, 1e4))
    wk = WhiteKernel(1.0, (1e-12, 1e1))
    if k == "RBF":
        return c * RBF(1.0, (1e-2, 1e3)) + wk
    return c * Matern(1.0, (1e-2, 1e3), nu=float(k[1:])) + wk


@dataclass
class FitResult:
    """One GPR fit. mu, sd: normalized posterior mean and sd on x_pred (default GRID); pred: {name: (mu, sd)} of the
    further prediction sets; C_fit, l_fit, noise_fit, lml_fit: hyperparameters and log marginal likelihood in
    StandardScaler fit space; fit_s, predict_s: wall times of the fit and of the predictions."""
    tag: Any
    kernel: str
    n_train: int
    mu: np.ndarray
    sd: np.ndarray
    pred: Dict[str, Tuple[np.ndarray, np.ndarray]]
    C_fit: float
    l_fit: float
    noise_fit: float
    lml_fit: float
    fit_s: float
    predict_s: float


def _predict(g, sx, sy, xp, std_from: str):
    Xp = sx.transform(np.asarray(xp, float).reshape(-1, 1))
    if std_from == "cov" and len(Xp) <= COV_MAX_POINTS:
        m, cov = g.predict(Xp, return_cov=True)
        sd = np.sqrt(np.diag(cov))
    else:
        m, sd = g.predict(Xp, return_std=True)
    return m * sy.scale_[0] + sy.mean_[0], sd * sy.scale_[0]


def fit_curve(x, y, kernel: Union[str, float] = "M1.5", x_pred: Optional[np.ndarray] = None, *,
              extra_pred: Optional[Dict[str, np.ndarray]] = None, n_restarts: int = N_RESTARTS,
              random_state: Optional[int] = RANDOM_STATE, std_from: str = "cov", tag: Any = None,
              alpha: float = 0.0, normalize_y: bool = True) -> FitResult:
    """Individual GPR of one training set (pipeline perform_gpr and generate_predictions).
    x: training abscissa (log10(t + 1e-9), or t in s for a linear time axis); y: normalized current.
    x_pred: prediction abscissa in the units of x (default GRID). extra_pred: {name: abscissa} of further prediction
    sets; sets larger than 5000 points use return_std. std_from: 'cov' or 'std'. alpha, normalize_y: 0.0 and True
    unless a test sets them. Optimizer warnings are not shown."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler
    t0 = time.perf_counter()
    X = np.asarray(x, float).reshape(-1, 1)
    yv = np.asarray(y, float).ravel()
    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(yv.reshape(-1, 1)).ravel()
    k0 = make_kernel(kernel)
    g = GaussianProcessRegressor(kernel=k0, n_restarts_optimizer=n_restarts, alpha=alpha, normalize_y=normalize_y,
                                 random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g.fit(Xs, ys)
    t1 = time.perf_counter()
    mu, sd = _predict(g, sx, sy, GRID if x_pred is None else x_pred, std_from)
    pred = {name: _predict(g, sx, sy, xp, std_from) for name, xp in (extra_pred or {}).items()}
    t2 = time.perf_counter()
    kk = g.kernel_
    C, l, nz = float(kk.k1.k1.constant_value), float(np.ravel(kk.k1.k2.length_scale)[0]), float(kk.k2.noise_level)
    lml = float(g.log_marginal_likelihood_value_)
    return FitResult(tag=tag, kernel=_kind(kernel), n_train=len(yv), mu=mu, sd=sd, pred=pred, C_fit=C, l_fit=l,
                     noise_fit=nz, lml_fit=lml, fit_s=t1 - t0, predict_s=t2 - t1)


def _limited(func: Callable, item, threads: int = 1):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=int(threads)):
        return func(item)


def pmap(func: Callable, items: Sequence[Any], n_jobs: int = 4, verbose: int = 5, blas_threads: int = 1) -> List[Any]:
    """[func(item) for item in items] in input order with `blas_threads` BLAS threads per call; n_jobs > 1 uses joblib
    loky workers. func must be importable by the workers (module level in the eval script or in these helpers)."""
    items = list(items)
    if n_jobs is None or n_jobs <= 1 or len(items) <= 1:
        return [_limited(func, it, blas_threads) for it in items]
    from joblib import Parallel, delayed
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return Parallel(n_jobs=n_jobs, backend="loky", batch_size=1, verbose=verbose)(
        delayed(_limited)(func, it, blas_threads) for it in items)


def _fit_kw(kw: Dict[str, Any]) -> FitResult:
    kw = dict(kw)
    return fit_curve(kw.pop("x"), kw.pop("y"), **kw)


def fit_jobs(jobs: Sequence[Dict[str, Any]], n_jobs: int = 4, verbose: int = 5, blas_threads: int = 1) -> List[FitResult]:
    """fit_curve for each job dict (keys: x, y and any fit_curve keyword, e.g. kernel, x_pred, extra_pred, tag,
    std_from) with `blas_threads` BLAS threads per fit; n_jobs > 1 runs the fits in parallel workers. Returns the
    FitResults in job order."""
    return pmap(_fit_kw, jobs, n_jobs=n_jobs, verbose=verbose, blas_threads=blas_threads)
