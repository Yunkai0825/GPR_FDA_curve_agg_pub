"""Fitted hyperparameters of the individual GPR fits in GPR data units, fit errors and the bound check.

Method
  The pipeline fits each record in StandardScaler space: x~ = (x - x_mean)/x_std with x = log10(t + 1e-9) and
  y~ = (y - y_mean)/y_std with y = normalized current, kernel C * Matern(l, nu = 1.5) + WhiteKernel(noise).
  For the 154 records at -1.95 V the fitted kernel of individual_GPR/Validation_Results_potential_-1.95.csv is
  rebuilt with its bounds and converted to data units with hyperparams_to_physical of
  py_FDA_GPR_modules/pip2_individual_gpr/gpr_functions.py:
    C and noise x (y_std * s)^2  in (normalized current)^2,   l x x_std  in decades of t,
  x_mean and x_std (ddof 0) from the training rows of Diagnostic_Data_<sample>.csv, y_mean and y_std from the
  statistical_scaler_mean / statistical_scaler_std header of Individual_GPR_<sample>.csv, and s = std(y~) of the
  training rows when normalize_y is set in batch_testing/settings_used.json (else 1).
  Amplitudes: Matern amplitude sqrt(C) and white-noise amplitude sqrt(noise), in normalized current.
  Fit errors: MAE and RMSE of Validation_Results (normalized current).
  Contact-loss records: the 3 records with the smallest converged GPR-FDA weight at -1.95 V (Converged_Weights of
  batch_testing/NS_norm__WM_iterative__WS_curve).
  Bound check: a fitted value counts as at a bound when |ln(value/bound)| < 1e-3 for either bound, in fit space,
  for every record of Validation_Results at the seven potentials. Bounds: C (1e-2, 1e4), l (1e-2, 1e3),
  noise (1e-12, 10).
  Self-checks: the rebuilt kernel prints as the pipeline kernel string; the header y scaler equals the mean and std
  of the training y; a GaussianProcessRegressor with the data-unit kernel, optimizer None, alpha 0 and
  normalize_y False, conditioned on the training rows offset by y_mean + y_std * mean(y~), reproduces the pipeline
  predictions at the training rows and the converted log marginal likelihood.

Output tables in output/hyperparameters/
  hyperparameters_per_curve   one row per record at -1.95 V: curve (position in the Converged_Weights order),
                              n_train, mae_norm, rmse_norm, signal_variance_fit, length_scale_fit, noise_level_fit
                              (fit space), signal_variance_norm2, noise_level_norm2 ((normalized current)^2),
                              length_scale_decades, matern_amplitude_norm, white_noise_amplitude_norm,
                              contact_loss (1 = one of the 3 contact-loss records), notes = sample id
  hyperparameter_summary      one row per quantity: fit_median, fit_min, fit_max (fit space, all 154 records;
                              empty for the two amplitudes), data_median (data units, all 154 records),
                              data_min_clean, data_max_clean (data units, the 151 records without contact loss),
                              data_min_contact_loss, data_max_contact_loss (data units, the 3 contact-loss records),
                              n_curves, n_clean, n_contact_loss, notes = quantity and data unit
  bound_check                 one row per potential: potential_v, n_curves, signal_variance_at_bound,
                              length_scale_at_bound, noise_level_at_bound (number of records)

Run
  python -X utf8 -B eval_scripts/eval_hyperparameters.py
"""
import sys

sys.dont_write_bytecode = True

import ast
import importlib.util
import json
import time
import types
import warnings

import numpy as np
import pandas as pd

import _eval_common as C

POT = "-1.95"
PIP2 = C.REPO / "py_FDA_GPR_modules" / "pip2_individual_gpr"
SETTINGS = C.BT / "settings_used.json"
N_CONTACT_LOSS = 3
QUANTITIES = (("signal_variance", "C", "(normalized current)^2"), ("length_scale", "l", "decades of t"),
              ("noise_level", "white-noise level", "(normalized current)^2"),
              ("matern_amplitude", "Matern amplitude sqrt(C)", "normalized current"),
              ("white_noise_amplitude", "white-noise amplitude sqrt(noise)", "normalized current"))
DATA_COLUMN = {"signal_variance": "signal_variance_norm2", "length_scale": "length_scale_decades",
               "noise_level": "noise_level_norm2", "matern_amplitude": "matern_amplitude_norm",
               "white_noise_amplitude": "white_noise_amplitude_norm"}


def load_pip2():
    """gpr_config and gpr_functions of pip2, loaded from their files under a stub package."""
    pkg = types.ModuleType("_pip2_repo")
    pkg.__path__ = [str(PIP2)]
    sys.modules["_pip2_repo"] = pkg
    mods = {}
    for name in ("gpr_config", "gpr_functions"):
        spec = importlib.util.spec_from_file_location(f"_pip2_repo.{name}", PIP2 / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["gpr_functions"]


def read_training_rows(sid):
    """Training rows of Diagnostic_Data_<sample>.csv with numeric x, y_observed and y_pred."""
    df = pd.read_csv(C.diagnostic_path(sid), comment="#")
    df = df[df.data_type.isin(["training", "validation"])].copy()
    for c in ("x", "y_observed", "y_pred"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip("[] "), errors="raise")
    return df[df.data_type == "training"]


def bounds(v):
    return v if v == "fixed" else tuple(float(b) for b in ast.literal_eval(v))


def fit_kernel(row):
    """Fit-space kernel of one Validation_Results row."""
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    return (ConstantKernel(row["kernel1_kernel1_constant_value"], bounds(row["kernel1_kernel1_constant_value_bounds"]))
            * Matern(row["kernel1_kernel2_length_scale"], bounds(row["kernel1_kernel2_length_scale_bounds"]),
                     nu=row["kernel1_kernel2_nu"])
            + WhiteKernel(row["kernel2_noise_level"], bounds(row["kernel2_noise_level_bounds"])))


def convert_record(gf, row, normalize_y):
    """Data-unit hyperparameters of one record and the self-check quantities."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    sid = row["sample_id"]
    hdr = C.read_header(C.IND / f"Individual_GPR_{sid}.csv")
    hp_header = json.loads(hdr["hyperparams"])
    tr = read_training_rows(sid)
    y_mean, y_std = float(hdr["statistical_scaler_mean"]), float(hdr["statistical_scaler_std"])
    xs, ys = tr.x.values, tr.y_observed.values
    x_mean, x_std = float(np.mean(xs)), float(np.std(xs))
    shared = bool(hp_header.get("shared_hyperparameters"))
    s_int = float(np.std((ys - y_mean) / y_std)) if (normalize_y and not shared) else 1.0
    kern = fit_kernel(row)
    n_lml = int(hp_header["fit_n_train"]) if "fit_n_train" in hp_header else len(tr)
    hp_ss = kern.get_params(deep=True)
    hp_ss["log_marginal_likelihood"] = float(row["log_marginal_likelihood"])
    hp_ss["log_marginal_likelihood_n_points"] = n_lml
    hp_ss["normalize_y_std"] = s_int
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, hp_ph = gf.hyperparams_to_physical(kern, hp_ss, x_mean=x_mean, x_scale=x_std, y_scale=y_std)
    k_ph = gf.kernel_to_physical(kern, x_mean, x_std, y_std * s_int)
    g = GaussianProcessRegressor(kernel=k_ph, optimizer=None, alpha=0.0, normalize_y=False)
    m_int = float(np.mean((ys - y_mean) / y_std)) if (normalize_y and not shared) else 0.0
    off = y_mean + y_std * m_int
    g.fit(xs.reshape(-1, 1), ys - off)
    pred = g.predict(xs.reshape(-1, 1)) + off
    C_d = float(hp_ph["k1__k1__constant_value"])
    noise_d = float(hp_ph["k2__noise_level"])
    return dict(
        sid=sid, n_train=len(tr), mae_norm=float(row["MAE"]), rmse_norm=float(row["RMSE"]),
        signal_variance_fit=float(row["kernel1_kernel1_constant_value"]),
        length_scale_fit=float(row["kernel1_kernel2_length_scale"]), noise_level_fit=float(row["kernel2_noise_level"]),
        signal_variance_norm2=C_d, noise_level_norm2=noise_d, length_scale_decades=float(hp_ph["k1__k2__length_scale"]),
        matern_amplitude_norm=float(np.sqrt(C_d)), white_noise_amplitude_norm=float(np.sqrt(noise_d)),
        kernel_string_matches=str(kern) == row["Optimized Hyperparameters"],
        shared=shared,
        y_std_rel=abs(float(np.std(ys)) / y_std - 1), y_mean_rel=abs(float(np.mean(ys)) - y_mean) / y_std,
        pred_rel=float(np.max(np.abs(pred - tr.y_pred.values)) / np.max(np.abs(tr.y_pred.values))),
        lml_abs=abs(float(g.log_marginal_likelihood_value_) - float(hp_ph["log_marginal_likelihood"])))


def main():
    t_start = time.time()
    gf = load_pip2()
    normalize_y = bool(json.loads(SETTINGS.read_text(encoding="utf-8"))["pip2_individual_gpr"]["gpr_params"]
                       .get("normalize_y", False))

    vres = pd.read_csv(C.IND / f"Validation_Results_potential_{POT}.csv")
    print(f"converting {len(vres)} records at {POT} V ...", flush=True)
    recs = {r["sid"]: r for r in (convert_record(gf, row, normalize_y) for _, row in vres.iterrows())}
    ck = pd.DataFrame(recs.values())
    C.check("rebuilt fit-space kernels print as the pipeline kernel strings", bool(ck.kernel_string_matches.all()),
            f"{int(ck.kernel_string_matches.sum())} of {len(ck)}")
    C.check("every record has its own fit (no shared hyperparameters)", not bool(ck.shared.any()))
    C.check("header y scaler = mean and std of the training y", max(ck.y_std_rel.max(), ck.y_mean_rel.max()) <= 1e-12,
            f"max rel {max(ck.y_std_rel.max(), ck.y_mean_rel.max()):.2g}")
    C.check("data-unit kernel reproduces the pipeline predictions at the training rows", ck.pred_rel.max() <= 1e-10,
            f"max rel {ck.pred_rel.max():.2g}")
    C.check("data-unit kernel reproduces the converted log marginal likelihood", ck.lml_abs.max() <= 1e-8,
            f"max abs {ck.lml_abs.max():.2g}")

    w = C.load_weights(POT)
    cl = set(w.sort_values("converged_weight").sample_id.head(N_CONTACT_LOSS))
    pool_sids = w.sample_id.tolist()
    C.check("Validation_Results and Converged_Weights hold the same records", set(pool_sids) == set(recs),
            f"{len(recs)} records")
    pool = C.load_pool(POT)
    C.check("contact-loss records are the records with the smallest scale factors",
            cl == {pool.sids[i] for i in C.contact_loss_indices(pool, N_CONTACT_LOSS)})

    cols = ["n_train", "mae_norm", "rmse_norm", "signal_variance_fit", "length_scale_fit", "noise_level_fit",
            "signal_variance_norm2", "noise_level_norm2", "length_scale_decades", "matern_amplitude_norm",
            "white_noise_amplitude_norm"]
    per = pd.DataFrame([dict(curve=i, **{c: recs[s][c] for c in cols}, contact_loss=int(s in cl), notes=s)
                        for i, s in enumerate(pool_sids)])
    C.check("training points per record <= 500", int(per.n_train.max()) <= 500,
            f"{int(per.n_train.min())} to {int(per.n_train.max())}")

    D = pd.DataFrame(recs.values()).set_index("sid")
    m = ~D.index.isin(cl)
    summary = []
    for key, label, unit in QUANTITIES:
        fit = D[f"{key}_fit"] if f"{key}_fit" in D else None
        dat = D[DATA_COLUMN[key]]
        summary.append(dict(
            fit_median=float(fit.median()) if fit is not None else np.nan,
            fit_min=float(fit.min()) if fit is not None else np.nan,
            fit_max=float(fit.max()) if fit is not None else np.nan,
            data_median=float(dat.median()), data_min_clean=float(dat[m].min()), data_max_clean=float(dat[m].max()),
            data_min_contact_loss=float(dat[~m].min()), data_max_contact_loss=float(dat[~m].max()),
            n_curves=len(dat), n_clean=int(m.sum()), n_contact_loss=int((~m).sum()), notes=f"{label}; {unit}"))
    summary = pd.DataFrame(summary)

    bc = []
    for pot in C.POTENTIALS:
        v = C.load_validation_results(pot)
        bc.append(dict(potential_v=float(pot), n_curves=len(v),
                       signal_variance_at_bound=sum(C.at_bound(x, b) for x, b in zip(v.C_fit, v.C_fit_bounds)),
                       length_scale_at_bound=sum(C.at_bound(x, b) for x, b in zip(v.l_fit, v.l_fit_bounds)),
                       noise_level_at_bound=sum(C.at_bound(x, b) for x, b in zip(v.noise_fit, v.noise_fit_bounds))))
    bc = pd.DataFrame(bc)
    C.check("records per potential", all(int(n) == C.N_CURVES[C.pkey(p)] for p, n in zip(bc.potential_v, bc.n_curves)),
            "/".join(str(int(n)) for n in bc.n_curves) + f", total {int(bc.n_curves.sum())}")

    C.write_tables(__file__, dict(hyperparameters_per_curve=per, hyperparameter_summary=summary, bound_check=bc))
    print(f"done in {time.time() - t_start:.0f} s; {C.checks_failed()} failed checks", flush=True)
    return 1 if C.checks_failed() else 0


if __name__ == "__main__":
    sys.exit(main())
