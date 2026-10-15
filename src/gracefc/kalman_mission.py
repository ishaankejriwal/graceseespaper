"""Two-variance Kalman filter: separate model-noise variances for GRACE and GRACE-FO.

The observation equation of gracefc.kalman gets one model-noise variance for months before
January 2018 and another from then on,

    y_t = x_t + eta_t,    eta_t ~ N(0, r_grace)   (months before 2018-01)
    y_t = x_t + eta_t,    eta_t ~ N(0, r_fo)      (months from 2018-01)

so the parameter vector is (rho, q, r_grace, r_fo). Everything else follows gracefc.kalman:
maximum likelihood with L-BFGS-B from four starts on training months only, and the forecast
at lead time h is rho**h times the filtered state at the issue month (code id
kalman_mission).
All test months are GRACE-FO months.

The training window of the first fold ends in May 2019 and contains only ten observed
GRACE-FO months, too few to separate a second variance from q. A basin whose training window
holds fewer than MIN_FO_OBS observed GRACE-FO months is reported with the single-variance fit
and flagged (r_fo_fallback).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .kalman import fit_kalman_ar1

# GRACE ends in June 2017 and GRACE-FO begins in June 2018. No month in between is observed,
# so any split date inside the gap gives the same partition.
MISSION_SPLIT = pd.Timestamp("2018-01-01")

# With under a year of GRACE-FO months, r_fo is not separately identified from q
MIN_FO_OBS = 12


def mission_flag(index: pd.DatetimeIndex) -> np.ndarray:
    """Per-month boolean: True where the observation comes from GRACE-FO."""
    return np.asarray(index >= MISSION_SPLIT, dtype=bool)


def _mission_loglik(params: np.ndarray, y: np.ndarray, is_fo: np.ndarray) -> float:
    """Negative log-likelihood; params is (arctanh rho, log q, log r_grace, log r_fo)."""
    rho = np.tanh(params[0])
    q = np.exp(params[1])
    r_grace = np.exp(params[2])
    r_fo = np.exp(params[3])
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    ll = 0.0
    for i, obs in enumerate(y):
        x, p = rho * x, rho * rho * p + q
        if np.isnan(obs):
            continue
        # The only difference from the single-variance filter is which r enters s
        r = r_fo if is_fo[i] else r_grace
        s = p + r
        ll -= 0.5 * (np.log(2 * np.pi * s) + (obs - x) ** 2 / s)
        k = p / s
        x, p = x + k * (obs - x), (1 - k) * p
    return -ll


def fit_kalman_mission(y_train: np.ndarray, fo_train: np.ndarray, full: bool = False):
    """Maximum-likelihood estimate of (rho, q, r_grace, r_fo), from fit_kalman_ar1's starts.

    Both variances start from the same value. If fo_train is constant, the likelihood
    does not depend on one of them and the other parameters land on the single-variance
    optimum. With full=True a diagnostics dict is also returned, as in fit_kalman_ar1,
    with a boundary flag (< 1e-6) for each variance.
    """
    v = float(np.nanvar(y_train))
    best, best_val, best_res = None, np.inf, None
    for rho0 in (0.5, 0.9):
        for split in (0.5, 0.2):
            x0 = np.array([np.arctanh(rho0), np.log(v * split + 1e-9),
                           np.log(v * (1 - split) + 1e-9), np.log(v * (1 - split) + 1e-9)])
            res = minimize(_mission_loglik, x0, args=(y_train, fo_train), method="L-BFGS-B")
            if res.fun < best_val:
                best, best_val, best_res = res.x, res.fun, res
    rho, q = float(np.tanh(best[0])), float(np.exp(best[1]))
    r_grace, r_fo = float(np.exp(best[2])), float(np.exp(best[3]))
    if not full:
        return rho, q, r_grace, r_fo
    diag = {
        "converged": bool(best_res.success),
        "n_iter": int(best_res.nit),
        "neg_loglik": float(best_val),
        "r_grace_at_boundary": bool(r_grace < 1e-6),
        "r_fo_at_boundary": bool(r_fo < 1e-6),
    }
    return rho, q, r_grace, r_fo, diag


def mission_filter_series(
    y: np.ndarray, rho: float, q: float, r_grace: float, r_fo: float, is_fo: np.ndarray
) -> np.ndarray:
    """Filtered state x_{t|t} at every index t, using observations up to and including t."""
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    out = np.empty(len(y))
    for i, obs in enumerate(y):
        x, p = rho * x, rho * rho * p + q
        if not np.isnan(obs):
            s = p + (r_fo if is_fo[i] else r_grace)
            k = p / s
            x, p = x + k * (obs - x), (1 - k) * p
        out[i] = x
    return out


def fit_basin_pair(
    y_train: np.ndarray, fo_train: np.ndarray, min_fo_obs: int = MIN_FO_OBS
) -> dict:
    """Fit the single- and two-variance models on one training window; choose the reported
    fit.

    With fewer than min_fo_obs observed GRACE-FO months the reported parameters are the
    single-variance fit, with r_grace = r_fo = r. The unguarded two-variance estimates
    (*_raw) are returned as well, so the effect of the fallback can be inspected. lr_stat
    and lr_p are the likelihood-ratio test of the reported fit against the single-variance
    fit (chi-squared, one degree of freedom); lr_stat_raw and lr_p_raw use the raw fit.
    """
    n_fo = int(np.isfinite(y_train[fo_train]).sum())
    rho1, q1, r1, diag_one = fit_kalman_ar1(y_train, full=True)
    rho2, q2, rg2, rf2, diag_two = fit_kalman_mission(y_train, fo_train, full=True)

    fallback = n_fo < min_fo_obs
    if fallback:
        rho, q, r_grace, r_fo = rho1, q1, r1, r1
        neg_ll_used = diag_one["neg_loglik"]
    else:
        rho, q, r_grace, r_fo = rho2, q2, rg2, rf2
        neg_ll_used = diag_two["neg_loglik"]

    # The models are nested and differ by one free parameter. A negative statistic means
    # the four-parameter optimizer found a worse optimum than the three-parameter one;
    # the p value treats it as zero.
    lr_stat = 2.0 * (diag_one["neg_loglik"] - neg_ll_used)
    lr_stat_raw = 2.0 * (diag_one["neg_loglik"] - diag_two["neg_loglik"])
    return {
        "n_train_obs": int(np.isfinite(y_train).sum()),
        "n_fo_train": n_fo,
        "rho": rho, "q": q, "r_grace": r_grace, "r_fo": r_fo,
        "r_fo_fallback": bool(fallback),
        "rho_one": rho1, "q_one": q1, "r_one": r1,
        "r_one_at_boundary": diag_one["r_at_boundary"],
        "one_converged": diag_one["converged"],
        "loglik_one": -diag_one["neg_loglik"],
        "rho_two_raw": rho2, "q_two_raw": q2,
        "r_grace_raw": rg2, "r_fo_raw": rf2,
        "r_grace_at_boundary": diag_two["r_grace_at_boundary"],
        "r_fo_at_boundary": diag_two["r_fo_at_boundary"],
        "two_converged": diag_two["converged"],
        "loglik_two_raw": -diag_two["neg_loglik"],
        "loglik_two": -neg_ll_used,
        "lr_stat": lr_stat,
        "lr_p": float(chi2.sf(max(lr_stat, 0.0), df=1)),
        "lr_stat_raw": lr_stat_raw,
        "lr_p_raw": float(chi2.sf(max(lr_stat_raw, 0.0), df=1)),
    }


def mission_predictions(
    resid_wide: pd.DataFrame, test_start: pd.Timestamp, horizons: range,
    min_obs: int = 60, min_fo_obs: int = MIN_FO_OBS, progress: str | None = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], pd.DataFrame]:
    """Two-variance and single-variance forecasts on identical rows, and the parameter table.

    Returns (two-variance forecasts, single-variance forecasts, per-basin parameters), each
    forecast set as {lead time h: frame}. The single-variance forecasts are recomputed here,
    so both share the fold, deseasonalization and basin list, and they can be checked
    against kalman_predictions.csv. With progress set, a line is printed every 25 basins.
    """
    is_fo = mission_flag(resid_wide.index)
    out_two = {h: [] for h in horizons}
    out_one = {h: [] for h in horizons}
    params = []
    for j, name in enumerate(resid_wide.columns):
        s = resid_wide[name]
        train_mask = np.asarray(s.index < test_start)
        y_train = s.values[train_mask]
        if np.isfinite(y_train).sum() < min_obs:
            continue
        fit = fit_basin_pair(y_train, is_fo[train_mask], min_fo_obs=min_fo_obs)
        params.append({"name": name, **fit})
        filt_two = mission_filter_series(
            s.values, fit["rho"], fit["q"], fit["r_grace"], fit["r_fo"], is_fo)
        filt_one = mission_filter_series(
            s.values, fit["rho_one"], fit["q_one"], fit["r_one"], fit["r_one"], is_fo)
        for h in horizons:
            base = pd.DataFrame({
                "name": name,
                "issue_date": s.index,
                "target_date": s.index + pd.DateOffset(months=h),
            })
            out_two[h].append(base.assign(pred=(fit["rho"] ** h) * filt_two))
            out_one[h].append(base.assign(pred=(fit["rho_one"] ** h) * filt_one))
        if progress and (j + 1) % 25 == 0:
            print(f"  {progress}: {j + 1}/{len(resid_wide.columns)} basins", flush=True)
    return (
        {h: pd.concat(v, ignore_index=True) for h, v in out_two.items() if v},
        {h: pd.concat(v, ignore_index=True) for h, v in out_one.items() if v},
        pd.DataFrame(params),
    )
