"""Mission-split observation noise: one r for GRACE, another for GRACE-FO.

The published filter estimates a single observation-noise variance r per basin-fold from a
training record that spans two different satellite missions. GRACE flew Apr 2002 - Jun 2017,
GRACE-FO from Jun 2018 onward, and the two have different accelerometer, instrument and
post-gap processing characteristics. Every test month in this study is GRACE-FO. A referee
can reasonably ask whether a single r, dominated by the longer GRACE record, mis-sizes the
Kalman gain on the months that actually get scored.

This module answers that with the same model, estimator, folds and forecast rule, changing
one thing: the observation equation gets two variances,

    what we observe  =  real level  +  noise of size r_grace   (months before 2018-01)
    what we observe  =  real level  +  noise of size r_fo      (months from 2018-01 on)

so the parameter vector is (rho, q, r_grace, r_fo) instead of (rho, q, r). Everything else
matches gracefc.kalman: L-BFGS-B from four starts, training months only, forecast at lead h
is rho**h times the filtered state at the issue month.

Identifiability caveat that drives the fallback below: the training window of the first fold
ends 2019-06 and contains only ten finite GRACE-FO months, which is too thin to separate a
second variance from q. Folds whose training window carries fewer than MIN_FO_OBS GRACE-FO
observations are refit with the one-r model and flagged, rather than reported as if the extra
parameter had been estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

from .kalman import fit_kalman_ar1

# GRACE ends 2017-06 and GRACE-FO begins 2018-06; the gap year holds no observations, so any
# split date inside it partitions the record identically. 2018-01 is the calendar convention.
MISSION_SPLIT = pd.Timestamp("2018-01-01")

# A variance estimated from under a year of observations is not separately identified from q
MIN_FO_OBS = 12


def mission_flag(index: pd.DatetimeIndex) -> np.ndarray:
    """Per-month boolean: True where the observation comes from GRACE-FO."""
    return np.asarray(index >= MISSION_SPLIT, dtype=bool)


def _mission_loglik(params: np.ndarray, y: np.ndarray, is_fo: np.ndarray) -> float:
    rho = np.tanh(params[0])
    q = np.exp(params[1])
    r_grace = np.exp(params[2])
    r_fo = np.exp(params[3])
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    ll = 0.0
    for i, obs in enumerate(y):
        # Predict
        x, p = rho * x, rho * rho * p + q
        if np.isnan(obs):
            continue
        # Update; the only departure from the one-r filter is which variance enters s
        r = r_fo if is_fo[i] else r_grace
        s = p + r
        ll -= 0.5 * (np.log(2 * np.pi * s) + (obs - x) ** 2 / s)
        k = p / s
        x, p = x + k * (obs - x), (1 - k) * p
    return -ll


def fit_kalman_mission(y_train: np.ndarray, fo_train: np.ndarray, full: bool = False):
    """MLE of (rho, q, r_grace, r_fo); same four starts and seeding as fit_kalman_ar1.

    Both noise starts take the same seed value, so a mission flag that is constant
    reduces this to the one-r problem embedded in a coordinate the likelihood does
    not depend on, and the identified parameters land on the one-r optimum.
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
    """Filtered state at every index t using observations up to and including t."""
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
    """Fit the one-r and two-r models on the same training window and pick the reported one.

    Returns the raw two-r estimates alongside the guarded ones so the fallback decision
    stays auditable: a reader can see what the unidentified fit would have claimed.
    """
    n_fo = int(np.isfinite(y_train[fo_train]).sum())
    rho1, q1, r1, d1 = fit_kalman_ar1(y_train, full=True)
    rho2, q2, rg2, rf2, d2 = fit_kalman_mission(y_train, fo_train, full=True)

    fallback = n_fo < min_fo_obs
    if fallback:
        # Too few GRACE-FO months to separate a second variance; report the one-r fit
        rho, q, r_grace, r_fo = rho1, q1, r1, r1
        neg_ll_used = d1["neg_loglik"]
    else:
        rho, q, r_grace, r_fo = rho2, q2, rg2, rf2
        neg_ll_used = d2["neg_loglik"]

    # Nested models differing in one free parameter; a negative statistic means the
    # four-parameter optimizer found a worse optimum than the three-parameter one
    lr_stat = 2.0 * (d1["neg_loglik"] - neg_ll_used)
    lr_stat_raw = 2.0 * (d1["neg_loglik"] - d2["neg_loglik"])
    return {
        "n_train_obs": int(np.isfinite(y_train).sum()),
        "n_fo_train": n_fo,
        "rho": rho, "q": q, "r_grace": r_grace, "r_fo": r_fo,
        "r_fo_fallback": bool(fallback),
        "rho_one": rho1, "q_one": q1, "r_one": r1,
        "r_one_at_boundary": d1["r_at_boundary"],
        "one_converged": d1["converged"],
        "loglik_one": -d1["neg_loglik"],
        "rho_two_raw": rho2, "q_two_raw": q2,
        "r_grace_raw": rg2, "r_fo_raw": rf2,
        "r_grace_at_boundary": d2["r_grace_at_boundary"],
        "r_fo_at_boundary": d2["r_fo_at_boundary"],
        "two_converged": d2["converged"],
        "loglik_two_raw": -d2["neg_loglik"],
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
    """Two-r and one-r forecasts on identical rows, plus the per-basin parameter table.

    The one-r forecasts are regenerated here rather than read back from disk so that the
    pair is guaranteed to share fold, deseasonalization and keep-list, and so the run can
    check itself against the published kalman_predictions.csv.
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
