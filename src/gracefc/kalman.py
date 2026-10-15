"""Scalar Kalman filter for the per-basin AR(1) state plus model-noise model (KF).

Each basin's deseasonalized, standardized residual y_t is modeled as

    x_t = rho * x_{t-1} + eps_t,    eps_t ~ N(0, q)
    y_t = x_t + eta_t,              eta_t ~ N(0, r)

with latent state x_t, process-noise variance q and model-noise variance r. The
parameters (rho, q, r) are estimated per basin and fold by maximum likelihood on
training months only (L-BFGS-B from four starting points). At each observed month the
filter updates its state estimate x_{t|t} with the Kalman gain k = p / (p + r), where p
is the predicted state variance. At a month with no observation it skips the update and
propagates x and p forward. The forecast at lead time h issued at month t is
rho**h * x_{t|t} (code id kalman_ar1).

With fixed_r=0.0 the gain is 1 at every observed month, so the forecast reduces to
damped persistence with a maximum-likelihood rho. This is the zero-r variant
(kalman_r0, scripts/run_r0_ablation.py).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _filter_loglik(params: np.ndarray, y: np.ndarray, fixed_r: float | None = None) -> float:
    """Negative Gaussian log-likelihood of y, the objective for fit_kalman_ar1.

    params is (arctanh rho, log q, log r), or (arctanh rho, log q) when fixed_r is
    given, so the optimizer works unconstrained while |rho| < 1 and q, r > 0. The
    filter starts from x = 0 with the stationary variance q / (1 - rho**2).
    """
    rho = np.tanh(params[0])
    q = np.exp(params[1])
    r = fixed_r if fixed_r is not None else np.exp(params[2])
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    ll = 0.0
    for obs in y:
        x, p = rho * x, rho * rho * p + q
        # A NaN month skips the update and adds nothing to the likelihood
        if np.isnan(obs):
            continue
        # s is the innovation variance and k the Kalman gain
        s = p + r
        ll -= 0.5 * (np.log(2 * np.pi * s) + (obs - x) ** 2 / s)
        k = p / s
        x, p = x + k * (obs - x), (1 - k) * p
    return -ll


def fit_kalman_ar1(y_train: np.ndarray, full: bool = False, fixed_r: float | None = None):
    """Maximum-likelihood estimate of (rho, q, r) from one basin's training residuals.

    L-BFGS-B runs from four starts (rho of 0.5 or 0.9; q given 50 % or 20 % of the
    sample variance and r the rest) and the lowest negative log-likelihood is kept.
    NaN months are allowed.

    Returns (rho, q, r), or (rho, q, r, diag) with full=True. diag holds the
    optimizer's convergence flag, iteration count and negative log-likelihood, and
    r_at_boundary, which is True when r < 1e-6 (the likelihood has driven r to
    effectively zero). When fixed_r is given only rho and q are fitted and r is
    returned as fixed_r; fixed_r=0.0 gives the zero-r variant.
    """
    v = float(np.nanvar(y_train))
    best, best_val, best_res = None, np.inf, None
    for rho0 in (0.5, 0.9):
        for split in (0.5, 0.2):
            if fixed_r is None:
                x0 = np.array([np.arctanh(rho0), np.log(v * split + 1e-9),
                               np.log(v * (1 - split) + 1e-9)])
            else:
                x0 = np.array([np.arctanh(rho0), np.log(v * split + 1e-9)])
            res = minimize(_filter_loglik, x0, args=(y_train, fixed_r), method="L-BFGS-B")
            if res.fun < best_val:
                best, best_val, best_res = res.x, res.fun, res
    rho, q = float(np.tanh(best[0])), float(np.exp(best[1]))
    r = fixed_r if fixed_r is not None else float(np.exp(best[2]))
    if not full:
        return rho, q, r
    diag = {
        "converged": bool(best_res.success),
        "n_iter": int(best_res.nit),
        "neg_loglik": float(best_val),
        "r_at_boundary": bool(r < 1e-6),
    }
    return rho, q, r, diag


def kalman_forecast_series(y: np.ndarray, rho: float, q: float, r: float) -> np.ndarray:
    """Filtered state x_{t|t} at every index t, using observations up to and including t.

    The lead-h forecast issued at t is rho**h times this value.
    """
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    out = np.empty(len(y))
    for i, obs in enumerate(y):
        x, p = rho * x, rho * rho * p + q
        if not np.isnan(obs):
            s = p + r
            k = p / s
            x, p = x + k * (obs - x), (1 - k) * p
        out[i] = x
    return out


def fit_fold_params(
    resid_wide: pd.DataFrame, test_start: pd.Timestamp, min_obs: int = 60
) -> pd.DataFrame:
    """Per-basin (rho, q, r) and fit diagnostics from months before test_start.

    Basins with fewer than min_obs observed training months are skipped. The result
    depends only on the data and the fold, so runners cache it (gracefc.cache).
    """
    rows = []
    for name in resid_wide.columns:
        y_train = resid_wide[name][resid_wide.index < test_start].values
        if np.isfinite(y_train).sum() < min_obs:
            continue
        rho, q, r, diag = fit_kalman_ar1(y_train, full=True)
        rows.append({"name": name, "rho": rho, "q": q, "r": r, **diag})
    return pd.DataFrame(rows)


def filtered_state_wide(resid_wide: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Filtered state x_{t|t} for every basin in params and every month of resid_wide.

    Each value uses observations up to and including its own month only.
    """
    out = {}
    p = params.set_index("name")
    for name in resid_wide.columns:
        if name not in p.index:
            continue
        rho, q, r = p.loc[name, ["rho", "q", "r"]]
        out[name] = kalman_forecast_series(resid_wide[name].values, rho, q, r)
    return pd.DataFrame(out, index=resid_wide.index)


def kalman_predictions(
    resid_wide: pd.DataFrame, test_start: pd.Timestamp, horizons: range,
    fixed_r: float | None = None,
) -> dict[int, pd.DataFrame]:
    """KF forecasts rho**h * x_{t|t} at every issue month, as {lead time h: frame}.

    (rho, q, r) are fit per basin on months before test_start; basins with fewer than
    60 observed training months are skipped. fixed_r=0.0 gives the zero-r variant.
    Rows cover the whole record, and callers select the test issue months.
    """
    out = {h: [] for h in horizons}
    for name in resid_wide.columns:
        s = resid_wide[name]
        y_train = s[s.index < test_start].values
        if np.isfinite(y_train).sum() < 60:
            continue
        rho, q, r = fit_kalman_ar1(y_train, fixed_r=fixed_r)
        filt = kalman_forecast_series(s.values, rho, q, r)
        for h in horizons:
            pred = pd.DataFrame({
                "name": name,
                "issue_date": s.index,
                "target_date": s.index + pd.DateOffset(months=h),
                "pred": (rho ** h) * filt,
            })
            out[h].append(pred)
    return {h: pd.concat(v, ignore_index=True) for h, v in out.items() if v}
