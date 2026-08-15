"""Per-basin AR(1) state + observation-noise Kalman baseline: filtering vs dynamics attribution."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _filter_loglik(params: np.ndarray, y: np.ndarray, fixed_r: float | None = None) -> float:
    rho = np.tanh(params[0])
    q = np.exp(params[1])
    r = fixed_r if fixed_r is not None else np.exp(params[2])
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    ll = 0.0
    for obs in y:
        # Predict
        x, p = rho * x, rho * rho * p + q
        if np.isnan(obs):
            continue
        # Update; NaN months simply skip the update, which handles gaps exactly
        s = p + r
        ll -= 0.5 * (np.log(2 * np.pi * s) + (obs - x) ** 2 / s)
        k = p / s
        x, p = x + k * (obs - x), (1 - k) * p
    return -ll


def fit_kalman_ar1(y_train: np.ndarray, full: bool = False, fixed_r: float | None = None):
    """MLE of (rho, q, r) with multiple starts; observed variance seeds the noise scales.

    With full=True also returns optimizer diagnostics: r is not separately
    identified when the likelihood is flat in the q/r split, and boundary
    collapse (r -> 0) must be visible downstream (audit 2026-08-13, P0-5).
    With fixed_r=0.0 the update gain is 1 wherever an observation exists, so the
    model degenerates to damped persistence with MLE rho — the ablation that
    isolates the observation-noise term (audit 2026-08-15, blocker 2).
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
    """Filtered state at every index t using observations up to and including t."""
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
    """Per-basin (rho, q, r) fit on pre-test data only; cacheable across treatments."""
    rows = []
    for name in resid_wide.columns:
        y_train = resid_wide[name][resid_wide.index < test_start].values
        if np.isfinite(y_train).sum() < min_obs:
            continue
        rho, q, r, diag = fit_kalman_ar1(y_train, full=True)
        rows.append({"name": name, "rho": rho, "q": q, "r": r, **diag})
    return pd.DataFrame(rows)


def filtered_state_wide(resid_wide: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    """Filtered state x_{t|t} for every basin and month; causal by construction."""
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
    """Fit per basin on pre-test data; forecast rho^h times the filtered state at each issue."""
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
