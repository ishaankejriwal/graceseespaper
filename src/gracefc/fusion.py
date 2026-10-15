"""Kalman filter that uses a neighbor basin's series as a second observation of the basin's
state.

Basin i keeps its AR(1) latent state x_i. Each month the filter updates on both
y_i = x_i + e_i and y_j = c * x_i + e_j, where y_j is the top-1 neighbor's residual. The
loading c and the neighbor noise variance r_j are fitted on training months only, with
(rho, q, r_i) held at the per-basin filter values. This is an alternative to adding the
neighbor's state to a ridge correction after filtering (experiment_kalman). With c = 0 the
neighbor observation has no effect and the filter reduces to the per-basin filter.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .kalman import kalman_forecast_series


def fused_filter(y_i: np.ndarray, y_j: np.ndarray, rho: float, q: float,
                 r_i: float, c: float, r_j: float, return_ll: bool = False):
    """Filtered state using both observations; a NaN in either skips only that update.

    With return_ll=True also returns the joint log-likelihood of both observation series,
    the objective for fitting (c, r_j) with (rho, q, r_i) held fixed.
    """
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    out = np.empty(len(y_i))
    ll = 0.0
    for t in range(len(y_i)):
        x, p = rho * x, rho * rho * p + q
        if not np.isnan(y_i[t]):
            s = p + r_i
            if return_ll:
                ll -= 0.5 * (np.log(2 * np.pi * s) + (y_i[t] - x) ** 2 / s)
            k = p / s
            x, p = x + k * (y_i[t] - x), (1 - k) * p
        if not np.isnan(y_j[t]):
            s = c * c * p + r_j
            if return_ll:
                ll -= 0.5 * (np.log(2 * np.pi * s) + (y_j[t] - c * x) ** 2 / s)
            k = c * p / s
            x, p = x + k * (y_j[t] - c * x), (1 - k * c) * p
        out[t] = x
    return (out, ll) if return_ll else out


def fit_fusion_obs(y_i_train: np.ndarray, y_j_train: np.ndarray, rho: float, q: float,
                   r_i: float, mle_polish: bool = False) -> tuple[float, float]:
    """Fit (c, r_j) on training months only.

    The closed-form fit regresses the neighbor series on the per-basin filtered state. It
    is cheap enough to repeat for every placebo draw, so real and placebo arms are fitted
    the same way. With mle_polish=True the closed-form values are refined by maximum
    likelihood; this is meant for the real arm only, so the paired placebo comparison uses
    the closed-form fit. Returns (0, 1) when fewer than 24 training months have both values.
    """
    x = kalman_forecast_series(y_i_train, rho, q, r_i)
    m = ~np.isnan(y_j_train) & np.isfinite(x)
    if m.sum() < 24:
        return 0.0, 1.0
    denom = float(np.dot(x[m], x[m]))
    c = float(np.dot(x[m], y_j_train[m]) / denom) if denom > 0 else 0.0
    resid = y_j_train[m] - c * x[m]
    r_j = max(float(np.var(resid)), 1e-8)
    if mle_polish:
        def neg_ll(prm):
            return -fused_filter(y_i_train, y_j_train, rho, q, r_i, prm[0], np.exp(prm[1]),
                                 return_ll=True)[1]
        res = minimize(neg_ll, np.array([c, np.log(r_j)]),
                       method="L-BFGS-B", options={"maxiter": 100})
        if np.isfinite(res.fun):
            c, r_j = float(res.x[0]), float(np.exp(res.x[1]))
    return c, r_j


def fusion_state_matrix(resid_wide: pd.DataFrame, params: pd.DataFrame, graph: dict,
                        test_start: pd.Timestamp, mle_polish: bool = False) -> pd.DataFrame:
    """Fused filtered state of each basin using its top-1 neighbor, with (c, r_j) fitted
    pre-test.

    Basins without a neighbor get the per-basin filtered state.
    """
    p = params.set_index("name")
    out = {}
    for name in resid_wide.columns:
        if name not in p.index:
            continue
        rho, q, r_i = p.loc[name, ["rho", "q", "r"]]
        nbrs = graph.get(name, [])
        y_i = resid_wide[name].values
        if not nbrs or nbrs[0] not in resid_wide.columns:
            out[name] = kalman_forecast_series(y_i, rho, q, r_i)
            continue
        y_j = resid_wide[nbrs[0]].values
        train_mask = resid_wide.index < test_start
        c, r_j = fit_fusion_obs(y_i[train_mask], y_j[train_mask], rho, q, r_i, mle_polish)
        out[name] = fused_filter(y_i, y_j, rho, q, r_i, c, r_j)
    return pd.DataFrame(out, index=resid_wide.index)
