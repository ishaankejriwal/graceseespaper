"""Neighbor-fused Kalman filter: the neighbor series enters the filter as a second observation.

The Phase 4 result says the neighbor acts as a second noisy sensor of the same regional signal.
This model takes that mechanism literally: basin i keeps its AR(1) latent state x_i, but each
month the filter updates on BOTH y_i = x_i + e_i and y_j = c * x_i + e_j, with the coupling c
and neighbor noise r_j fitted on training months only. If the mechanism is right, in-filter
fusion should beat the post-hoc ridge correction used in Phase 3b; if the neighbor is
uninformative the MLE drives c toward 0 and the model degrades gracefully to the own filter.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .kalman import kalman_forecast_series


def fused_filter(y_i: np.ndarray, y_j: np.ndarray, rho: float, q: float,
                 r_i: float, c: float, r_j: float) -> np.ndarray:
    """Filtered state using both sensors; either observation being NaN skips only its update."""
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    out = np.empty(len(y_i))
    for t in range(len(y_i)):
        x, p = rho * x, rho * rho * p + q
        if not np.isnan(y_i[t]):
            s = p + r_i
            k = p / s
            x, p = x + k * (y_i[t] - x), (1 - k) * p
        if not np.isnan(y_j[t]):
            s = c * c * p + r_j
            k = c * p / s
            x, p = x + k * (y_j[t] - c * x), (1 - k * c) * p
        out[t] = x
    return out


def _fusion_negloglik(params: np.ndarray, y_i: np.ndarray, y_j: np.ndarray,
                      rho: float, q: float, r_i: float) -> float:
    """Joint likelihood of both observation streams with (rho, q, r_i) held at cached values."""
    c = params[0]
    r_j = np.exp(params[1])
    x, p = 0.0, q / max(1 - rho**2, 1e-6)
    ll = 0.0
    for t in range(len(y_i)):
        x, p = rho * x, rho * rho * p + q
        if not np.isnan(y_i[t]):
            s = p + r_i
            ll -= 0.5 * (np.log(2 * np.pi * s) + (y_i[t] - x) ** 2 / s)
            k = p / s
            x, p = x + k * (y_i[t] - x), (1 - k) * p
        if not np.isnan(y_j[t]):
            s = c * c * p + r_j
            ll -= 0.5 * (np.log(2 * np.pi * s) + (y_j[t] - c * x) ** 2 / s)
            k = c * p / s
            x, p = x + k * (y_j[t] - c * x), (1 - k * c) * p
    return -ll


def fit_fusion_obs(y_i_train: np.ndarray, y_j_train: np.ndarray, rho: float, q: float,
                   r_i: float, mle_polish: bool = False) -> tuple[float, float]:
    """Fit (c, r_j) on training months only.

    Closed form: regress the neighbor on the own-filter state — cheap enough to repeat for
    every placebo draw, which keeps real and placebo arms capacity-matched. The optional MLE
    polish refines the real arm from that start; placebos skip it so the comparison stays paired
    on the closed-form fit reported as primary.
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
        res = minimize(_fusion_negloglik, np.array([c, np.log(r_j)]),
                       args=(y_i_train, y_j_train, rho, q, r_i),
                       method="L-BFGS-B", options={"maxiter": 100})
        if np.isfinite(res.fun):
            c, r_j = float(res.x[0]), float(np.exp(res.x[1]))
    return c, r_j


def fusion_state_matrix(resid_wide: pd.DataFrame, params: pd.DataFrame, graph: dict,
                        test_start: pd.Timestamp, mle_polish: bool = False) -> pd.DataFrame:
    """Fused filtered state per basin using its top-1 neighbor; (c, r_j) fitted pre-test only."""
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
