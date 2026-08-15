"""Coupled bivariate Kalman filter: two basins, own latent states, correlated process noise.

The single-latent fusion failed because it forced the neighbor to observe the basin's own
state plus white noise; in truth the neighbor carries persistent local signal. Here each basin
keeps its own AR(1) latent, and coupling enters only through the process-noise correlation c:
regional weather shocks hit both states together, everything local stays separate. At c = 0
this reduces exactly to the independent own filter, so an uninformative neighbor costs nothing.
All per-basin (rho, q, r) come from the cached own fits; c is the single new parameter, fitted
on training months only by joint MLE.

The filter is hand-coded 2x2 scalar arithmetic: it runs inside a per-evaluation MLE loop for
thousands of basin pairs, where numpy matrix overhead dominates the actual math.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

LOG2PI = float(np.log(2 * np.pi))


def _bivar_filter(y_i, y_j, rho_i, rho_j, q_i, q_j, r_i, r_j, c, return_ll=False):
    """2-state filter; either observation being NaN drops only its row of the update."""
    cq = c * (q_i * q_j) ** 0.5
    # Stationary prior for the diagonal-AR pair with cross process noise
    x0 = x1 = 0.0
    p00 = q_i / max(1 - rho_i * rho_i, 1e-6)
    p11 = q_j / max(1 - rho_j * rho_j, 1e-6)
    p01 = cq / max(1 - rho_i * rho_j, 1e-6)
    n = len(y_i)
    out = np.empty(n)
    ll = 0.0
    for t in range(n):
        # Predict
        x0, x1 = rho_i * x0, rho_j * x1
        p00 = rho_i * rho_i * p00 + q_i
        p11 = rho_j * rho_j * p11 + q_j
        p01 = rho_i * rho_j * p01 + cq
        yi, yj = y_i[t], y_j[t]
        oi, oj = yi == yi, yj == yj  # NaN-safe availability checks
        if oi and oj:
            # Joint 2x2 update via closed-form inverse
            s00, s11, s01 = p00 + r_i, p11 + r_j, p01
            det = s00 * s11 - s01 * s01
            v0, v1 = yi - x0, yj - x1
            i00, i11, i01 = s11 / det, s00 / det, -s01 / det
            if return_ll:
                quad = v0 * (i00 * v0 + i01 * v1) + v1 * (i01 * v0 + i11 * v1)
                ll -= 0.5 * (2 * LOG2PI + np.log(det) + quad)
            k00 = p00 * i00 + p01 * i01
            k01 = p00 * i01 + p01 * i11
            k10 = p01 * i00 + p11 * i01
            k11 = p01 * i01 + p11 * i11
            x0 += k00 * v0 + k01 * v1
            x1 += k10 * v0 + k11 * v1
            n00 = (1 - k00) * p00 - k01 * p01
            n01 = (1 - k00) * p01 - k01 * p11
            n11 = -k10 * p01 + (1 - k11) * p11
            p00, p01, p11 = n00, n01, n11
        elif oi:
            s = p00 + r_i
            v = yi - x0
            if return_ll:
                ll -= 0.5 * (LOG2PI + np.log(s) + v * v / s)
            k0, k1 = p00 / s, p01 / s
            x0 += k0 * v
            x1 += k1 * v
            p11 -= k1 * p01
            p01 -= k0 * p01
            p00 -= k0 * p00
        elif oj:
            s = p11 + r_j
            v = yj - x1
            if return_ll:
                ll -= 0.5 * (LOG2PI + np.log(s) + v * v / s)
            k0, k1 = p01 / s, p11 / s
            x0 += k0 * v
            x1 += k1 * v
            p00 -= k0 * p01
            p01 -= k1 * p01
            p11 -= k1 * p11
        out[t] = x0
    return (out, ll) if return_ll else out


def fit_coupling(y_i_train, y_j_train, rho_i, rho_j, q_i, q_j, r_i, r_j) -> float:
    """MLE of the process-noise correlation c on training months; bounded away from +/-1."""
    def neg_ll(c):
        _, ll = _bivar_filter(y_i_train, y_j_train, rho_i, rho_j, q_i, q_j, r_i, r_j,
                              c, return_ll=True)
        return -ll
    res = minimize_scalar(neg_ll, bounds=(-0.95, 0.95), method="bounded",
                          options={"xatol": 0.02})
    return float(res.x) if np.isfinite(res.fun) else 0.0


def coupled_state_matrix(resid_wide: pd.DataFrame, params: pd.DataFrame, graph: dict,
                         test_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    """Coupled filtered state x_i(t|t) per basin with its top-1 neighbor; returns fitted c too."""
    p = params.set_index("name")
    train_mask = np.asarray(resid_wide.index < test_start)
    out, cs = {}, {}
    for name in resid_wide.columns:
        if name not in p.index:
            continue
        rho_i, q_i, r_i = p.loc[name, ["rho", "q", "r"]]
        nbrs = graph.get(name, [])
        nbr = nbrs[0] if nbrs else None
        if nbr is None or nbr not in p.index or nbr not in resid_wide.columns:
            # No usable neighbor: c = 0 coupled filter IS the own filter, so run it that way
            y_i = resid_wide[name].values
            out[name] = _bivar_filter(y_i, np.full_like(y_i, np.nan),
                                      rho_i, rho_i, q_i, q_i, r_i, r_i, 0.0)
            cs[name] = 0.0
            continue
        rho_j, q_j, r_j = p.loc[nbr, ["rho", "q", "r"]]
        y_i, y_j = resid_wide[name].values, resid_wide[nbr].values
        c = fit_coupling(y_i[train_mask], y_j[train_mask],
                         rho_i, rho_j, q_i, q_j, r_i, r_j)
        out[name] = _bivar_filter(y_i, y_j, rho_i, rho_j, q_i, q_j, r_i, r_j, c)
        cs[name] = c
    return pd.DataFrame(out, index=resid_wide.index), pd.Series(cs, name="coupling")
