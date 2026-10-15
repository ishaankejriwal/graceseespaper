"""Significance tests and confidence intervals for paired forecast comparisons.

- diebold_mariano: Diebold-Mariano test with the Harvey et al. (1997) small-sample
  correction and a Newey-West variance at maximum lag max(h - 1, 1).
- pooled_monthly_dm: the same test on the monthly cross-basin mean loss differential.
- block_bootstrap_skill_ci: moving-block bootstrap interval for the MSE skill score,
  block length max(3, h), 2000 draws.
- per_basin_dm_fdr: per-basin Diebold-Mariano tests with Benjamini-Hochberg
  false-discovery-rate control.

Every comparison pairs the two models through _paired_losses, which raises if the models
have different (basin, target month) rows or different targets.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps


def _hac_variance(d: np.ndarray, max_lag: int) -> float:
    """Newey-West variance of the mean of d, with Bartlett weights up to max_lag."""
    n = len(d)
    dc = d - d.mean()
    v = float(np.dot(dc, dc)) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        w = 1 - lag / (max_lag + 1)
        v += 2 * w * float(np.dot(dc[:-lag], dc[lag:])) / n
    return v / n


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano statistic and two-sided p value for loss_a against loss_b.

    The statistic carries the Harvey-Leybourne-Newbold small-sample correction and is
    referred to a t distribution with n - 1 degrees of freedom. A negative statistic
    means A has the lower mean loss. Returns (nan, nan) for fewer than 10 values,
    identical losses or a non-positive variance.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    var = _hac_variance(d, max_lag=max(horizon - 1, 1))
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    # Harvey-Leybourne-Newbold small-sample factor
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = k * dm
    p = 2 * sps.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


def _paired_losses(
    sub: pd.DataFrame, model_a: str, model_b: str,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> pd.DataFrame:
    """Squared losses of two models, paired row by row on (name, target_date).

    Pairing rows, rather than aggregating each model and intersecting target dates,
    keeps unbalanced basin coverage out of the loss differential. Raises AssertionError
    if both models have rows but the row sets differ, or if the targets differ on a
    paired row. Returns an empty frame with the paired columns if either model has no
    rows in sub.
    """
    tcol, pcol = value_cols
    keys = ["name", "target_date"]
    a = sub[sub["model"] == model_a][keys + [tcol, pcol]]
    b = sub[sub["model"] == model_b][keys + [tcol, pcol]]
    if len(a) == 0 or len(b) == 0:
        # A model missing from this slice (for example, not run at this lead time) is
        # not a pairing error. The empty frame keeps the paired columns so that the
        # callers' groupby and dropna calls give NaN or empty results.
        cols = keys + [f"{tcol}_a", f"{pcol}_a", f"{tcol}_b", f"{pcol}_b",
                       "loss_a", "loss_b"]
        return pd.DataFrame(columns=cols)
    j = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(j) != len(a) or len(j) != len(b):
        # An inner join would score each model on a subset of its rows, so callers
        # must match rows before pairing.
        raise AssertionError(
            f"row sets differ on paired keys: {model_a} has {len(a)} rows, "
            f"{model_b} has {len(b)}, join keeps {len(j)}"
        )
    if not np.allclose(j[f"{tcol}_a"], j[f"{tcol}_b"], atol=1e-8):
        raise AssertionError(f"targets differ on paired rows: {model_a} vs {model_b}")
    j["loss_a"] = (j[f"{tcol}_a"] - j[f"{pcol}_a"]) ** 2
    j["loss_b"] = (j[f"{tcol}_b"] - j[f"{pcol}_b"]) ** 2
    return j


def pooled_monthly_dm(
    pred_rows: pd.DataFrame, model_a: str, model_b: str, horizon: int,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> tuple[float, float]:
    """Pooled Diebold-Mariano test on monthly cross-basin mean squared losses at one lead time."""
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    la = j.groupby("target_date")["loss_a"].mean().sort_index()
    lb = j.groupby("target_date")["loss_b"].mean().sort_index()
    return diebold_mariano(la.values, lb.values, horizon=horizon)


def block_bootstrap_skill_ci(
    pred_rows: pd.DataFrame, model_a: str, model_b: str, horizon: int,
    n_boot: int = 2000, block: int | None = None, seed: int = 0,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> tuple[float, float, float]:
    """Skill score 1 - MSE_a / MSE_b and its 95 % moving-block bootstrap interval.

    Returns (point, lower, upper) as fractions. The bootstrap resamples the monthly
    cross-basin mean losses. The block length defaults to max(3, horizon), because
    overlapping h-step forecasts make the losses dependent out to lag h - 1. Returns
    NaNs when there are fewer months than the block length.
    """
    if block is None:
        block = max(3, horizon)
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    la = j.groupby("target_date")["loss_a"].mean().sort_index().values
    lb = j.groupby("target_date")["loss_b"].mean().sort_index().values
    n = len(la)
    if n < block:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    point = 1 - la.mean() / lb.mean()
    draws = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        draws[i] = 1 - la[idx].mean() / lb[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def per_basin_dm_fdr(
    pred_rows: pd.DataFrame, model_a: str, model_b: str, horizon: int,
    q: float = 0.10, value_cols: tuple[str, str] = ("target", "pred"),
) -> pd.DataFrame:
    """Per-basin Diebold-Mariano tests with Benjamini-Hochberg control at level q.

    Here q is the false-discovery-rate level, unrelated to the Kalman q. Returns one row
    per basin with a finite p value, sorted by p: dm_stat, p, a_better (negative
    statistic), bh_threshold and significant.
    """
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    rows = []
    for name, grp in j.sort_values("target_date").groupby("name"):
        stat, p = diebold_mariano(grp["loss_a"].values, grp["loss_b"].values, horizon=horizon)
        rows.append({"name": name, "dm_stat": stat, "p": p, "a_better": stat < 0 if np.isfinite(stat) else False})
    if not rows:
        return pd.DataFrame(columns=["name", "dm_stat", "p", "a_better",
                                     "bh_threshold", "significant"])
    df = pd.DataFrame(rows).dropna(subset=["p"]).sort_values("p").reset_index(drop=True)
    m = len(df)
    df["bh_threshold"] = (np.arange(1, m + 1) / m) * q
    passed = df[df["p"] <= df["bh_threshold"]]
    cutoff = passed.index.max() if len(passed) else -1
    df["significant"] = df.index <= cutoff
    return df
