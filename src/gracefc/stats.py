"""Forecast-comparison inference: Diebold-Mariano, block bootstrap, and FDR across basins."""
import numpy as np
import pandas as pd
from scipy import stats as sps


def _hac_variance(d: np.ndarray, max_lag: int) -> float:
    # Newey-West with Bartlett weights; forecast-error loss differentials are autocorrelated
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
    """DM test with Harvey small-sample correction. Negative stat means A beats B."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan
    var = _hac_variance(d, max_lag=max(horizon - 1, 1))
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    # Harvey-Leybourne-Newbold correction against a t distribution
    k = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = k * dm
    p = 2 * sps.t.sf(np.abs(stat), df=n - 1)
    return float(stat), float(p)


def _paired_losses(
    sub: pd.DataFrame, model_a: str, model_b: str,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> pd.DataFrame:
    """Row-paired squared losses on the (name, target_date) join of two models.

    Aggregating each model separately and intersecting only target dates lets
    unbalanced basin coverage bias the differential (audit 2026-08-13, P1-5);
    pairing on rows and asserting target equality closes that hole.
    """
    tcol, pcol = value_cols
    keys = ["name", "target_date"]
    a = sub[sub["model"] == model_a][keys + [tcol, pcol]]
    b = sub[sub["model"] == model_b][keys + [tcol, pcol]]
    j = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(j) == 0:
        return j.assign(loss_a=[], loss_b=[])
    if not np.allclose(j[f"{tcol}_a"], j[f"{tcol}_b"], atol=1e-8):
        raise AssertionError(f"targets differ on paired rows: {model_a} vs {model_b}")
    j["loss_a"] = (j[f"{tcol}_a"] - j[f"{pcol}_a"]) ** 2
    j["loss_b"] = (j[f"{tcol}_b"] - j[f"{pcol}_b"]) ** 2
    return j


def pooled_monthly_dm(
    pred_rows: pd.DataFrame, model_a: str, model_b: str, horizon: int,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> tuple[float, float]:
    """DM on the cross-basin mean monthly squared-loss differential — the pooled headline test."""
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    la = j.groupby("target_date")["loss_a"].mean().sort_index()
    lb = j.groupby("target_date")["loss_b"].mean().sort_index()
    return diebold_mariano(la.values, lb.values, horizon=horizon)


def block_bootstrap_skill_ci(
    pred_rows: pd.DataFrame, model_a: str, model_b: str, horizon: int,
    n_boot: int = 2000, block: int = 3, seed: int = 0,
    value_cols: tuple[str, str] = ("target", "pred"),
) -> tuple[float, float, float]:
    """Moving-block bootstrap CI on skill = 1 - MSE_a/MSE_b over monthly pooled losses."""
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    la = j.groupby("target_date")["loss_a"].mean().sort_index().values
    lb = j.groupby("target_date")["loss_b"].mean().sort_index().values
    n = len(la)
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
    """DM per basin then Benjamini-Hochberg: how many basins individually favor A over B."""
    sub = pred_rows[pred_rows["horizon"] == horizon]
    j = _paired_losses(sub, model_a, model_b, value_cols)
    rows = []
    for name, grp in j.sort_values("target_date").groupby("name"):
        stat, p = diebold_mariano(grp["loss_a"].values, grp["loss_b"].values, horizon=horizon)
        rows.append({"name": name, "dm_stat": stat, "p": p, "a_better": stat < 0 if np.isfinite(stat) else False})
    df = pd.DataFrame(rows).dropna(subset=["p"]).sort_values("p").reset_index(drop=True)
    m = len(df)
    df["bh_threshold"] = (np.arange(1, m + 1) / m) * q
    passed = df[df["p"] <= df["bh_threshold"]]
    cutoff = passed.index.max() if len(passed) else -1
    df["significant"] = df.index <= cutoff
    return df
