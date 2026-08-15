"""Decides which basins count as a basin's "neighbours".

Three ways to pick them: most-correlated (on the training window only), nearest by
distance, and — most importantly — RANDOM.

The random ones are the placebo. To claim "knowing what neighbouring basins do helps",
it is not enough to show the model improved when neighbours were added; maybe any extra
input would have helped. So we rerun everything with randomly chosen basins wired up in a
graph of exactly the same shape (same number of connections per basin) using the same
model seed. The only thing that differs is WHICH basins are connected. If the real graph
does not beat the random ones, the effect was never about geography.

"Degree-matched" below just means the fake graph gives each basin the same number of
neighbours as the real one, so the comparison is not confounded by graph size.
"""
import numpy as np
import pandas as pd


def corr_topk(wide_train: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """Top-k positively correlated source basins per target, computed on the training window only."""
    corr = wide_train.corr()
    graph = {}
    for name in corr.columns:
        s = corr[name].drop(name).dropna()
        s = s[s > 0].sort_values(ascending=False)
        graph[name] = list(s.index[:k])
    return graph


def predictive_lag_topk(wide_train: pd.DataFrame, k: int, lag: int = 1) -> dict[str, list[str]]:
    """Top-k basins whose lagged TWSA best correlates with the target's current TWSA."""
    lagged = wide_train.shift(lag)
    graph = {}
    for name in wide_train.columns:
        cors = lagged.corrwith(wide_train[name]).drop(name).dropna()
        cors = cors[cors > 0].sort_values(ascending=False)
        graph[name] = list(cors.index[:k])
    return graph


def geographic_topk(meta: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """k nearest basins by great-circle centroid distance."""
    lat = np.deg2rad(meta["centroid_lat"].values)
    lon = np.deg2rad(meta["centroid_lon"].values)
    # Haversine distance matrix in km
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    names = meta["name"].values
    graph = {}
    for i, name in enumerate(names):
        order = np.argsort(dist[i])
        graph[name] = [names[j] for j in order if j != i][:k]
    return graph


def distance_matrix_km(meta: pd.DataFrame) -> pd.DataFrame:
    """Centroid great-circle distances for distance-stratified neighbor selection."""
    lat = np.deg2rad(meta["centroid_lat"].values)
    lon = np.deg2rad(meta["centroid_lon"].values)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return pd.DataFrame(d, index=meta["name"].values, columns=meta["name"].values)


def corr_topk_min_distance(
    wide_train: pd.DataFrame, meta: pd.DataFrame, k: int, min_km: float
) -> dict[str, list[str]]:
    """Correlation top-k restricted to sources at least min_km away — the leakage control."""
    dist = distance_matrix_km(meta)
    corr = wide_train.corr()
    graph = {}
    for name in corr.columns:
        s = corr[name].drop(name).dropna()
        far = dist.loc[name].drop(name)
        s = s[(s > 0) & (far.reindex(s.index) >= min_km)].sort_values(ascending=False)
        graph[name] = list(s.index[:k])
    return graph


def random_degree_matched(
    graph: dict[str, list[str]], seed: int
) -> dict[str, list[str]]:
    """Random neighbors with the same in-degree per target — the chance placebo."""
    rng = np.random.default_rng(seed)
    names = list(graph.keys())
    out = {}
    for name, nbrs in graph.items():
        pool = [n for n in names if n != name]
        out[name] = list(rng.choice(pool, size=len(nbrs), replace=False)) if nbrs else []
    return out
