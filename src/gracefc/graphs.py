"""Neighbor graphs between basins, and degree-matched random graphs used as placebos.

A graph maps each target basin to a ranked list of source basins. Sources are selected by
training-window correlation, by lagged correlation, by centroid distance, or by correlation
among sources at least a minimum distance away (graph kinds corr_min<km>).

The placebo for a graph gives each basin the same number of sources, chosen at random, so a
real graph and its placebos differ only in which basins are connected. For a top-1 graph
every basin has one source, and the placebo source is a uniform random other basin. It
therefore differs from the real neighbor in both proximity and correlation, so a gain over
the placebo can come from mascon leakage or shared forcing as well as from a hydrological
connection. The distance-band graphs exclude nearby sources to control for leakage.
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


def distance_matrix_km(meta: pd.DataFrame) -> pd.DataFrame:
    """Centroid great-circle (haversine) distances in km."""
    lat = np.deg2rad(meta["centroid_lat"].values)
    lon = np.deg2rad(meta["centroid_lon"].values)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return pd.DataFrame(d, index=meta["name"].values, columns=meta["name"].values)


def geographic_topk(meta: pd.DataFrame, k: int) -> dict[str, list[str]]:
    """k nearest basins by great-circle centroid distance."""
    dist = distance_matrix_km(meta).to_numpy()
    names = meta["name"].values
    graph = {}
    for i, name in enumerate(names):
        order = np.argsort(dist[i])
        graph[name] = [names[j] for j in order if j != i][:k]
    return graph


def corr_topk_min_distance(
    wide_train: pd.DataFrame, meta: pd.DataFrame, k: int, min_km: float
) -> dict[str, list[str]]:
    """Correlation top-k restricted to sources at least min_km away (the leakage control)."""
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
    """Placebo graph: random sources, as many per target as in graph, drawn without replacement.

    For a top-1 graph this is a uniform random other basin for each target.
    """
    rng = np.random.default_rng(seed)
    names = list(graph.keys())
    out = {}
    for name, nbrs in graph.items():
        pool = [n for n in names if n != name]
        out[name] = list(rng.choice(pool, size=len(nbrs), replace=False)) if nbrs else []
    return out


GRAPH_BUILDERS = {
    "corr": lambda wide_train, meta, k: corr_topk(wide_train, k),
    "pred_lag1": lambda wide_train, meta, k: predictive_lag_topk(wide_train, k),
    "geo": lambda wide_train, meta, k: geographic_topk(meta, k),
}


def resolve_builder(kind: str):
    """Return the builder for a graph kind.

    Besides the kinds in GRAPH_BUILDERS, accepts distance-band kinds such as corr_min300:
    correlation top-k among sources at least 300 km away.
    """
    if kind in GRAPH_BUILDERS:
        return GRAPH_BUILDERS[kind]
    if kind.startswith("corr_min"):
        min_km = float(kind.removeprefix("corr_min"))
        return lambda wide_train, meta, k: corr_topk_min_distance(wide_train, meta, k, min_km)
    raise KeyError(f"unknown graph kind: {kind}")
