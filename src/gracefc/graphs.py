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
neighbours as the real one, so the comparison is not confounded by graph size. For the
top-1 graphs the stacked system uses, every basin has in-degree 1, so degree matching
constrains nothing: random_degree_matched is a UNIFORM random other basin. That placebo
therefore differs from the real neighbour in identity AND in proximity and correlation
at once, so a mascon-leakage or shared-forcing artefact would beat it just as a real
teleconnection would.

random_distance_matched and random_correlation_matched close that hole. They draw a fake
neighbour that sits at (roughly) the same centroid distance, or carries (roughly) the same
training-window correlation, as the real one — so the only thing left varying is which
basin it is.
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
    """Random neighbors with the same in-degree per target — the chance placebo.

    For a top-1 graph every target has in-degree 1, so "degree-matched" imposes no
    constraint at all and this reduces to a UNIFORM random other basin. The draw
    differs from the real neighbor in identity, proximity and correlation together,
    which is why the distance- and correlation-matched placebos below exist.
    """
    rng = np.random.default_rng(seed)
    names = list(graph.keys())
    out = {}
    for name, nbrs in graph.items():
        pool = [n for n in names if n != name]
        out[name] = list(rng.choice(pool, size=len(nbrs), replace=False)) if nbrs else []
    return out


# Widening cap: beyond half the earth's circumference every basin is inside the window,
# so a distance tolerance past this point cannot admit further candidates
_MAX_TOL_KM = 40_000.0
_MIN_CANDIDATES = 3


def _matched_draw(
    graph: dict[str, list[str]],
    score: np.ndarray,
    names: list[str],
    seed: int,
    tol: float,
    max_tol: float,
    eligible: np.ndarray | None,
    tol_used: dict[str, float] | None,
) -> dict[str, list[str]]:
    """Shared body of the matched placebos: draw a basin whose score[target, j] sits
    within tol of the real neighbor's score, widening tol until enough candidates exist.

    Seeded exactly like random_degree_matched — one default_rng(seed), targets visited in
    graph insertion order, one draw per neighbor rank — so a given seed is reproducible.
    """
    rng = np.random.default_rng(seed)
    pos = {n: i for i, n in enumerate(names)}
    out: dict[str, list[str]] = {}
    for name, nbrs in graph.items():
        out[name] = []
        if not nbrs:
            continue
        i = pos[name]
        row = score[i]
        # Self and every real neighbor of this target are banned, and picks are drawn
        # without replacement so the placebo keeps the real graph's in-degree
        banned = np.zeros(len(row), dtype=bool)
        banned[i] = True
        for nb in nbrs:
            banned[pos[nb]] = True
        if eligible is not None:
            banned |= ~eligible[i]
        banned |= ~np.isfinite(row)
        worst_tol = 0.0
        for nb in nbrs:
            gap = np.abs(row - row[pos[nb]])
            width = tol
            while True:
                cand = np.flatnonzero((gap <= width) & ~banned)
                if len(cand) >= _MIN_CANDIDATES or width >= max_tol:
                    break
                width *= 2
            if not len(cand):
                continue
            worst_tol = max(worst_tol, width)
            pick = int(rng.choice(cand))
            banned[pick] = True
            out[name].append(names[pick])
        if tol_used is not None:
            tol_used[name] = worst_tol
    return out


def random_distance_matched(
    graph: dict[str, list[str]],
    meta: pd.DataFrame,
    seed: int,
    tol_km: float = 150.0,
    tol_used: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """Placebo neighbors drawn from basins at the SAME centroid distance as the real one.

    For each target and each real neighbor, candidates are the basins whose great-circle
    centroid distance to the target is within tol_km of the real neighbor's distance,
    excluding the target itself and its real neighbors. The tolerance doubles until at
    least three candidates exist; pass tol_used to receive {target: widest tolerance used}.

    This is the control random_degree_matched cannot be: a mascon-leakage or
    shared-forcing artefact lives in proximity, and here proximity is held fixed, so a
    surviving real-minus-placebo gap has to come from neighbor identity.
    """
    dist = distance_matrix_km(meta)
    names = [n for n in graph if n in dist.index]
    if len(names) != len(graph):
        raise KeyError("meta is missing centroids for some graph targets")
    d = dist.loc[names, names].to_numpy()
    return _matched_draw(graph, d, names, seed, tol_km, _MAX_TOL_KM, None, tol_used)


def random_correlation_matched(
    graph: dict[str, list[str]],
    wide_train: pd.DataFrame,
    seed: int,
    tol: float = 0.05,
    tol_used: dict[str, float] | None = None,
) -> dict[str, list[str]]:
    """Placebo neighbors drawn from basins with the SAME training-window correlation.

    Candidates are restricted to positive correlations, matching corr_topk's own rule, and
    to |corr(target, candidate) - corr(target, real neighbor)| <= tol; the tolerance
    doubles until at least three candidates exist. Correlations come from the training
    window only, so the draw leaks no test information. Pass tol_used to receive
    {target: widest tolerance used}.

    Held against random_degree_matched this separates "a well-correlated partner helps"
    from "this particular partner helps": both arms now see an equally correlated series.
    """
    corr = wide_train.corr()
    names = [n for n in graph if n in corr.index]
    if len(names) != len(graph):
        raise KeyError("wide_train is missing columns for some graph targets")
    c = corr.loc[names, names].to_numpy()
    # Correlations live on [-1, 1], so a tolerance of 2 already spans the whole range
    return _matched_draw(graph, c, names, seed, tol, 2.0, c > 0, tol_used)
