"""Phase 5 engine: nonlinear heads (GBM, MLP) on the Kalman backbone, plus a 2-hop arm.

Same construction as experiment_kalman — heads learn the residual (target - kalman) from
own_state plus neighbor propagated states — so any head's gain over its ridge twin isolates
nonlinearity, and any gain over its placebo twin isolates the graph. The 2-hop arm chains
corr_top1 (neighbor's neighbor); mutual top-1 pairs have no distinct 2-hop node and are
zero-padded, and placebos zero-pad the exact same basins so capacity stays matched.
"""
import zlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .evaluate import DEFAULT_FOLDS, Fold
from .graphs import corr_topk, random_degree_matched
from .phase7 import (emit_placebo_rows, emit_rows, fold_setup, horizon_frame,
                     neighbor_rank_matrix, propagated_neighbor_features)


def two_hop_map(graph: dict) -> dict:
    """name -> 2-hop node via the top-1 chain, or None when the chain folds back on itself."""
    out = {}
    for name, nbrs in graph.items():
        nbr = nbrs[0] if nbrs else None
        hop2 = (graph.get(nbr) or [None])[0] if nbr else None
        out[name] = hop2 if hop2 not in (None, name) else None
    return out


def randomized_two_hop(hop2: dict, names: list, seed: int) -> dict:
    """Placebo 2-hop: random node wherever the real chain has one, None exactly where it doesn't."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, h2 in hop2.items():
        if h2 is None:
            out[name] = None
        else:
            choices = [n for n in names if n != name]
            out[name] = choices[rng.integers(len(choices))]
    return out


def _hop_matrix(node_of: dict, names: list, name_pos: dict) -> np.ndarray:
    """(N, 1) position matrix from a name -> name-or-None map; -1 where the map is empty."""
    return neighbor_rank_matrix({n: [h] if h else [] for n, h in node_of.items()}, names, name_pos, 1)


def _fit_head(head: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    if head == "ridge":
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        return m.predict(Xte)
    if head == "gbm":
        m = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        ).fit(Xtr, ytr)
        return m.predict(Xte)
    if head == "mlp":
        sc = StandardScaler().fit(Xtr)
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                         early_stopping=True, random_state=seed).fit(sc.transform(Xtr), ytr)
        return m.predict(sc.transform(Xte))
    raise KeyError(head)


def run_nonlinear_experiment(
    wide: pd.DataFrame,
    horizons: range = range(1, 4),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_placebo: int = 20,
    mlp_seeds: tuple[int, ...] = (0, 1, 2),
    params_cache: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms)."""
    out, placebo_monthly = [], []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos = setup["names"], setup["name_pos"]
        graph = corr_topk(setup["train_src"], 1)
        nbr_idx = neighbor_rank_matrix(graph, names, name_pos, 1)
        hop2 = two_hop_map(graph)
        hop_idx = _hop_matrix(hop2, names, name_pos)

        for h in horizons:
            frame = horizon_frame(setup, fold, h)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values
            feat = partial(propagated_neighbor_features, frame)
            emit = partial(emit_rows, out, te, fold, h)
            emit_placebo = partial(emit_placebo_rows, placebo_monthly, te, fold, h)

            f1_tr, f1_te = feat(nbr_idx, "tr"), feat(nbr_idx, "te")
            f2_tr, f2_te = feat(hop_idx, "tr"), feat(hop_idx, "te")

            arms = {
                "own": (own_tr, own_te),
                "corr_top1": (np.column_stack([own_tr, f1_tr]), np.column_stack([own_te, f1_te])),
                "corr_top1_2hop": (np.column_stack([own_tr, f1_tr, f2_tr]),
                                   np.column_stack([own_te, f1_te, f2_te])),
            }
            for arm, (Xtr, Xte) in arms.items():
                emit(f"ridge_{arm}", te["kalman"].values + _fit_head("ridge", Xtr, ytr, Xte, 0))
                emit(f"gbm_{arm}", te["kalman"].values + _fit_head("gbm", Xtr, ytr, Xte, 0))
                if arm != "corr_top1_2hop":
                    for s in mlp_seeds:
                        emit(f"mlp_{arm}_s{s}", te["kalman"].values + _fit_head("mlp", Xtr, ytr, Xte, s))

            # Placebo heads reuse the real arms' seeds (GBM already did; the MLP now does
            # too) so each null varies only the graph. Draws seeded per (fold, horizon).
            cell_base = zlib.crc32(f"nonlinear_corr_top1:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                p_idx = neighbor_rank_matrix(random_degree_matched(graph, cell_base + seed), names, name_pos, 1)
                p1_tr, p1_te = feat(p_idx, "tr"), feat(p_idx, "te")
                Xtr1 = np.column_stack([own_tr, p1_tr])
                Xte1 = np.column_stack([own_te, p1_te])
                emit_placebo(f"gbm_corr_top1_rand{seed}",
                             te["kalman"].values + _fit_head("gbm", Xtr1, ytr, Xte1, 0))
                for s in mlp_seeds:
                    emit_placebo(f"mlp_corr_top1_s{s}_rand{seed}",
                                 te["kalman"].values + _fit_head("mlp", Xtr1, ytr, Xte1, s))
                r2_idx = _hop_matrix(randomized_two_hop(hop2, names, cell_base + 500 + seed), names, name_pos)
                p2_tr, p2_te = feat(r2_idx, "tr"), feat(r2_idx, "te")
                emit_placebo(f"gbm_corr_top1_2hop_rand{seed}",
                             te["kalman"].values + _fit_head(
                                 "gbm", np.column_stack([own_tr, p1_tr, p2_tr]), ytr,
                                 np.column_stack([own_te, p1_te, p2_te]), 0))
        print(f"{fold.name} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
