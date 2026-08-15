"""Phase 5 engine: nonlinear heads (GBM, MLP) on the Kalman backbone, plus a 2-hop arm.

Same construction as experiment_kalman — heads learn the residual (target - kalman) from
own_state plus neighbor propagated states — so any head's gain over its ridge twin isolates
nonlinearity, and any gain over its placebo twin isolates the graph. The 2-hop arm chains
corr_top1 (neighbor's neighbor); mutual top-1 pairs have no distinct 2-hop node and are
zero-padded, and placebos zero-pad the exact same basins so capacity stays matched.
"""
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .evaluate import DEFAULT_FOLDS, Fold, deseasonalize_fold, split_fold
from .graphs import corr_topk, random_degree_matched
from .kalman import filtered_state_wide, fit_fold_params


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
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        if params_cache is not None and fold.name in params_cache:
            params = params_cache[fold.name]
        else:
            params = fit_fold_params(resid_wide, fold.test_start)
            if params_cache is not None:
                params_cache[fold.name] = params
        filt = filtered_state_wide(resid_wide[params["name"]], params)
        rho = params.set_index("name")["rho"]
        names = list(filt.columns)
        name_pos = {n: i for i, n in enumerate(names)}

        train_src = resid_wide[resid_wide.index < fold.test_start]
        graph = corr_topk(train_src[names], 1)
        hop2 = two_hop_map(graph)
        base = zlib.crc32(b"nonlinear_corr_top1") % 1_000_000

        F = filt.values
        R = rho[names].values
        for h in horizons:
            prop = F * (R[None, :] ** h)
            base_df = pd.DataFrame({
                "issue_date": np.repeat(filt.index.values, len(names)),
                "name": np.tile(names, filt.shape[0]),
                "kalman": prop.ravel(),
                "own_state": F.ravel(),
            })
            tgt = resid_wide[names].shift(-h).stack(future_stack=True).rename("target").reset_index()
            tgt.columns = ["issue_date", "name", "target"]
            base_df = base_df.merge(tgt, on=["issue_date", "name"]).dropna(subset=["target", "kalman"])
            base_df["target_date"] = base_df["issue_date"] + pd.DateOffset(months=h)
            tr, te = split_fold(base_df, fold)
            if len(tr) < 100 or len(te) == 0:
                continue

            t_idx = filt.index.get_indexer(tr["issue_date"].values)
            e_idx = filt.index.get_indexer(te["issue_date"].values)
            tr_pos = np.array([name_pos[n] for n in tr["name"].values])
            te_pos = np.array([name_pos[n] for n in te["name"].values])

            def node_feat(node_of: dict, row_t, row_p) -> np.ndarray:
                # Propagated state of the mapped node per row; zero where the map is empty
                idx = np.array([name_pos.get(node_of.get(names[p]), -1)
                                if node_of.get(names[p]) is not None else -1 for p in row_p])
                return np.where(idx >= 0, prop[row_t, np.clip(idx, 0, None)], 0.0)

            ytr = (tr["target"] - tr["kalman"]).values
            own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values

            def emit(label: str, pred: np.ndarray) -> None:
                df = te[["name", "issue_date", "target_date", "target"]].copy()
                df["pred"] = pred
                df["model"], df["fold"], df["horizon"] = label, fold.name, h
                out.append(df)

            def emit_placebo(label: str, pred: np.ndarray) -> None:
                loss = (te["target"].values - pred) ** 2
                ldf = pd.DataFrame({"target_date": te["target_date"].values, "loss": loss})
                monthly = ldf.groupby("target_date")["loss"].agg(["sum", "count"]).reset_index()
                monthly["model"], monthly["fold"], monthly["horizon"] = label, fold.name, h
                placebo_monthly.append(monthly)

            nbr1 = {n: (graph.get(n, [None]) + [None])[0] for n in names}
            f1_tr, f1_te = node_feat(nbr1, t_idx, tr_pos), node_feat(nbr1, e_idx, te_pos)
            f2_tr, f2_te = node_feat(hop2, t_idx, tr_pos), node_feat(hop2, e_idx, te_pos)

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

            for seed in range(n_placebo):
                g_rand = random_degree_matched(graph, base + seed)
                r1 = {n: (g_rand.get(n, [None]) + [None])[0] for n in names}
                p1_tr, p1_te = node_feat(r1, t_idx, tr_pos), node_feat(r1, e_idx, te_pos)
                Xtr1 = np.column_stack([own_tr, p1_tr])
                Xte1 = np.column_stack([own_te, p1_te])
                emit_placebo(f"gbm_corr_top1_rand{seed}",
                             te["kalman"].values + _fit_head("gbm", Xtr1, ytr, Xte1, 0))
                emit_placebo(f"mlp_corr_top1_rand{seed}",
                             te["kalman"].values + _fit_head("mlp", Xtr1, ytr, Xte1, 1000 + seed))
                r2 = randomized_two_hop(hop2, names, base + 500 + seed)
                p2_tr, p2_te = node_feat(r2, t_idx, tr_pos), node_feat(r2, e_idx, te_pos)
                emit_placebo(f"gbm_corr_top1_2hop_rand{seed}",
                             te["kalman"].values + _fit_head(
                                 "gbm", np.column_stack([own_tr, p1_tr, p2_tr]), ytr,
                                 np.column_stack([own_te, p1_te, p2_te]), 0))
        print(f"{fold.name} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
