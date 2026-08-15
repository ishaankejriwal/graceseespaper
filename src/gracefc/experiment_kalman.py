"""Phase 3b engine: neighbor filtered states on the Kalman backbone, paired with placebos.

Design (per Phase 3 analysis): residual target = target - rho_i^h x_i(t). Features are each
neighbor's propagated state rho_j^h x_j(t) plus the target's own state x_i(t) in BOTH the real
and placebo arms, so rho miscalibration can never masquerade as neighbor skill. Missing
neighbor ranks (distance bands can empty a basin's candidate set) are zero-padded, keeping
row sets and capacity identical across every model in a cell.
"""
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .evaluate import DEFAULT_FOLDS, Fold, deseasonalize_fold, split_fold
from .graphs import random_degree_matched
from .kalman import filtered_state_wide, fit_fold_params
from .experiment import resolve_builder


def run_kalman_backbone_experiment(
    wide: pd.DataFrame,
    meta: pd.DataFrame,
    cells: tuple[tuple[str, int], ...] = (("corr", 1), ("corr", 2), ("geo", 1)),
    horizons: range = range(1, 7),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_random_seeds: int = 50,
    params_cache: dict | None = None,
    indices: pd.DataFrame | None = None,
    index_lags: tuple[int, ...] = (0, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (pred_rows for real arms, placebo monthly losses, placebo per-basin losses).

    Placebo arms never emit raw predictions — at 50+ seeds that would be ~1e8 rows — they
    accumulate pooled-by-month and by-basin squared errors instead, which is all the paired
    DM, rank-among-placebos, and per-basin comparisons need.

    indices: optional climate-index table (date-indexed). Its lags are appended to EVERY arm
    — backbone comparator, real graphs, and placebos alike — so any neighbor gain that
    survives is conditioned on shared large-scale forcing rather than explained by it.
    """
    meta_kept = meta[meta["name"].isin(wide.columns)].reset_index(drop=True)
    out = []
    placebo_monthly = []
    placebo_basin = []
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

        train_graph_src = resid_wide[resid_wide.index < fold.test_start]
        graphs = {}
        for kind, k in cells:
            g = resolve_builder(kind)(train_graph_src[names], meta_kept, k)
            graphs[f"{kind}_top{k}"] = g
            base = zlib.crc32(f"{kind}_top{k}".encode()) % 1_000_000
            for seed in range(n_random_seeds):
                graphs[f"{kind}_top{k}_rand{seed}"] = random_degree_matched(g, base + seed)

        F = filt.values  # (time, basin) filtered states
        R = rho[names].values

        for h in horizons:
            prop = F * (R[None, :] ** h)  # each basin's state propagated h months ahead
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
            idx_cols = []
            if indices is not None:
                # Index lags at or before issue only, joined identically into every arm
                for c in indices.columns:
                    for lag in index_lags:
                        col = f"{c}_l{lag}"
                        base_df[col] = base_df["issue_date"].map(indices[c].shift(lag))
                        idx_cols.append(col)
                base_df = base_df.dropna(subset=idx_cols)
            tr, te = split_fold(base_df, fold)
            if len(tr) < 100 or len(te) == 0:
                continue

            def emit(label: str, pred: np.ndarray, frame: pd.DataFrame) -> None:
                df = frame[["name", "issue_date", "target_date", "target"]].copy()
                df["pred"] = pred
                df["model"] = label
                df["fold"] = fold.name
                df["horizon"] = h
                out.append(df)

            emit("kalman_ar1", te["kalman"].values, te)

            # Own-state correction: the equal-footing comparator both arms must beat
            own_cols = ["own_state"] + idx_cols
            r_own = Ridge(alpha=1.0).fit(tr[own_cols].values, (tr["target"] - tr["kalman"]).values)
            emit("kalman_own_ridge", te["kalman"].values + r_own.predict(te[own_cols].values), te)

            t_idx = filt.index.get_indexer(tr["issue_date"].values)
            e_idx = filt.index.get_indexer(te["issue_date"].values)
            tr_pos = np.array([name_pos[n] for n in tr["name"].values])
            te_pos = np.array([name_pos[n] for n in te["name"].values])

            for gname, graph in graphs.items():
                kmax = max((len(v) for v in graph.values()), default=0)
                if kmax == 0:
                    continue
                # Neighbor propagated-state matrix per rank, zero-padded where degree < kmax
                nbr_idx = np.full((len(names), kmax), -1, dtype=int)
                for n, nbrs in graph.items():
                    for r_i, nb in enumerate(nbrs):
                        nbr_idx[name_pos[n], r_i] = name_pos[nb]

                def rank_feats(row_t: np.ndarray, row_p: np.ndarray) -> np.ndarray:
                    cols = []
                    for r_i in range(kmax):
                        j = nbr_idx[row_p, r_i]
                        v = np.where(j >= 0, prop[row_t, np.clip(j, 0, None)], 0.0)
                        cols.append(v)
                    return np.column_stack(cols)

                Xtr = np.column_stack([tr[own_cols].values, rank_feats(t_idx, tr_pos)])
                Xte = np.column_stack([te[own_cols].values, rank_feats(e_idx, te_pos)])
                ok_tr = np.isfinite(Xtr).all(axis=1)
                model = Ridge(alpha=1.0).fit(Xtr[ok_tr], (tr["target"].values - tr["kalman"].values)[ok_tr])
                pred = te["kalman"].values + model.predict(np.nan_to_num(Xte))
                if "_rand" in gname:
                    # Placebo arms: keep aggregated losses only, never raw prediction rows
                    loss = (te["target"].values - pred) ** 2
                    ldf = pd.DataFrame({"target_date": te["target_date"].values,
                                        "name": te["name"].values, "loss": loss})
                    monthly = ldf.groupby("target_date")["loss"].agg(["sum", "count"]).reset_index()
                    monthly["model"], monthly["fold"], monthly["horizon"] = gname, fold.name, h
                    placebo_monthly.append(monthly)
                    basin = ldf.groupby("name")["loss"].agg(["sum", "count"]).reset_index()
                    basin["model"], basin["fold"], basin["horizon"] = gname, fold.name, h
                    placebo_basin.append(basin)
                else:
                    emit(f"kalman_{gname}", pred, te)

    return (
        pd.concat(out, ignore_index=True),
        pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame(),
        pd.concat(placebo_basin, ignore_index=True) if placebo_basin else pd.DataFrame(),
    )
