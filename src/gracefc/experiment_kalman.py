"""Phase 3b: does knowing your neighbours help, when added as a plain linear term?

The answer this engine produces is "no" (+0.31% at lead 1, not significant) — which is a
real finding, not a failure. It is the linear half of the study's third contribution; the
nonlinear half lives in experiment_lstm_combined.py and does work.

The care taken below is the reason the null is trustworthy. Every real arm is paired with
a placebo arm that gets randomly chosen neighbours in an identically shaped graph, and the
basin's OWN state is a feature in both arms — so a badly calibrated decay rate can never
show up looking like neighbour skill.

Technical detail follows.


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

from .evaluate import DEFAULT_FOLDS, Fold
from .graphs import random_degree_matched, resolve_builder
from .phase7 import (emit_rows, fold_setup, horizon_frame, neighbor_rank_matrix,
                     propagated_neighbor_features)


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
    # Index lags at or before issue only, joined identically into every arm
    idx_feats, idx_cols = None, []
    if indices is not None:
        idx_feats = pd.DataFrame({f"{c}_l{lag}": indices[c].shift(lag)
                                  for c in indices.columns for lag in index_lags})
        idx_cols = list(idx_feats.columns)
        idx_feats = idx_feats.rename_axis("issue_date").reset_index()
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos = setup["names"], setup["name_pos"]
        graphs = {}
        for kind, k in cells:
            g = resolve_builder(kind)(setup["train_src"], meta_kept, k)
            graphs[f"{kind}_top{k}"] = g
            base = zlib.crc32(f"{kind}_top{k}".encode()) % 1_000_000
            for seed in range(n_random_seeds):
                graphs[f"{kind}_top{k}_rand{seed}"] = random_degree_matched(g, base + seed)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, idx_feats, idx_cols)
            if frame is None:
                continue
            tr, te = frame["tr"], frame["te"]
            emit_rows(out, te, fold, h, "kalman_ar1", te["kalman"].values)

            # Own-state correction: the equal-footing comparator both arms must beat
            own_cols = ["own_state"] + idx_cols
            r_own = Ridge(alpha=1.0).fit(tr[own_cols].values, frame["ytr"])
            emit_rows(out, te, fold, h, "kalman_own_ridge",
                      te["kalman"].values + r_own.predict(te[own_cols].values))

            for gname, graph in graphs.items():
                kmax = max((len(v) for v in graph.values()), default=0)
                if kmax == 0:
                    continue
                # Neighbor propagated-state matrix per rank, zero-padded where degree < kmax
                nbr_idx = neighbor_rank_matrix(graph, names, name_pos, kmax)
                Xtr = np.column_stack([tr[own_cols].values, propagated_neighbor_features(frame, nbr_idx, "tr")])
                Xte = np.column_stack([te[own_cols].values, propagated_neighbor_features(frame, nbr_idx, "te")])
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
                    emit_rows(out, te, fold, h, f"kalman_{gname}", pred)

    return (
        pd.concat(out, ignore_index=True),
        pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame(),
        pd.concat(placebo_basin, ignore_index=True) if placebo_basin else pd.DataFrame(),
    )
