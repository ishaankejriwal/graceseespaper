"""Two-stage correction: a ridge on the own filtered state, then an MLP on correction features.

The ridge fits the filter error (target - kalman) from own_state, and an MLP then fits the
ridge's training residual from the correction features alone: neighbor propagated states,
with ERA5-Land lags added in the _era5 arms, or ERA5-Land lags only in the own_era5 arm.
The forecast is kalman + ridge + MLP. Each arm has the same information as the one-stage
ridge of the same name (flat_ridge_arms), so resmlp_<arm> and ridge_<arm> differ only in
the two-stage structure. Placebo arms randomize only the neighbor graph and refit the MLP
with the same seed as the real arm; the ridge stage has no neighbor input and is shared.
ERA5-Land features are kept in the _era5 placebos, as in experiment_era5. Placebo draws
are seeded per (fold, lead time).

The resmlp_own_era5_s* arms are the residual MLP of the paper.
"""
import zlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (emit_placebo_rows, emit_rows, flat_ridge_arms, fold_setup, horizon_frame,
                     neighbor_rank_matrix, propagated_neighbor_features)


def run_resmlp_experiment(
    wide: pd.DataFrame,
    era5_wide: dict[str, pd.DataFrame],
    horizons: range = range(1, 4),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_placebo: int = 20,
    mlp_seeds: tuple[int, ...] = (0, 1, 2),
    params_cache: dict | None = None,
    era5_lags: tuple[int, ...] = (0, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prediction rows for the real arms, monthly summed losses for the placebo arms)."""
    out, placebo_monthly = [], []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos = setup["names"], setup["name_pos"]
        # One top-2 graph serves both arms; its first rank is the corr_top1 neighbor
        graph = corr_topk(setup["train_src"], 2)
        nbr_idx = neighbor_rank_matrix(graph, names, name_pos, 2)
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values
            era_tr, era_te = tr[era5_cols].values, te[era5_cols].values
            kal_te = te["kalman"].values
            nb_tr = propagated_neighbor_features(frame, nbr_idx, "tr")
            nb_te = propagated_neighbor_features(frame, nbr_idx, "te")
            emit = partial(emit_rows, out, te, fold, h)
            emit_placebo = partial(emit_placebo_rows, placebo_monthly, te, fold, h)

            emit("kalman_ar1", kal_te)

            # Ridge twins: one-stage ridges on the same features as the two-stage arms
            for arm, (Xtr, Xte) in flat_ridge_arms(tr, te, era5_cols, nb_tr, nb_te).items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            # Stage 1 once per fold-horizon: ridge on own state only
            ridge0 = Ridge(alpha=1.0).fit(own_tr, ytr)
            r_tr, r_te = ridge0.predict(own_tr), ridge0.predict(own_te)
            resid2 = ytr - r_tr

            # Stage 2: the MLP sees only the listed correction features, never the own state
            mlp_arms = {
                "corr_top1": (nb_tr[:, :1], nb_te[:, :1]),
                "corr_top2": (nb_tr, nb_te),
                "own_era5": (era_tr, era_te),
                "corr_top1_era5": (np.column_stack([nb_tr[:, :1], era_tr]),
                                   np.column_stack([nb_te[:, :1], era_te])),
                "corr_top2_era5": (np.column_stack([nb_tr, era_tr]),
                                   np.column_stack([nb_te, era_te])),
            }
            for arm, (Ctr, Cte) in mlp_arms.items():
                for s in mlp_seeds:
                    emit(f"resmlp_{arm}_s{s}",
                         kal_te + r_te + _fit_head("mlp", Ctr, resid2, Cte, s))

            # Placebo MLPs reuse the real arms' seeds so that each (arm, s) placebo differs
            # only in the graph; the _s{s} label pairs them seed by seed in the summary.
            cell_base = zlib.crc32(f"phase7_resmlp:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                g_rand = random_degree_matched(graph, cell_base + seed)
                p_idx = neighbor_rank_matrix(g_rand, names, name_pos, 2)
                pb_tr = propagated_neighbor_features(frame, p_idx, "tr")
                pb_te = propagated_neighbor_features(frame, p_idx, "te")
                plac_arms = {
                    "corr_top1": (pb_tr[:, :1], pb_te[:, :1]),
                    "corr_top2": (pb_tr, pb_te),
                    "corr_top1_era5": (np.column_stack([pb_tr[:, :1], era_tr]),
                                       np.column_stack([pb_te[:, :1], era_te])),
                    "corr_top2_era5": (np.column_stack([pb_tr, era_tr]),
                                       np.column_stack([pb_te, era_te])),
                }
                for arm, (Ctr, Cte) in plac_arms.items():
                    for s in mlp_seeds:
                        emit_placebo(f"resmlp_{arm}_s{s}_rand{seed}",
                                     kal_te + r_te + _fit_head("mlp", Ctr, resid2, Cte, s))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
