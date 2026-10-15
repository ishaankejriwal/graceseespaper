"""ERA5-Land forcing as features for ridge, GBM and MLP corrections to the Kalman filter forecast.

As in experiment_nonlinear, each head predicts the filter error (target - kalman). Four
feature sets are used: the own filtered state, own state plus ERA5-Land lags, own state
plus the propagated state of the correlation top-1 neighbor, and all three. ERA5-Land
features are deseasonalized and standardized on each fold's training window, as the
target is, and every real and placebo arm is fitted on the same rows. Placebo arms replace
only the neighbor with a random basin and keep the ERA5-Land features, so they test the
neighbor conditional on the shared forcing. Placebo heads use the same seeds as the real
arms, and the random draws are seeded per (fold, lead time).
"""
import zlib
from functools import partial

import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (emit_placebo_rows, emit_rows, fold_setup, horizon_frame,
                     neighbor_rank_matrix, propagated_neighbor_features)


def run_era5_experiment(
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
        graph = corr_topk(setup["train_src"], 1)
        nbr_idx = neighbor_rank_matrix(graph, names, name_pos, 1)

        # ERA5-Land climatology and standard deviation are fitted before test_start. Rows
        # with missing ERA5-Land features are dropped in horizon_frame before any arm is
        # fitted, so models with and without ERA5-Land use the same rows.
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values
            era_tr, era_te = tr[era5_cols].values, te[era5_cols].values
            emit = partial(emit_rows, out, te, fold, h)
            emit_placebo = partial(emit_placebo_rows, placebo_monthly, te, fold, h)

            emit("kalman_ar1", te["kalman"].values)

            f1_tr = propagated_neighbor_features(frame, nbr_idx, "tr")
            f1_te = propagated_neighbor_features(frame, nbr_idx, "te")

            arms = {
                "own": (own_tr, own_te),
                "own_era5": (np.column_stack([own_tr, era_tr]), np.column_stack([own_te, era_te])),
                "corr_top1": (np.column_stack([own_tr, f1_tr]), np.column_stack([own_te, f1_te])),
                "corr_top1_era5": (np.column_stack([own_tr, f1_tr, era_tr]),
                                   np.column_stack([own_te, f1_te, era_te])),
            }
            for arm, (Xtr, Xte) in arms.items():
                emit(f"ridge_{arm}", te["kalman"].values + _fit_head("ridge", Xtr, ytr, Xte, 0))
                emit(f"gbm_{arm}", te["kalman"].values + _fit_head("gbm", Xtr, ytr, Xte, 0))
                for s in mlp_seeds:
                    emit(f"mlp_{arm}_s{s}", te["kalman"].values + _fit_head("mlp", Xtr, ytr, Xte, s))

            cell_base = zlib.crc32(f"era5_corr_top1:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                p_idx = neighbor_rank_matrix(random_degree_matched(graph, cell_base + seed), names, name_pos, 1)
                p1_tr = propagated_neighbor_features(frame, p_idx, "tr")
                p1_te = propagated_neighbor_features(frame, p_idx, "te")
                Xtr_p = np.column_stack([own_tr, p1_tr, era_tr])
                Xte_p = np.column_stack([own_te, p1_te, era_te])
                emit_placebo(f"ridge_corr_top1_era5_rand{seed}",
                             te["kalman"].values + _fit_head("ridge", Xtr_p, ytr, Xte_p, 0))
                emit_placebo(f"gbm_corr_top1_era5_rand{seed}",
                             te["kalman"].values + _fit_head("gbm", Xtr_p, ytr, Xte_p, 0))
                for s in mlp_seeds:
                    emit_placebo(f"mlp_corr_top1_era5_s{s}_rand{seed}",
                                 te["kalman"].values + _fit_head("mlp", Xtr_p, ytr, Xte_p, s))
                # Ridge placebo without ERA5-Land on the same rows, as in experiment_kalman
                emit_placebo(f"ridge_corr_top1_rand{seed}",
                             te["kalman"].values + _fit_head(
                                 "ridge", np.column_stack([own_tr, p1_tr]), ytr,
                                 np.column_stack([own_te, p1_te]), 0))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
