"""Phase 7 arch 1: the Africa-study two-stage head — ridge on own state, MLP on neighbors only.

Ridge fits the Kalman residual from own_state; an MLP then fits the RIDGE residual from
neighbor propagated states alone (plus ERA5 lags in the _era5 arms, ERA5-only in the
own_era5 control). Final prediction = kalman + ridge + MLP. Information sets match the
Phase 5/6 one-stage twins exactly, so resmlp vs ridge_corr_top1 isolates the two-stage
architecture rather than the features. Placebos randomize ONLY the neighbor graph and
retrain the MLP stage with the SAME seed as the real arm they null (audit 2026-08-15:
the old 1000+draw seed changed MLP init/shuffle/early-stop split alongside the graph);
the ridge stage is neighbor-free and shared. ERA5 stays aboard in _era5 placebos,
mirroring experiment_era5's "neighbor beyond shared meteorology" test. Placebo draws are
seeded per (fold, horizon) cell, so cells are independent draws rather than one reused set.
"""
import zlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (fold_setup, horizon_frame, neighbor_rank_matrix,
                     propagated_neighbor_features)


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
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms)."""
    out, placebo_monthly = [], []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos = setup["names"], setup["name_pos"]
        # One top-2 graph serves both arms: rank 0 of corr_top2 IS the corr_top1 neighbor
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

            emit("kalman_ar1", kal_te)

            # One-stage ridge twins: DM references that pin down what the architecture adds
            ridge_arms = {
                "own": (own_tr, own_te),
                "own_era5": (np.column_stack([own_tr, era_tr]), np.column_stack([own_te, era_te])),
                "corr_top1": (np.column_stack([own_tr, nb_tr[:, :1]]),
                              np.column_stack([own_te, nb_te[:, :1]])),
                "corr_top2": (np.column_stack([own_tr, nb_tr]), np.column_stack([own_te, nb_te])),
                "corr_top1_era5": (np.column_stack([own_tr, nb_tr[:, :1], era_tr]),
                                   np.column_stack([own_te, nb_te[:, :1], era_te])),
                "corr_top2_era5": (np.column_stack([own_tr, nb_tr, era_tr]),
                                   np.column_stack([own_te, nb_te, era_te])),
            }
            for arm, (Xtr, Xte) in ridge_arms.items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            # Stage 1 once per fold-horizon: ridge on own state only
            ridge0 = Ridge(alpha=1.0).fit(own_tr, ytr)
            r_tr, r_te = ridge0.predict(own_tr), ridge0.predict(own_te)
            resid2 = ytr - r_tr

            # Stage 2: MLP sees ONLY the listed correction features (never own state)
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

            # Placebo MLPs reuse the real arms' seeds so each (arm, s) null varies only
            # the graph; labels carry _s{s} so the summary matches seed to seed.
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
