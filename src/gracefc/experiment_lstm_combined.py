"""Two-stage correction: an LSTM on own-basin inputs, then an MLP on a neighbor's state.

The forecast is kalman + LSTM + MLP. Stage 1 is the shared-encoder LSTM of experiment_lstm
over 12-month windows of the basin's filtered state and the 11 ERA5-Land anomaly channels
(lstm_own_era5), or the same with the correlation top-1 neighbor's filtered-state history
as a further channel (lstm_corr_top1_era5). Stage 2 is an sklearn MLP fitted to the
stage-1 training residual (target - kalman - lstm) from the neighbor's propagated state
alone. lstmres_corr_top1 adds stage 2 to lstm_own_era5, and lstmres_nbrin_corr_top1 adds
it to lstm_corr_top1_era5, so the arms supply the neighbor as a second stage, as an LSTM
input channel, or both.

Placebo arms randomize only the stage-2 neighbor graph and refit the MLP with the same
seed as the real arm. The stage-1 nets are shared across placebo draws for each
(arm, seed), as the ridge stage is in experiment_resmlp. Placebo draws are seeded per
(fold, lead time).

The stage-2 residuals are computed on rows the LSTM was trained on (about 85 %) or
early-stopped on (about 15 %), so they are in-sample and generally smaller than test-time
residuals. The test evaluation is unaffected, since challenger and reference share the
stage-1 net and the MLP sees no test data, but the fitted correction can differ from one
fitted to out-of-sample residuals. The oof arms fit stage 2 to out-of-fold residuals
instead (see run_lstm_combined_experiment).
"""
import zlib
from functools import partial

import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_flat12 import _era5_state_tensor, _state_channel, _window_channels
from .experiment_lstm import lstm_predict, train_lstm
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (emit_placebo_rows, emit_rows, flat_ridge_arms, fold_setup, horizon_frame,
                     neighbor_rank_matrix, propagated_neighbor_features, stack_channels as stack,
                     train_val_mask)


def run_lstm_combined_experiment(
    wide: pd.DataFrame,
    era5_wide: dict[str, pd.DataFrame],
    horizons: range = range(1, 4),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_placebo: int = 20,
    seeds: tuple[int, ...] = (0, 1),
    params_cache: dict | None = None,
    era5_lags: tuple[int, ...] = (0, 1, 2),
    oof_blocks: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prediction rows for the real arms, monthly summed losses for the placebo arms).

    With oof_blocks > 0, the lstm_own_era5 backbone also gets stage-2 arms fitted to
    out-of-fold stage-1 residuals: the training issue months are split into oof_blocks
    contiguous blocks, and each block's residuals come from an LSTM trained without it.
    These arms are lstmres_oof_corr_top1 and the own-state control lstmres_oof_own.
    lstmres_corr_top1_hist12 gives stage 2 the neighbor's 12-month filtered-state
    history, the representation the input-channel arm receives, in place of its
    propagated state.
    """
    out, placebo_monthly = [], []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos, F = setup["names"], setup["name_pos"], setup["F"]
        graph = corr_topk(setup["train_src"], 1)
        nbr_idx = neighbor_rank_matrix(graph, names, name_pos, 1)
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)
        E = _era5_state_tensor(era5_feats, era5_wide, setup["filt"].index, names)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            kal_te = te["kalman"].values
            val_mask = train_val_mask(tr)

            widx_tr, valid_tr = _window_channels(frame["t_idx"])
            widx_te, valid_te = _window_channels(frame["e_idx"])
            own_tr = _state_channel(F, widx_tr, valid_tr, frame["tr_pos"])
            own_te = _state_channel(F, widx_te, valid_te, frame["te_pos"])
            era_tr = np.where(valid_tr[:, :, None], E[widx_tr, frame["tr_pos"][:, None], :], 0.0)
            era_te = np.where(valid_te[:, :, None], E[widx_te, frame["te_pos"][:, None], :], 0.0)
            nb_tr = _state_channel(F, widx_tr, valid_tr, nbr_idx[frame["tr_pos"], 0])
            nb_te = _state_channel(F, widx_te, valid_te, nbr_idx[frame["te_pos"], 0])

            emit = partial(emit_rows, out, te, fold, h)
            emit_placebo = partial(emit_placebo_rows, placebo_monthly, te, fold, h)

            emit("kalman_ar1", kal_te)

            # The same ridge twins as in experiment_lstm
            fn_tr = propagated_neighbor_features(frame, nbr_idx, "tr")
            fn_te = propagated_neighbor_features(frame, nbr_idx, "te")
            for arm, (Xtr, Xte) in flat_ridge_arms(tr, te, era5_cols, fn_tr, fn_te).items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            # Stage-2 control without a neighbor: a self-graph passes each basin its own
            # propagated state through the same feature path, so lstmres_own measures the
            # effect of a second stage without neighbor information
            self_idx = np.arange(len(names), dtype=int).reshape(-1, 1)
            fs_tr = propagated_neighbor_features(frame, self_idx, "tr")
            fs_te = propagated_neighbor_features(frame, self_idx, "te")

            # Stage 1 per seed, without and with the neighbor input channel
            stage1 = {
                "lstm_own_era5": stack([own_tr, era_tr], [own_te, era_te]),
                "lstm_corr_top1_era5": stack([own_tr, nb_tr, era_tr], [own_te, nb_te, era_te]),
            }
            stacked_of = {"lstm_own_era5": "lstmres_corr_top1",
                          "lstm_corr_top1_era5": "lstmres_nbrin_corr_top1"}
            # Keep each seed's stage-1 residual and test prediction so that placebos use the
            # same stage-1 net as the arm they are compared with. The RMSE difference between
            # seeds (about 0.002-0.009) is larger than the placebo spread (sd about 0.0008).
            resid_by, lstm_te_by, oof_by = {}, {}, {}
            for arm, (Xtr, Xte) in stage1.items():
                for s in seeds:
                    net = train_lstm(Xtr, ytr, val_mask, s)
                    lstm_te = lstm_predict(net, Xte)
                    emit(f"{arm}_s{s}", kal_te + lstm_te)
                    resid2 = ytr - lstm_predict(net, Xtr)
                    emit(f"{stacked_of[arm]}_s{s}",
                         kal_te + lstm_te + _fit_head("mlp", fn_tr, resid2, fn_te, s))
                    if arm == "lstm_own_era5":
                        emit(f"lstmres_own_s{s}",
                             kal_te + lstm_te + _fit_head("mlp", fs_tr, resid2, fs_te, s))
                        # Stage 2 on the neighbor's 12-month history, the representation
                        # the input-channel arm receives
                        emit(f"lstmres_corr_top1_hist12_s{s}",
                             kal_te + lstm_te + _fit_head("mlp", nb_tr, resid2, nb_te, s))
                        if oof_blocks > 0:
                            oof = np.empty_like(ytr)
                            blocks = np.array_split(np.sort(tr["issue_date"].unique()),
                                                    oof_blocks)
                            for blk in blocks:
                                in_blk = tr["issue_date"].isin(blk).values
                                sub_val = train_val_mask(tr[~in_blk])
                                net_k = train_lstm(Xtr[~in_blk], ytr[~in_blk], sub_val, s)
                                oof[in_blk] = ytr[in_blk] - lstm_predict(net_k, Xtr[in_blk])
                            emit(f"lstmres_oof_corr_top1_s{s}",
                                 kal_te + lstm_te + _fit_head("mlp", fn_tr, oof, fn_te, s))
                            emit(f"lstmres_oof_own_s{s}",
                                 kal_te + lstm_te + _fit_head("mlp", fs_tr, oof, fs_te, s))
                            oof_by[(arm, s)] = oof
                    resid_by[(arm, s)], lstm_te_by[(arm, s)] = resid2, lstm_te

            # Placebos randomize only the stage-2 neighbor graph, with the MLP seed equal to
            # the real arm's seed s, so within each (arm, s) comparison only the graph
            # differs. Draws are seeded per (fold, lead time).
            cell_base = zlib.crc32(f"phase8_lstm_combined:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                g_rand = random_degree_matched(graph, cell_base + seed)
                p_idx = neighbor_rank_matrix(g_rand, names, name_pos, 1)
                pb_tr = propagated_neighbor_features(frame, p_idx, "tr")
                pb_te = propagated_neighbor_features(frame, p_idx, "te")
                for arm in stage1:
                    for s in seeds:
                        emit_placebo(f"{stacked_of[arm]}_s{s}_rand{seed}",
                                     kal_te + lstm_te_by[(arm, s)]
                                     + _fit_head("mlp", pb_tr, resid_by[(arm, s)], pb_te, s))
                # Seed-matched placebos for the hist12 and oof arms on the lstm_own_era5 backbone
                p_hist_tr = _state_channel(F, widx_tr, valid_tr, p_idx[frame["tr_pos"], 0])
                p_hist_te = _state_channel(F, widx_te, valid_te, p_idx[frame["te_pos"], 0])
                for s in seeds:
                    key = ("lstm_own_era5", s)
                    emit_placebo(f"lstmres_corr_top1_hist12_s{s}_rand{seed}",
                                 kal_te + lstm_te_by[key]
                                 + _fit_head("mlp", p_hist_tr, resid_by[key], p_hist_te, s))
                    if key in oof_by:
                        emit_placebo(f"lstmres_oof_corr_top1_s{s}_rand{seed}",
                                     kal_te + lstm_te_by[key]
                                     + _fit_head("mlp", pb_tr, oof_by[key], pb_te, s))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
