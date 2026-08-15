"""Phase 8: stacked LSTM + neighbor-only residual MLP — the two Phase 7 winners combined.

Stage 1 is the Phase 7 shared-encoder LSTM over 12-month windows of the Kalman-filtered
own state plus the 11 ERA5 anomaly channels (and, in the _nbrin arms, the corr_top1
neighbor's filtered-state history as a further channel — the literal Phase 7
lstm_corr_top1_era5). Stage 2 is the Phase 7 resMLP correction: an sklearn MLP fits the
STAGE-1 residual (target - kalman - lstm) from the propagated top-1 neighbor state alone,
so final prediction = kalman + lstm + mlp. Placebos randomize ONLY the stage-2 neighbor
graph and retrain the MLP with the SAME model seed as the real arm they null (audit
2026-08-15: the old 1000+draw MLP seed changed init, shuffle, and early-stop split
alongside the graph, so the null was not graph-identity-only); the stage-1 nets are
shared per (arm, seed) across draws, mirroring experiment_resmlp's shared ridge stage.
Placebo graph draws are seeded per (fold, horizon) cell, so the 6 leads x 5 folds are
independent draws rather than one reused set.

Caveat by construction: stage-2 residuals are computed on train rows the LSTM itself
trained on (~85%) or early-stopped on (~15%), so they are in-sample and smaller than
test-time residuals — same design as resmlp's in-sample ridge residuals, but the LSTM
overfits harder than ridge. The EVALUATION stays honest either way (challenger and
reference share the identical stage-1 net and test prediction; the MLP sees no test
information), but the learned correction is plausibly attenuated — and in-sample
residuals are also less noisy than honest ones, so "attenuated" is the likely, not
guaranteed, direction (phase 8 audit, finding 2).
"""
import zlib

import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_lstm import (_era5_state_tensor, _state_channel, _window_channels,
                              lstm_predict, train_lstm)
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (fold_setup, horizon_frame, neighbor_rank_matrix,
                     propagated_neighbor_features, train_val_mask)


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
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms).

    oof_blocks > 0 adds, on the lstm_own_era5 backbone only, stage-2 arms trained on
    OUT-OF-FOLD stage-1 residuals (K contiguous issue-month blocks, one held-out net
    each): lstmres_oof_corr_top1 and the lstmres_oof_own diagnostic. The audit expected
    in-sample residuals to attenuate the correction and to be the mechanism behind the
    own-state control's wrong-direction behavior; these arms answer both. A
    delivery-equalized arm (lstmres_corr_top1_hist12) feeds stage 2 the SAME 12-month
    neighbor history the input-channel arm receives, closing the representation
    confound in the "same information, different delivery" claim (audit 2026-08-15).
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

            def stack(chans_tr: list, chans_te: list) -> tuple[np.ndarray, np.ndarray]:
                Xtr = np.concatenate([c[:, :, None] if c.ndim == 2 else c for c in chans_tr], axis=2)
                Xte = np.concatenate([c[:, :, None] if c.ndim == 2 else c for c in chans_te], axis=2)
                return Xtr, Xte

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

            # Ridge twins on the flat Phase 5/6 features — must stay bit-identical to Phase 7
            flat_own_tr, flat_own_te = tr[["own_state"]].values, te[["own_state"]].values
            flat_era_tr, flat_era_te = tr[era5_cols].values, te[era5_cols].values
            fn_tr = propagated_neighbor_features(frame, nbr_idx, "tr")
            fn_te = propagated_neighbor_features(frame, nbr_idx, "te")
            ridge_arms = {
                "own": (flat_own_tr, flat_own_te),
                "own_era5": (np.column_stack([flat_own_tr, flat_era_tr]),
                             np.column_stack([flat_own_te, flat_era_te])),
                "corr_top1": (np.column_stack([flat_own_tr, fn_tr]),
                              np.column_stack([flat_own_te, fn_te])),
                "corr_top1_era5": (np.column_stack([flat_own_tr, fn_tr, flat_era_tr]),
                                   np.column_stack([flat_own_te, fn_te, flat_era_te])),
            }
            for arm, (Xtr, Xte) in ridge_arms.items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            # Neighbor-free stage-2 control (audit 2026-08-14, phase7_corrected_analysis):
            # a self-graph feeds each basin its OWN propagated state through the identical
            # feature path, so lstmres_own isolates the "any second stage helps" term from
            # the "the neighbor's information helps" term the lstmres_corr_top1 arm claims
            self_idx = np.arange(len(names), dtype=int).reshape(-1, 1)
            fs_tr = propagated_neighbor_features(frame, self_idx, "tr")
            fs_te = propagated_neighbor_features(frame, self_idx, "te")

            # Stage 1 per seed: own_era5 (Phase 7 winner) and corr_top1_era5 (literal variant 2)
            stage1 = {
                "lstm_own_era5": stack([own_tr, era_tr], [own_te, era_te]),
                "lstm_corr_top1_era5": stack([own_tr, nb_tr, era_tr], [own_te, nb_te, era_te]),
            }
            stacked_of = {"lstm_own_era5": "lstmres_corr_top1",
                          "lstm_corr_top1_era5": "lstmres_nbrin_corr_top1"}
            # Keep every seed's stage-1 residual and test prediction: placebos must ride the
            # SAME stage-1 net as the arm they are compared against. Scoring seed-1 arms
            # against seed-0 placebos made the seed gap (~0.002-0.009 RMSE) swamp the placebo
            # spread (sd ~0.0008) and produced a spurious 16/20 cell (phase 8 audit, 2026-08-15).
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
                        # Delivery-equalized: stage 2 sees the neighbor's 12-month
                        # history — the input-channel arm's exact representation
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

            # Placebos randomize the stage-2 neighbor graph ONLY: the MLP seed equals the
            # real arm's seed s, so within each (arm, s) comparison everything but graph
            # identity is held fixed. Draws are per (fold, horizon) cell.
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
                # Seed-matched nulls for the new arms on the own_era5 backbone
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
