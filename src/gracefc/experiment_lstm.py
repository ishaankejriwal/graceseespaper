"""Shared-encoder LSTM correcting the Kalman filter forecast from 12-month filtered-state windows.

One LSTM (32 units, one layer) is trained across all basins for each fold and lead time.
Its input is a 12-month window of the basin's filtered state, optionally with the
correlation top-1 neighbor's filtered state as a second channel and the 11 ERA5-Land
standardized anomaly series as further channels. It predicts the filter error
(target - kalman), so the forecast is kalman + LSTM. The encoder is shared across basins
because each basin has only about 200 training months. Months before the start of the
series are padded with zero, the prior mean of the state. Early stopping holds out the
last 15 % of training issue months. Placebo arms replace the neighbor channel with a
degree-matched random basin's history and retrain with the same torch seed as the real
arm; the ERA5-Land channels are kept in the _era5 placebos, as in experiment_era5.
Placebo draws are seeded per (fold, lead time).

The LSTM and MLP of the paper are the lstm_own_era5_s* and mlp_own_era5_flat12_s* arms.
"""
import copy
import zlib
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch import nn

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
# The window construction and the 12-month flattening live in the torch-free module, so
# the ridge models can be produced without torch from the same design matrix.
from .experiment_flat12 import _era5_state_tensor, _state_channel, window_design
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (emit_placebo_rows, emit_rows, flat_ridge_arms, fold_setup, horizon_frame,
                     neighbor_rank_matrix, propagated_neighbor_features, stack_channels as stack,
                     train_val_mask)

# The torch-free helpers are not re-exported; import them from gracefc.experiment_flat12
# so that callers do not need torch.
__all__ = ["run_lstm_experiment", "train_lstm", "lstm_predict", "fit_lstm"]


class _SeqNet(nn.Module):
    def __init__(self, c_in: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1]).squeeze(-1)


def train_lstm(
    Xtr: np.ndarray, ytr: np.ndarray, val_mask: np.ndarray, seed: int,
    hidden: int = 32, max_epochs: int = 60, patience: int = 6, batch: int = 2048, lr: float = 1e-3,
) -> _SeqNet:
    """Train on rows outside val_mask, early-stop on the val_mask rows, return the best net."""
    torch.manual_seed(seed)
    Xa = torch.tensor(Xtr[~val_mask], dtype=torch.float32)
    ya = torch.tensor(ytr[~val_mask], dtype=torch.float32)
    Xv = torch.tensor(Xtr[val_mask], dtype=torch.float32)
    yv = torch.tensor(ytr[val_mask], dtype=torch.float32)
    net = _SeqNet(Xtr.shape[2], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    best_state, best_val, bad = copy.deepcopy(net.state_dict()), np.inf, 0
    n = len(Xa)
    for _ in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(net(Xa[idx]), ya[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xv), yv))
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def lstm_predict(net: _SeqNet, X: np.ndarray) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            preds.append(net(torch.tensor(X[i:i + 8192], dtype=torch.float32)).numpy())
    return np.concatenate(preds)


def fit_lstm(
    Xtr: np.ndarray, ytr: np.ndarray, val_mask: np.ndarray, Xte: np.ndarray, seed: int,
    hidden: int = 32, max_epochs: int = 60, patience: int = 6, batch: int = 2048, lr: float = 1e-3,
) -> np.ndarray:
    """Train as in train_lstm and return predictions for Xte."""
    net = train_lstm(Xtr, ytr, val_mask, seed, hidden, max_epochs, patience, batch, lr)
    return lstm_predict(net, Xte)


def run_lstm_experiment(
    wide: pd.DataFrame,
    era5_wide: dict[str, pd.DataFrame],
    horizons: range = range(1, 4),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_placebo: int = 20,
    lstm_seeds: tuple[int, ...] = (0, 1),
    params_cache: dict | None = None,
    era5_lags: tuple[int, ...] = (0, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prediction rows for the real arms, monthly summed losses for the placebo arms)."""
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

            design = window_design(F, E, frame)
            widx_tr, valid_tr = design["widx_tr"], design["valid_tr"]
            widx_te, valid_te = design["widx_te"], design["valid_te"]
            own_tr, own_te = design["own_tr"], design["own_te"]
            era_tr, era_te = design["era_tr"], design["era_te"]

            def nbr_channel(idx_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                node_tr = idx_matrix[frame["tr_pos"], 0]
                node_te = idx_matrix[frame["te_pos"], 0]
                return (_state_channel(F, widx_tr, valid_tr, node_tr),
                        _state_channel(F, widx_te, valid_te, node_te))

            emit = partial(emit_rows, out, te, fold, h)
            emit_placebo = partial(emit_placebo_rows, placebo_monthly, te, fold, h)

            emit("kalman_ar1", kal_te)

            # Ridge twins on the current state, ERA5-Land lags 0-2 and the
            # neighbor's propagated state
            fn_tr = propagated_neighbor_features(frame, nbr_idx, "tr")
            fn_te = propagated_neighbor_features(frame, nbr_idx, "te")
            for arm, (Xtr, Xte) in flat_ridge_arms(tr, te, era5_cols, fn_tr, fn_te).items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            # Ridge and MLP on the flattened 12-month window, the same information as the
            # LSTM without sequence structure. They separate the effect of the sequence
            # model from that of the longer input history, since the flat references
            # above see only the current state and ERA5-Land lags 0-2.
            X12_tr, X12_te = design["X12_tr"], design["X12_te"]
            emit("ridge_own_flat12", kal_te + _fit_head("ridge", own_tr, ytr, own_te, 0))
            emit("ridge_own_era5_flat12", kal_te + _fit_head("ridge", X12_tr, ytr, X12_te, 0))
            for s in lstm_seeds:
                emit(f"mlp_own_era5_flat12_s{s}",
                     kal_te + _fit_head("mlp", X12_tr, ytr, X12_te, s))

            nb_tr, nb_te = nbr_channel(nbr_idx)
            arms = {
                "own": stack([own_tr], [own_te]),
                "corr_top1": stack([own_tr, nb_tr], [own_te, nb_te]),
                "own_era5": stack([own_tr, era_tr], [own_te, era_te]),
                "corr_top1_era5": stack([own_tr, nb_tr, era_tr], [own_te, nb_te, era_te]),
            }
            for arm, (Xtr, Xte) in arms.items():
                for s in lstm_seeds:
                    emit(f"lstm_{arm}_s{s}", kal_te + fit_lstm(Xtr, ytr, val_mask, Xte, s))

            # Placebo nets reuse the real arms' torch seeds so that each (arm, s) placebo
            # differs only in the graph; the _s{s} label pairs them seed by seed in the summary.
            cell_base = zlib.crc32(f"phase7_lstm:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                g_rand = random_degree_matched(graph, cell_base + seed)
                p_idx = neighbor_rank_matrix(g_rand, names, name_pos, 1)
                pb_tr, pb_te = nbr_channel(p_idx)
                for s in lstm_seeds:
                    Xtr, Xte = stack([own_tr, pb_tr], [own_te, pb_te])
                    emit_placebo(f"lstm_corr_top1_s{s}_rand{seed}",
                                 kal_te + fit_lstm(Xtr, ytr, val_mask, Xte, s))
                    Xtr, Xte = stack([own_tr, pb_tr, era_tr], [own_te, pb_te, era_te])
                    emit_placebo(f"lstm_corr_top1_era5_s{s}_rand{seed}",
                                 kal_te + fit_lstm(Xtr, ytr, val_mask, Xte, s))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
