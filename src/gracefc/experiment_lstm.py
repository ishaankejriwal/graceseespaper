"""Phase 7 arch 2: shared-encoder LSTM over Kalman-filtered state sequences.

One small LSTM (32 units, 1 layer) is trained across all basins per fold-horizon on
12-month windows of the filtered state — own basin, optionally the corr_top1 neighbor's
filtered-state history as a second channel, optionally the 11 ERA5 standardized anomaly
series as extra channels — and learns the residual (target - kalman), so kalman + LSTM
is the forecast. Per-basin training is off the table with ~200 train months per basin;
a shared encoder is the honest capacity for 234 basins. Months before the record start
pad with zeros, which is the state prior mean, so padding is principled rather than
arbitrary. Early stopping holds out the LAST 15% of train issue months (time-ordered,
so the stopping rule never sees the future). Placebos swap the neighbor channel for a
degree-matched random basin's history and retrain with the SAME torch seed as the real
arm they null (audit 2026-08-15: the old 1000+draw seed changed net init and batch
order alongside the graph); ERA5 channels stay aboard in _era5 placebos, mirroring
experiment_era5. Placebo draws are seeded per (fold, horizon) cell.
"""
import copy
import zlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .phase7 import (fold_setup, horizon_frame, neighbor_rank_matrix,
                     propagated_neighbor_features, train_val_mask)

LOOKBACK = 12


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
    """Train on ~val rows, early-stop on the val tail, return the best net."""
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
    """Train on ~val rows, early-stop on the val tail, return test predictions."""
    net = train_lstm(Xtr, ytr, val_mask, seed, hidden, max_epochs, patience, batch, lr)
    return lstm_predict(net, Xte)


def _window_channels(t_idx: np.ndarray) -> tuple:
    """Window index grid and validity mask shared by every channel of a row set."""
    offs = np.arange(LOOKBACK - 1, -1, -1)
    widx = t_idx[:, None] - offs[None, :]
    return np.clip(widx, 0, None), widx >= 0


def _state_channel(mat: np.ndarray, widx: np.ndarray, valid: np.ndarray, node: np.ndarray) -> np.ndarray:
    """(rows, L) history of mat for each row's node; zeros where node or month is absent."""
    ok = valid & (node >= 0)[:, None]
    return np.where(ok, mat[widx, np.clip(node, 0, None)[:, None]], 0.0)


def _era5_state_tensor(era5_feats: pd.DataFrame, era5_wide: dict, filt_index, names) -> np.ndarray:
    """(T, N, V) standardized ERA5 anomalies aligned to the filter grid; NaN -> 0."""
    mats = []
    for var in era5_wide:
        m = era5_feats.pivot(index="issue_date", columns="name", values=f"{var}_l0")
        mats.append(m.reindex(index=filt_index, columns=names).values)
    return np.nan_to_num(np.stack(mats, axis=-1))


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
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms)."""
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
            # ERA5 sequence channels come from the same window grid as the state channel
            era_tr = np.where(valid_tr[:, :, None], E[widx_tr, frame["tr_pos"][:, None], :], 0.0)
            era_te = np.where(valid_te[:, :, None], E[widx_te, frame["te_pos"][:, None], :], 0.0)

            def nbr_channel(idx_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                node_tr = idx_matrix[frame["tr_pos"], 0]
                node_te = idx_matrix[frame["te_pos"], 0]
                return (_state_channel(F, widx_tr, valid_tr, node_tr),
                        _state_channel(F, widx_te, valid_te, node_te))

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

            # Ridge twins on the flat Phase 5/6 features: the linear same-question anchors
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

            # Placebo nets reuse the real arms' torch seeds so each (arm, s) null varies
            # only the graph; labels carry _s{s} so the summary matches seed to seed.
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
