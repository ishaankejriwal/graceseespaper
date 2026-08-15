"""Phase 7 arch 3: one-layer graph attention over the correlation graph.

Nodes are basins; directed edges point from each basin's corr_top-k sources into it; node
features are the Kalman-filtered state (plus the 33 ERA5 lag anomalies in _era5 arms).
A single attention layer aggregates messages from the k sources, so with k=2 the network
learns how to weight a second neighbor — the direct test of whether graph structure adds
anything beyond the top-1 edge. Note the _era5 message also carries the NEIGHBOR's ERA5,
information the flat ridge twin never sees. The head predicts the residual
(target - kalman) at every (month, basin); training touches only pre-test months, and
messages use same-issue-month features only, so causality is untouched. Placebos rewire
to degree-matched random graphs and retrain with the SAME torch seed as the real arm
they null (audit 2026-08-15: the old 1000+draw seed changed net init alongside the
graph); draws are seeded per (fold, horizon) cell. Training is full-batch Adam with
early stopping on the last 15% of train issue months.
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


class _GATLite(nn.Module):
    """One attention head, one message-passing layer, linear residual readout."""

    def __init__(self, d_in: int, hidden: int = 16):
        super().__init__()
        self.W = nn.Linear(d_in, hidden, bias=False)
        self.a_dst = nn.Linear(hidden, 1, bias=False)
        self.a_src = nn.Linear(hidden, 1, bias=False)
        self.self_lin = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X, nbr, mask):
        # X: (T, N, d); nbr: (N, k) source indices; mask: (N, k) validity
        Wx = self.W(X)                                   # (T, N, H)
        nb = Wx[:, nbr, :]                               # (T, N, k, H)
        e = torch.nn.functional.leaky_relu(
            self.a_dst(Wx).unsqueeze(2) + self.a_src(nb), negative_slope=0.2)
        e = e.masked_fill(~mask[None, :, :, None], -1e9)
        alpha = torch.softmax(e, dim=2) * mask[None, :, :, None]
        m = (alpha * nb).sum(dim=2)                      # (T, N, H)
        h = torch.relu(self.self_lin(X) + m)
        return self.head(h).squeeze(-1)                  # (T, N)


def fit_gat(
    X: np.ndarray, nbr_idx: np.ndarray,
    tr_entries: tuple, ytr: np.ndarray, val_mask: np.ndarray, te_entries: tuple, seed: int,
    hidden: int = 16, max_epochs: int = 400, patience: int = 30, lr: float = 5e-3,
) -> np.ndarray:
    """Full-batch training on train (t, node) entries; returns predictions at test entries."""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    nbr = torch.tensor(np.clip(nbr_idx, 0, None), dtype=torch.long)
    mask = torch.tensor(nbr_idx >= 0)
    t_tr = torch.tensor(tr_entries[0][~val_mask]), torch.tensor(tr_entries[1][~val_mask])
    t_va = torch.tensor(tr_entries[0][val_mask]), torch.tensor(tr_entries[1][val_mask])
    ya = torch.tensor(ytr[~val_mask], dtype=torch.float32)
    yv = torch.tensor(ytr[val_mask], dtype=torch.float32)
    net = _GATLite(X.shape[2], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best_state, best_val, bad = copy.deepcopy(net.state_dict()), np.inf, 0
    for _ in range(max_epochs):
        net.train()
        opt.zero_grad()
        pred = net(Xt, nbr, mask)
        loss = lossf(pred[t_tr[0], t_tr[1]], ya)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = float(lossf(net(Xt, nbr, mask)[t_va[0], t_va[1]], yv))
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        full = net(Xt, nbr, mask)
    return full[torch.tensor(te_entries[0]), torch.tensor(te_entries[1])].numpy()


def _era5_node_tensor(era5_feats: pd.DataFrame, era5_cols: list[str], filt_index, names) -> np.ndarray:
    """(T, N, C) ERA5 lag features aligned to the filter grid; NaN -> 0 for message passing."""
    mats = []
    for col in era5_cols:
        m = era5_feats.pivot(index="issue_date", columns="name", values=col)
        mats.append(m.reindex(index=filt_index, columns=names).values)
    return np.nan_to_num(np.stack(mats, axis=-1))


def run_gnn_experiment(
    wide: pd.DataFrame,
    era5_wide: dict[str, pd.DataFrame],
    horizons: range = range(1, 4),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_placebo: int = 20,
    gnn_seeds: tuple[int, ...] = (0, 1, 2),
    params_cache: dict | None = None,
    era5_lags: tuple[int, ...] = (0, 1, 2),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms)."""
    out, placebo_monthly = [], []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names, name_pos, F = setup["names"], setup["name_pos"], setup["F"]
        graph = corr_topk(setup["train_src"], 2)
        idx2 = neighbor_rank_matrix(graph, names, name_pos, 2)
        idx1 = idx2[:, :1]
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)
        E = _era5_node_tensor(era5_feats, era5_cols, setup["filt"].index, names)
        X_state = F[:, :, None]
        X_era5 = np.concatenate([X_state, E], axis=2)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            kal_te = te["kalman"].values
            val_mask = train_val_mask(tr)
            tr_entries = (frame["t_idx"], frame["tr_pos"])
            te_entries = (frame["e_idx"], frame["te_pos"])

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

            # Flat ridge twins as linear anchors (own ERA5 only; the GNN also sees neighbor ERA5)
            flat_own_tr, flat_own_te = tr[["own_state"]].values, te[["own_state"]].values
            flat_era_tr, flat_era_te = tr[era5_cols].values, te[era5_cols].values
            nb_tr = propagated_neighbor_features(frame, idx2, "tr")
            nb_te = propagated_neighbor_features(frame, idx2, "te")
            ridge_arms = {
                "own": (flat_own_tr, flat_own_te),
                "corr_top1": (np.column_stack([flat_own_tr, nb_tr[:, :1]]),
                              np.column_stack([flat_own_te, nb_te[:, :1]])),
                "corr_top2": (np.column_stack([flat_own_tr, nb_tr]),
                              np.column_stack([flat_own_te, nb_te])),
                "own_era5": (np.column_stack([flat_own_tr, flat_era_tr]),
                             np.column_stack([flat_own_te, flat_era_te])),
                "corr_top1_era5": (np.column_stack([flat_own_tr, nb_tr[:, :1], flat_era_tr]),
                                   np.column_stack([flat_own_te, nb_te[:, :1], flat_era_te])),
                "corr_top2_era5": (np.column_stack([flat_own_tr, nb_tr, flat_era_tr]),
                                   np.column_stack([flat_own_te, nb_te, flat_era_te])),
            }
            for arm, (Xtr, Xte) in ridge_arms.items():
                emit(f"ridge_{arm}", kal_te + _fit_head("ridge", Xtr, ytr, Xte, 0))

            arms = {
                "corr_top1": (X_state, idx1),
                "corr_top2": (X_state, idx2),
                "corr_top1_era5": (X_era5, idx1),
                "corr_top2_era5": (X_era5, idx2),
            }
            for arm, (X, nidx) in arms.items():
                for s in gnn_seeds:
                    emit(f"gnn_{arm}_s{s}",
                         kal_te + fit_gat(X, nidx, tr_entries, ytr, val_mask, te_entries, s))

            # Placebo nets reuse the real arms' torch seeds so each (arm, s) null varies
            # only the graph; labels carry _s{s} so the summary matches seed to seed.
            cell_base = zlib.crc32(f"phase7_gnn:{fold.name}:h{h}".encode()) % 1_000_000
            for seed in range(n_placebo):
                g_rand = random_degree_matched(graph, cell_base + seed)
                p2 = neighbor_rank_matrix(g_rand, names, name_pos, 2)
                p1 = p2[:, :1]
                plac_arms = {
                    "corr_top1": (X_state, p1),
                    "corr_top2": (X_state, p2),
                    "corr_top1_era5": (X_era5, p1),
                    "corr_top2_era5": (X_era5, p2),
                }
                for arm, (X, nidx) in plac_arms.items():
                    for s in gnn_seeds:
                        emit_placebo(f"gnn_{arm}_s{s}_rand{seed}",
                                     kal_te + fit_gat(X, nidx, tr_entries, ytr, val_mask,
                                                      te_entries, s))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
