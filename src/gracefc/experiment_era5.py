"""Phase 6 engine: ERA5-Land basin forcings as exogenous features on the Kalman backbone.

Same construction as experiment_nonlinear — heads (ridge, GBM, MLP) learn the residual
(target - kalman) — but every arm now comes in a with-ERA5 twin: own state, own+ERA5,
own+neighbor, own+neighbor+ERA5. ERA5 features are deseasonalized and standardized on the
train window per fold (identical treatment to the target), and every arm real or placebo is
fit on the identical row set, so gains isolate added information rather than sample changes.
Placebos randomize ONLY the neighbor while keeping ERA5 aboard: the graph test becomes
"does the neighbor still beat chance once shared meteorology is conditioned on?" Placebo
heads reuse the real arms' seeds (ridge/GBM always did; the MLP now does too — audit
2026-08-15), and draws are seeded per (fold, horizon) cell.
"""
import zlib

import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold, deseasonalize_fold, split_fold
from .experiment_nonlinear import _fit_head
from .graphs import corr_topk, random_degree_matched
from .kalman import filtered_state_wide, fit_fold_params


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
    """Returns (pred rows for real arms, aggregated monthly losses for placebo arms)."""
    out, placebo_monthly = [], []
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

        train_src = resid_wide[resid_wide.index < fold.test_start]
        graph = corr_topk(train_src[names], 1)

        # Fold-safe ERA5 features: climatology and std fit strictly before test_start
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)

        F = filt.values
        R = rho[names].values
        for h in horizons:
            prop = F * (R[None, :] ** h)
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
            # ERA5 joins on (issue_date, name); dropping its NaNs BEFORE any arm is fit keeps
            # every model — with or without ERA5 — on exactly the same rows
            base_df = base_df.merge(era5_feats, on=["issue_date", "name"], how="left")
            base_df = base_df.dropna(subset=era5_cols)
            tr, te = split_fold(base_df, fold)
            if len(tr) < 100 or len(te) == 0:
                continue

            t_idx = filt.index.get_indexer(tr["issue_date"].values)
            e_idx = filt.index.get_indexer(te["issue_date"].values)
            tr_pos = np.array([name_pos[n] for n in tr["name"].values])
            te_pos = np.array([name_pos[n] for n in te["name"].values])

            def node_feat(node_of: dict, row_t, row_p) -> np.ndarray:
                idx = np.array([name_pos.get(node_of.get(names[p]), -1)
                                if node_of.get(names[p]) is not None else -1 for p in row_p])
                return np.where(idx >= 0, prop[row_t, np.clip(idx, 0, None)], 0.0)

            ytr = (tr["target"] - tr["kalman"]).values
            own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values
            era_tr, era_te = tr[era5_cols].values, te[era5_cols].values

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

            emit("kalman_ar1", te["kalman"].values)

            nbr1 = {n: (graph.get(n, [None]) + [None])[0] for n in names}
            f1_tr, f1_te = node_feat(nbr1, t_idx, tr_pos), node_feat(nbr1, e_idx, te_pos)

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
                g_rand = random_degree_matched(graph, cell_base + seed)
                r1 = {n: (g_rand.get(n, [None]) + [None])[0] for n in names}
                p1_tr, p1_te = node_feat(r1, t_idx, tr_pos), node_feat(r1, e_idx, te_pos)
                Xtr_p = np.column_stack([own_tr, p1_tr, era_tr])
                Xte_p = np.column_stack([own_te, p1_te, era_te])
                emit_placebo(f"ridge_corr_top1_era5_rand{seed}",
                             te["kalman"].values + _fit_head("ridge", Xtr_p, ytr, Xte_p, 0))
                emit_placebo(f"gbm_corr_top1_era5_rand{seed}",
                             te["kalman"].values + _fit_head("gbm", Xtr_p, ytr, Xte_p, 0))
                for s in mlp_seeds:
                    emit_placebo(f"mlp_corr_top1_era5_s{s}_rand{seed}",
                                 te["kalman"].values + _fit_head("mlp", Xtr_p, ytr, Xte_p, s))
                # No-ERA5 ridge placebo on the same rows: re-anchors the Phase 3b null here
                emit_placebo(f"ridge_corr_top1_rand{seed}",
                             te["kalman"].values + _fit_head(
                                 "ridge", np.column_stack([own_tr, p1_tr]), ytr,
                                 np.column_stack([own_te, p1_te]), 0))
            print(f"{fold.name} h{h} done", flush=True)
    return (pd.concat(out, ignore_index=True),
            pd.concat(placebo_monthly, ignore_index=True) if placebo_monthly else pd.DataFrame())
