"""Phase 3 engine: neighbor-graph treatments vs own-lag baselines inside the fold harness."""
import zlib

import numpy as np
import pandas as pd

from .evaluate import DEFAULT_FOLDS, Fold, deseasonalize_fold, split_fold, assert_no_leakage
from .features import build_lag_table, interpolate_short_gaps, neighbor_features, DEFAULT_LAGS
from .graphs import corr_topk, corr_topk_min_distance, geographic_topk, predictive_lag_topk, random_degree_matched
from .models import fit_residual_mlp, fit_ridge, per_basin_ridge_predict, predict_ridge

GRAPH_BUILDERS = {
    "corr": lambda wide_train, meta, k: corr_topk(wide_train, k),
    "pred_lag1": lambda wide_train, meta, k: predictive_lag_topk(wide_train, k),
    "geo": lambda wide_train, meta, k: geographic_topk(meta, k),
}


def resolve_builder(kind: str):
    """Also accept distance-band kinds like corr_min300: correlation top-k at >=300 km only."""
    if kind in GRAPH_BUILDERS:
        return GRAPH_BUILDERS[kind]
    if kind.startswith("corr_min"):
        min_km = float(kind.removeprefix("corr_min"))
        return lambda wide_train, meta, k: corr_topk_min_distance(wide_train, meta, k, min_km)
    raise KeyError(f"unknown graph kind: {kind}")


def run_neighbor_experiment(
    wide: pd.DataFrame,
    meta: pd.DataFrame,
    graph_kinds: tuple[str, ...] = ("corr", "pred_lag1", "geo"),
    ks: tuple[int, ...] = (1, 2, 3, 5),
    horizons: range = range(1, 7),
    folds: list[Fold] = DEFAULT_FOLDS,
    n_random_seeds: int = 0,
    mlp_seed: int = 0,
    fill_short_gaps: bool = True,
    include_mlp: bool = False,
) -> pd.DataFrame:
    """Treatments and their random placebos, all folds and horizons. Returns tidy prediction rows.

    Primary treatment: neighbor lags appended directly to the pooled ridge — linear shrinkage
    stays well-behaved when neighbors are uninformative, keeping attribution clean. The
    residual MLP is a secondary nonlinearity check (include_mlp). Random placebos are
    degree-matched per (kind, k) so every treatment carries its own equal-capacity null.
    """
    meta_kept = meta[meta["name"].isin(wide.columns)].reset_index(drop=True)
    out = []
    for fold in folds:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        feat_wide, anchors = interpolate_short_gaps(resid_wide) if fill_short_gaps else (None, None)
        graph_src = (feat_wide if feat_wide is not None else resid_wide)
        train_graph_src = graph_src[graph_src.index < fold.test_start]

        # Graphs are rebuilt inside each fold from training data only
        graphs = {}
        for kind in graph_kinds:
            for k in ks:
                g = resolve_builder(kind)(train_graph_src, meta_kept, k)
                graphs[f"{kind}_top{k}"] = g
                # Per-cell seed stream: identical seeds across kinds would reuse placebo draws
                base = zlib.crc32(f"{kind}_top{k}".encode()) % 1_000_000
                for seed in range(n_random_seeds):
                    graphs[f"{kind}_top{k}_rand{seed}"] = random_degree_matched(g, base + seed)

        # Neighbor features depend on fold + graph only, so hoist them out of the horizon loop
        nbr_feats = {
            gname: neighbor_features(graph_src, graph, anchors=anchors)
            for gname, graph in graphs.items()
        }

        for h in horizons:
            lag_df = build_lag_table(resid_wide, horizon=h, feature_wide=feat_wide, anchors=anchors)
            train, test = split_fold(lag_df, fold)
            assert_no_leakage(train, fold, DEFAULT_LAGS, test_rows=test)
            if len(test) == 0 or len(train) == 0:
                continue
            own_cols = [f"lag_{k}" for k in DEFAULT_LAGS]

            # Shared backbone: one pooled own-lag ridge per fold/horizon; treatments add on top
            ridge, scaler = fit_ridge(train, own_cols)
            base_train = predict_ridge(ridge, scaler, train, own_cols)
            base_test = predict_ridge(ridge, scaler, test, own_cols)
            resid_train = train["target"].values - base_train

            def emit(label: str, pred: np.ndarray, frame: pd.DataFrame) -> None:
                df = frame[["name", "issue_date", "target_date", "target"]].copy()
                df["pred"] = pred
                df["model"] = label
                df["fold"] = fold.name
                df["horizon"] = h
                scale = df["name"].map(train_std)
                df["target_cm"] = df["target"] * scale
                df["pred_cm"] = df["pred"] * scale
                out.append(df)

            emit("ridge_own_lags", base_test, test)
            # Gate baseline at h1-3 per the Phase 2 audit: the strongest own-only model
            emit("ridge_own_perbasin", per_basin_ridge_predict(train, test, own_cols), test)

            for gname in graphs:
                nf = nbr_feats[gname]
                nbr_cols = [c for c in nf.columns if c.startswith("nbr")]
                tr = train.merge(nf, on=["issue_date", "name"], how="left").dropna(subset=nbr_cols)
                te = test.merge(nf, on=["issue_date", "name"], how="left").dropna(subset=nbr_cols)
                if len(tr) == 0 or len(te) == 0:
                    continue
                # Primary: neighbor lags enter the pooled ridge directly, same capacity per k
                r_full, s_full = fit_ridge(tr, own_cols + nbr_cols)
                emit(f"nbrridge_{gname}", predict_ridge(r_full, s_full, te, own_cols + nbr_cols), te)
                if include_mlp:
                    # Secondary nonlinearity check: MLP on the backbone residual, neighbors only
                    r_tr = tr["target"].values - predict_ridge(ridge, scaler, tr, own_cols)
                    mlp, ms = fit_residual_mlp(tr, r_tr, nbr_cols, seed=mlp_seed)
                    pred = predict_ridge(ridge, scaler, te, own_cols) + mlp.predict(ms.transform(te[nbr_cols].values))
                    emit(f"nbrmlp_{gname}", pred, te)

    return pd.concat(out, ignore_index=True)
