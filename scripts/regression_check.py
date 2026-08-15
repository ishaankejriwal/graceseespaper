"""Phase 1 gate: does the new global pipeline reproduce the prior Africa-37 CSR benchmark?

Prior result (context1): ridge_neighbor_residual_mlp | corr_top2_directed, one-month,
test RMSE 3.2480 cm on the 70/10/20 chronological split. Exact hyperparameters of the
old MLP are unrecorded, so we check for closeness, not equality.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.features import DEFAULT_LAGS, build_lag_table, chronological_split, neighbor_features, pivot_wide  # noqa: E402
from gracefc.graphs import corr_topk, random_degree_matched  # noqa: E402
from gracefc.models import damped_persistence_forecast, fit_residual_mlp, fit_ridge, predict_ridge, rmse  # noqa: E402


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    # Mainland Africa = the continent tag minus Madagascar, mirroring the old 37-basin setup
    afr = meta[(meta["continent"] == "africa") & (meta["name"] != "C_Madagascar")]["name"]
    print(f"mainland africa basins: {afr.size}")
    wide = pivot_wide(long_df[long_df["name"].isin(afr)])

    lag_df = build_lag_table(wide, horizon=1)
    train, val, test = chronological_split(lag_df)
    print(f"split rows: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"test dates: {train['target_date'].max():%Y-%m} | {test['target_date'].min():%Y-%m} to {test['target_date'].max():%Y-%m}")

    own_cols = [f"lag_{k}" for k in DEFAULT_LAGS]

    # Persistence and ridge AR floor
    print(f"\npersistence            RMSE {rmse(test['target'].values, test['lag_0'].values):.4f}")
    print(f"damped persistence     RMSE {rmse(test['target'].values, damped_persistence_forecast(train, test, 1)):.4f}")
    ridge, scaler = fit_ridge(train, own_cols)
    print(f"ridge own-lags         RMSE {rmse(test['target'].values, predict_ridge(ridge, scaler, test, own_cols)):.4f}")

    # corr_top2_directed graph built on the training window only
    train_end = train["issue_date"].max()
    graph = corr_topk(wide[wide.index <= train_end], k=2)
    nbr = neighbor_features(wide, graph)
    nbr_cols = [c for c in nbr.columns if c.startswith("nbr")]

    def run_variant(g, label, seed=0):
        nf = neighbor_features(wide, g)
        merged = {split_name: split.merge(nf, on=["issue_date", "name"], how="left").dropna(subset=nbr_cols)
                  for split_name, split in (("train", train), ("test", test))}
        tr, te = merged["train"], merged["test"]
        r, rs = fit_ridge(tr, own_cols)
        resid = tr["target"].values - predict_ridge(r, rs, tr, own_cols)
        mlp, ms = fit_residual_mlp(tr, resid, nbr_cols, seed=seed)
        pred = predict_ridge(r, rs, te, own_cols) + mlp.predict(ms.transform(te[nbr_cols].values))
        print(f"{label:22s} RMSE {rmse(te['target'].values, pred):.4f}  (n_test={len(te)})")

    run_variant(graph, "ridge+nbrMLP corr_top2")
    for seed in (1, 2, 3):
        run_variant(random_degree_matched(graph, seed), f"ridge+nbrMLP random s{seed}", seed=0)


if __name__ == "__main__":
    main()
