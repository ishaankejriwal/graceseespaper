"""Which ERA5-Land variables carry the 1-month skill of ridge_own_era5 over ridge_own.

Not used in the manuscript. Ridge heads only, at
a 1-month lead, without placebo arms:

1. Leave one variable out: refit ridge_own_era5 without that variable's three lag
   columns. The skill lost (full minus reduced) is the variable's unique contribution.
2. One variable alone: own state plus that variable's lags.
3. Per-fold skill of ridge_own_era5 over ridge_own.
4. Per-basin DM tests with Benjamini-Hochberg control (q = 0.10), and pooled skill
   by continent.

Items 1 and 2 refit the arms with the same feature construction as the experiment
engine. Items 3 and 4 use the stored results/phase6_era5_predictions.csv rows, so they
decompose exactly the contrast in that file.

Inputs: data/processed/basin_month_twsa_global.csv, basin_meta.csv,
era5_basin_month.csv; results/kalman_fold_params.pkl, phase6_era5_predictions.csv.
Outputs: results/phase6_era5_attribution.csv (one row per arm), _folds.csv,
_continent.csv and _fdr.csv.

Usage:
  python scripts/run_phase6_era5_attribution.py
  GRACEFC_SOURCE=jpl python scripts/run_phase6_era5_attribution.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import era5_fold_features  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold, split_fold  # noqa: E402
from gracefc.experiment_nonlinear import _fit_head  # noqa: E402
from gracefc.kalman import filtered_state_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import per_basin_dm_fdr, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402
from gracefc.runtime import load_era5, load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
H = 1  # lead time in months


def build_pred_rows() -> tuple[pd.DataFrame, list[str]]:
    """Test rows of ridge_own, ridge_own_era5 and the per-variable arms over all folds.

    Returns (rows, ERA5-Land variable names). Arms are labelled
    ridge_own_era5_wo_<var> (variable removed) and ridge_own_era5_only_<var>.
    """
    wide, meta, keep = load_sample(DATA)
    era5_wide = load_era5(keep)
    variables = list(era5_wide)

    cache = load_params_cache(OUT_DIR / "kalman_fold_params.pkl",
                              DATA / "basin_month_twsa_global.csv")
    assert cache, "Kalman parameter cache missing or stale; run flat12_ridge or phase3b first"

    out = []
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        params = cache[fold.name]
        names = list(params["name"])
        filt = filtered_state_wide(resid_wide[names], params)
        rho = params.set_index("name")["rho"][names].values
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names)

        # KF forecast rho^h x_{t|t}; every ridge head fits its error
        prop = filt.values * (rho[None, :] ** H)
        base = pd.DataFrame({
            "issue_date": np.repeat(filt.index.values, len(names)),
            "name": np.tile(names, filt.shape[0]),
            "kalman": prop.ravel(),
            "own_state": filt.values.ravel(),
        })
        tgt = resid_wide[names].shift(-H).stack(future_stack=True).rename("target").reset_index()
        tgt.columns = ["issue_date", "name", "target"]
        base = base.merge(tgt, on=["issue_date", "name"]).dropna(subset=["target", "kalman"])
        base["target_date"] = base["issue_date"] + pd.DateOffset(months=H)
        base = base.merge(era5_feats, on=["issue_date", "name"], how="left").dropna(subset=era5_cols)
        tr, te = split_fold(base, fold)

        ytr = (tr["target"] - tr["kalman"]).values
        own_tr, own_te = tr[["own_state"]].values, te[["own_state"]].values

        def emit(label: str, Xtr: np.ndarray, Xte: np.ndarray) -> None:
            df = te[["name", "issue_date", "target_date", "target"]].copy()
            df["pred"] = te["kalman"].values + _fit_head("ridge", Xtr, ytr, Xte, 0)
            df["model"], df["fold"], df["horizon"] = label, fold.name, H
            out.append(df)

        emit("ridge_own", own_tr, own_te)
        emit("ridge_own_era5",
             np.column_stack([own_tr, tr[era5_cols].values]),
             np.column_stack([own_te, te[era5_cols].values]))
        for var in variables:
            keep_cols = [c for c in era5_cols if not c.startswith(f"{var}_l")]
            solo_cols = [c for c in era5_cols if c.startswith(f"{var}_l")]
            emit(f"ridge_own_era5_wo_{var}",
                 np.column_stack([own_tr, tr[keep_cols].values]),
                 np.column_stack([own_te, te[keep_cols].values]))
            emit(f"ridge_own_era5_only_{var}",
                 np.column_stack([own_tr, tr[solo_cols].values]),
                 np.column_stack([own_te, te[solo_cols].values]))
        print(f"{fold.name} done", flush=True)
    return pd.concat(out, ignore_index=True), variables


def main() -> None:
    pred_rows, variables = build_pred_rows()

    def pooled_skill(a: str, b: str) -> tuple[float, float]:
        stat, p = pooled_monthly_dm(pred_rows, a, b, H)
        ra = rmse(*pred_rows.loc[pred_rows["model"] == a, ["target", "pred"]].values.T)
        rb = rmse(*pred_rows.loc[pred_rows["model"] == b, ["target", "pred"]].values.T)
        return 100 * (1 - (ra / rb) ** 2), p

    full_skill, full_p = pooled_skill("ridge_own_era5", "ridge_own")
    rows = [{"arm": "full", "variable": "", "skill_vs_own_pct": full_skill, "dm_p_vs_own": full_p}]
    for var in variables:
        wo_skill, wo_p = pooled_skill(f"ridge_own_era5_wo_{var}", "ridge_own")
        drop_stat, drop_p = pooled_monthly_dm(pred_rows, "ridge_own_era5", f"ridge_own_era5_wo_{var}", H)
        solo_skill, solo_p = pooled_skill(f"ridge_own_era5_only_{var}", "ridge_own")
        rows.append({
            "arm": "lovo", "variable": var, "skill_vs_own_pct": wo_skill, "dm_p_vs_own": wo_p,
            "unique_contribution_pct": full_skill - wo_skill, "dm_p_full_vs_wo": drop_p,
            "solo_skill_pct": solo_skill, "solo_dm_p": solo_p,
        })
    attribution = pd.DataFrame(rows)
    attribution.to_csv(OUT_DIR / "phase6_era5_attribution.csv", index=False)

    # Per-fold and per-continent breakdowns use the stored predictions file
    pub = pd.read_csv(OUT_DIR / "phase6_era5_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    pub = pub[(pub["horizon"] == H) & (pub["model"].isin(["ridge_own_era5", "ridge_own"]))]
    fold_rows = []
    for fold, grp in pub.groupby("fold"):
        ra = rmse(*grp.loc[grp["model"] == "ridge_own_era5", ["target", "pred"]].values.T)
        rb = rmse(*grp.loc[grp["model"] == "ridge_own", ["target", "pred"]].values.T)
        _, p = pooled_monthly_dm(grp, "ridge_own_era5", "ridge_own", H)
        fold_rows.append({"fold": fold, "skill_pct": 100 * (1 - (ra / rb) ** 2), "dm_p": p,
                          "test_months": grp["target_date"].nunique()})
    folds = pd.DataFrame(fold_rows).sort_values("fold")
    folds.to_csv(OUT_DIR / "phase6_era5_attribution_folds.csv", index=False)

    meta = pd.read_csv(DATA / "basin_meta.csv")[["name", "continent"]]
    pub_c = pub.merge(meta, on="name")
    cont_rows = []
    for cont, grp in pub_c.groupby("continent"):
        ra = rmse(*grp.loc[grp["model"] == "ridge_own_era5", ["target", "pred"]].values.T)
        rb = rmse(*grp.loc[grp["model"] == "ridge_own", ["target", "pred"]].values.T)
        _, p = pooled_monthly_dm(grp, "ridge_own_era5", "ridge_own", H)
        cont_rows.append({"continent": cont, "n_basins": grp["name"].nunique(),
                          "skill_pct": 100 * (1 - (ra / rb) ** 2), "dm_p": p})
    continent = pd.DataFrame(cont_rows).sort_values("skill_pct", ascending=False)
    continent.to_csv(OUT_DIR / "phase6_era5_attribution_continent.csv", index=False)

    fdr = per_basin_dm_fdr(pub, "ridge_own_era5", "ridge_own", H)
    fdr.to_csv(OUT_DIR / "phase6_era5_attribution_fdr.csv", index=False)
    sig = fdr[fdr["significant"]]
    helped, hurt = int(sig["a_better"].sum()), int((~sig["a_better"]).sum())

    print("\n=== attribution (lead 1) ===")
    print(attribution.round(3).to_string(index=False))
    print("\n=== per fold ===")
    print(folds.round(3).to_string(index=False))
    print("\n=== per continent ===")
    print(continent.round(3).to_string(index=False))
    print(f"\nper-basin FDR (q=0.10): {len(sig)} of {fdr.shape[0]} significant "
          f"= {helped} helped / {hurt} hurt")


if __name__ == "__main__":
    main()
