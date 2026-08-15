"""Phase 6 basin analysis: which basins benefit from neighbors, and can we predict it?

Builds a per-basin table joining neighbor skill (phase3b kalman backbone, phase4
index-conditioned, phase6 ERA5-conditioned ridge backbone) with static covariates
(continent, SNR stratum, area, latitude, ERA5 coverage, neighbor distance/correlation,
Kalman signal-to-noise, coupling), then answers:
  1. what predicts benefit (Spearman + random-forest permutation importance)
  2. whether selective application beats apply-everywhere (fold-honest selection)
  3. geographic patterns
  4. whether ERA5 conditioning changes who benefits
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.graphs import corr_topk, distance_matrix_km  # noqa: E402
from gracefc.stats import diebold_mariano  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402

OUT = ROOT / "results"
RNG = 0


# ---------------------------------------------------------------- covariates

def build_neighbor_table() -> pd.DataFrame:
    """Reconstruct the fold-specific corr_top1 graphs exactly as the experiments built them."""
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    meta_kept = meta[meta["name"].isin(wide.columns)].reset_index(drop=True)
    dist = distance_matrix_km(meta_kept)
    cache = load_params_cache(OUT / "kalman_fold_params.pkl", ROOT / "data/processed/basin_month_twsa_global.csv")
    assert cache, "params cache missing or stale for current data/protocol - run phase3b first"

    rows = []
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        names = list(cache[fold.name]["name"])
        train_src = resid_wide[resid_wide.index < fold.test_start][names]
        graph = corr_topk(train_src, 1)
        corr = train_src.corr()
        for n, nbrs in graph.items():
            if not nbrs:
                continue
            nb = nbrs[0]
            rows.append({"name": n, "fold": fold.name, "neighbor": nb,
                         "nbr_corr": corr.loc[n, nb], "nbr_dist_km": dist.loc[n, nb]})
    per_fold = pd.DataFrame(rows)
    agg = per_fold.groupby("name").agg(
        nbr_corr=("nbr_corr", "mean"),
        nbr_dist_km=("nbr_dist_km", "mean"),
        nbr_n_unique=("neighbor", "nunique"),
        nbr_modal=("neighbor", lambda s: s.mode().iloc[0]),
    ).reset_index()
    agg["nbr_stable"] = (agg["nbr_n_unique"] == 1).astype(int)
    return agg, meta_kept, cache


def build_covariates() -> pd.DataFrame:
    nbr, meta_kept, cache = build_neighbor_table()
    cov = meta_kept[["name", "continent", "centroid_lat", "centroid_lon",
                     "area_km2", "glaciated"]].copy()
    cov["abs_lat"] = cov["centroid_lat"].abs()
    cov["log10_area"] = np.log10(cov["area_km2"])
    cov["glaciated"] = cov["glaciated"].astype(int)

    strata = pd.read_csv(OUT / "phase2_strata.csv", index_col=0).rename_axis("name").reset_index()
    cov = cov.merge(strata, on="name", how="left")
    era5 = pd.read_csv(ROOT / "data/processed/era5_basin_coverage.csv")
    cov = cov.merge(era5, on="name", how="left")
    cov = cov.merge(nbr, on="name", how="left")

    # Kalman signal-to-noise q/r, fold-averaged in log space; r floors at ~1e-8 so clip first
    snr = pd.concat(cache.values())
    snr["log_qr"] = np.log10(snr["q"].clip(1e-8) / snr["r"].clip(1e-8))
    kal = snr.groupby("name").agg(rho=("rho", "mean"), log_qr=("log_qr", "mean")).reset_index()
    cov = cov.merge(kal, on="name", how="left")

    coup = pd.read_csv(OUT / "phase5_coupled_coupling.csv")
    cov = cov.merge(coup.groupby("name")["coupling"].mean().reset_index(), on="name", how="left")
    return cov


# ------------------------------------------------------------ per-basin skill

def perbasin_skill(pred_csv: str, challenger: str, reference: str, horizon: int,
                   label: str) -> pd.DataFrame:
    """Per-basin pooled skill and DM stat for challenger vs reference at one horizon."""
    df = pd.read_csv(OUT / pred_csv)
    df = df[(df["horizon"] == horizon) & (df["model"].isin([challenger, reference]))]
    df["loss"] = (df["target"] - df["pred"]) ** 2
    # Matched rows only: identical (name, target_date) sets guard against row-set drift
    piv = df.pivot_table(index=["name", "target_date"], columns="model", values="loss")
    piv = piv.dropna()
    rows = []
    for n, grp in piv.groupby(level="name"):
        la, lb = grp[challenger].values, grp[reference].values
        stat, p = diebold_mariano(la, lb, horizon=horizon)
        rows.append({"name": n,
                     f"skill_{label}": 1 - la.mean() / lb.mean(),
                     f"dm_{label}": stat, f"dm_p_{label}": p,
                     f"n_{label}": len(grp)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- selective application

def selective_skill(pred_csv: str, challenger: str, reference: str, horizon: int,
                    selections: dict[str, dict[str, set]]) -> pd.DataFrame:
    """Pooled skill when the challenger is applied only to selected basins per fold.

    selections: strategy -> fold -> set of basin names that get the challenger.
    Reference rows fill in everywhere else, so every strategy scores the same row set.
    """
    df = pd.read_csv(OUT / pred_csv)
    df = df[(df["horizon"] == horizon) & (df["model"].isin([challenger, reference]))]
    df["loss"] = (df["target"] - df["pred"]) ** 2
    piv = df.pivot_table(index=["name", "target_date", "fold"], columns="model",
                        values="loss").dropna().reset_index()
    ref_mse = piv[reference].mean()
    out = []
    for strat, per_fold in selections.items():
        sel = np.array([r["name"] in per_fold.get(r["fold"], set())
                        for _, r in piv.iterrows()])
        loss = np.where(sel, piv[challenger], piv[reference])
        monthly = pd.DataFrame({"target_date": piv["target_date"], "loss": loss,
                                "ref": piv[reference]}).groupby("target_date").mean()
        stat, p = diebold_mariano(monthly["loss"].values, monthly["ref"].values,
                                  horizon=horizon)
        out.append({"strategy": strat, "skill_pct": 100 * (1 - loss.mean() / ref_mse),
                    "n_selected_mean": np.mean([len(per_fold.get(f.name, set()))
                                                for f in DEFAULT_FOLDS]),
                    "dm_stat": stat, "dm_p": p})
    return pd.DataFrame(out)


def fold_honest_selections(pred_csv: str, challenger: str, reference: str,
                           horizon: int, cov: pd.DataFrame) -> dict:
    """Build per-fold basin selections: oracle (in-sample), leave-one-fold-out empirical,
    and covariate-RF predicted benefit (double cross-validated)."""
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(OUT / pred_csv)
    df = df[(df["horizon"] == horizon) & (df["model"].isin([challenger, reference]))]
    df["loss"] = (df["target"] - df["pred"]) ** 2
    piv = df.pivot_table(index=["name", "fold"], columns="model", values="loss",
                        aggfunc="mean").dropna().reset_index()
    folds = [f.name for f in DEFAULT_FOLDS if f.name in set(piv["fold"])]
    all_names = sorted(piv["name"].unique())

    # Oracle: full-sample per-basin benefit (upper bound, in-sample by construction)
    full = piv.groupby("name")[[challenger, reference]].mean()
    oracle = set(full[full[challenger] < full[reference]].index)

    X = feature_matrix(cov.set_index("name").loc[all_names])
    sel = {"all": {}, "oracle": {}, "lofo_empirical": {}, "rf_covariates": {}}
    rng = np.random.default_rng(RNG)
    groups = {n: i % 5 for i, n in enumerate(rng.permutation(all_names))}
    for f in folds:
        sel["all"][f] = set(all_names)
        sel["oracle"][f] = oracle
        other = piv[piv["fold"] != f].groupby("name")[[challenger, reference]].mean()
        lofo_benefit = other[challenger] < other[reference]
        sel["lofo_empirical"][f] = set(other[lofo_benefit].index)
        # RF trained on out-of-fold labels AND out-of-group basins: honest in time and space
        y = lofo_benefit.reindex(all_names).fillna(False).astype(int).values
        picked = set()
        for g in range(5):
            te_mask = np.array([groups[n] == g for n in all_names])
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                         random_state=RNG).fit(X[~te_mask], y[~te_mask])
            prob = clf.predict_proba(X[te_mask])[:, list(clf.classes_).index(1)] \
                if len(clf.classes_) > 1 else np.ones(te_mask.sum())
            picked |= {n for n, p in zip(np.array(all_names)[te_mask], prob) if p > 0.5}
        sel["rf_covariates"][f] = picked
    return sel


# ------------------------------------------------------------------ modelling

NUM_FEATS = ["abs_lat", "centroid_lat", "log10_area", "era5_coverage",
             "nbr_corr", "nbr_dist_km", "nbr_stable", "rho", "log_qr",
             "coupling", "glaciated"]


def feature_matrix(cov: pd.DataFrame) -> np.ndarray:
    X = cov[NUM_FEATS].copy()
    X = pd.concat([X, pd.get_dummies(cov["continent"], prefix="cont"),
                   pd.get_dummies(cov["stratum"], prefix="snr")], axis=1)
    return X.fillna(X.median()).astype(float).values


def rf_importance(cov: pd.DataFrame, target: pd.Series, label: str) -> pd.DataFrame:
    """Out-of-sample R2 and permutation importance for predicting per-basin benefit."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import KFold, cross_val_score

    ok = target.notna()
    Xdf = cov.loc[ok.values, NUM_FEATS].copy()
    Xdf = pd.concat([Xdf, pd.get_dummies(cov.loc[ok.values, "continent"], prefix="cont"),
                     pd.get_dummies(cov.loc[ok.values, "stratum"], prefix="snr")], axis=1)
    cols = list(Xdf.columns)
    X = Xdf.fillna(Xdf.median()).astype(float).values
    y = target[ok].values
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=RNG)
    cv_r2 = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=RNG), scoring="r2")
    rf.fit(X, y)
    imp = permutation_importance(rf, X, y, n_repeats=20, random_state=RNG)
    return (pd.DataFrame({"feature": cols, "importance": imp.importances_mean,
                          "importance_std": imp.importances_std,
                          "target": label, "cv_r2_mean": cv_r2.mean()})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def spearman_table(cov: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats as sps
    rows = []
    for tcol in [c for c in targets.columns if c.startswith(("skill_", "dm_")) and not c.startswith("dm_p_")]:
        for feat in NUM_FEATS:
            sub = pd.concat([cov.set_index("name")[feat],
                             targets.set_index("name")[tcol]], axis=1).dropna()
            r, p = sps.spearmanr(sub[feat], sub[tcol])
            rows.append({"target": tcol, "feature": feat, "spearman_r": r, "p": p, "n": len(sub)})
    return pd.DataFrame(rows)


def strat_table(merged: pd.DataFrame, by: str, skill_col: str, dm_col: str) -> pd.DataFrame:
    g = merged.groupby(by, observed=True)
    tab = pd.DataFrame({
        "n": g.size(),
        "median_skill_pct": g[skill_col].median() * 100,
        "frac_helped": g.apply(lambda d: (d[skill_col] > 0).mean(), include_groups=False),
        "frac_dm_sig_helped": g.apply(lambda d: ((d[dm_col] < 0) & (d[f"{dm_col.replace('dm_', 'dm_p_')}"] < .05)).mean(), include_groups=False),
        "frac_dm_sig_hurt": g.apply(lambda d: ((d[dm_col] > 0) & (d[f"{dm_col.replace('dm_', 'dm_p_')}"] < .05)).mean(), include_groups=False),
    })
    return tab.sort_values("median_skill_pct", ascending=False).round(3)


# ----------------------------------------------------------------------- main

def main() -> None:
    cov = build_covariates()
    print(f"covariates: {len(cov)} basins")

    # Per-basin h1 neighbor skill in the three settings + unconditioned ridge backbone
    settings = {
        "p3b": ("phase3b_predictions.csv", "kalman_corr_top1", "kalman_own_ridge"),
        "p4cond": ("phase4_conditioned_predictions.csv", "kalman_corr_top1", "kalman_own_ridge"),
        "era5r": ("phase6_era5_predictions.csv", "ridge_corr_top1", "ridge_own"),
        "era5c": ("phase6_era5_predictions.csv", "ridge_corr_top1_era5", "ridge_own_era5"),
    }
    skills = None
    for label, (csv, ch, ref) in settings.items():
        t = perbasin_skill(csv, ch, ref, horizon=1, label=label)
        print(f"{label}: {len(t)} basins, median skill {t[f'skill_{label}'].median()*100:+.2f}%")
        skills = t if skills is None else skills.merge(t, on="name", how="outer")

    # Li long-lead edge as a covariate: does exogenous-forcing benefit align with neighbor benefit?
    li = pd.read_csv(OUT / "phase6_li_comparison_perbasin.csv")
    li4 = (li[(li["model"] == "li_lstm_full") & (li["horizon"] == 4)]
           [["name", "dm_stat"]].rename(columns={"dm_stat": "li_dm_h4"}))
    merged = cov.merge(skills, on="name", how="inner").merge(li4, on="name", how="left")

    # Q1: correlations and RF importance against the primary (phase3b) benefit measures
    spear = spearman_table(cov, skills)
    spear.to_csv(OUT / "phase6_basin_analysis_spearman.csv", index=False)
    imps = pd.concat([
        rf_importance(cov, skills.set_index("name")["dm_p3b"].reindex(cov["name"]).reset_index(drop=True), "dm_p3b"),
        rf_importance(cov, skills.set_index("name")["skill_p3b"].clip(-.5, .5).reindex(cov["name"]).reset_index(drop=True), "skill_p3b_clipped"),
    ])
    imps.to_csv(OUT / "phase6_basin_analysis_rf_importance.csv", index=False)

    # Q3: stratified tables
    tabs = {}
    for by in ["continent", "stratum"]:
        tabs[by] = strat_table(merged, by, "skill_p3b", "dm_p3b")
    merged["dist_band"] = pd.cut(merged["nbr_dist_km"], [0, 300, 600, 1000, 1e9],
                                 labels=["<300km", "300-600", "600-1000", ">1000"])
    merged["corr_band"] = pd.qcut(merged["nbr_corr"], 3, labels=["low_corr", "mid_corr", "high_corr"])
    merged["lat_band"] = pd.cut(merged["centroid_lat"], [-90, -23.5, 23.5, 50, 90],
                                labels=["S_extratropics", "tropics", "N_midlat", "N_high"])
    for by in ["dist_band", "corr_band", "lat_band"]:
        tabs[by] = strat_table(merged, by, "skill_p3b", "dm_p3b")
    for k, t in tabs.items():
        print(f"\n=== {k} ===\n{t}")

    # Q2: selective application, fold-honest
    sel = fold_honest_selections("phase3b_predictions.csv", "kalman_corr_top1",
                                 "kalman_own_ridge", 1, cov)
    selective = selective_skill("phase3b_predictions.csv", "kalman_corr_top1",
                                "kalman_own_ridge", 1, sel)
    # Same question on the ERA5-conditioned backbone
    sel_e = fold_honest_selections("phase6_era5_predictions.csv", "ridge_corr_top1_era5",
                                   "ridge_own_era5", 1, cov)
    selective_e = selective_skill("phase6_era5_predictions.csv", "ridge_corr_top1_era5",
                                  "ridge_own_era5", 1, sel_e)
    selective["setting"], selective_e["setting"] = "p3b_kalman", "era5_ridge"
    selective_all = pd.concat([selective, selective_e], ignore_index=True)
    selective_all.to_csv(OUT / "phase6_basin_analysis_selective.csv", index=False)
    print(f"\n=== selective application (h1) ===\n{selective_all.round(4)}")

    # Q4: does ERA5 conditioning move who benefits?
    from scipy import stats as sps
    q4 = {}
    for a, b in [("dm_p3b", "dm_era5c"), ("dm_era5r", "dm_era5c"), ("skill_era5r", "skill_era5c")]:
        sub = skills[[a, b]].dropna()
        q4[f"spearman_{a}_vs_{b}"] = sps.spearmanr(sub[a], sub[b])[0]
    both = skills[["skill_era5r", "skill_era5c"]].dropna()
    q4["frac_sign_flip_era5"] = ((both["skill_era5r"] > 0) != (both["skill_era5c"] > 0)).mean()
    # Replication check: the no-ERA5 ridge arms in phase6 should reproduce phase3b exactly
    rep = skills[["dm_p3b", "dm_era5r"]].dropna()
    q4["max_abs_diff_dm_p3b_vs_era5r"] = (rep["dm_p3b"] - rep["dm_era5r"]).abs().max()
    for lab in ["p3b", "era5r", "era5c"]:
        sig = (skills[f"dm_{lab}"] < 0) & (skills[f"dm_p_{lab}"] < .05)
        q4[f"n_sig_helped_{lab}"] = int(sig.sum())
    sig_r = set(skills.loc[(skills["dm_era5r"] < 0) & (skills["dm_p_era5r"] < .05), "name"])
    sig_c = set(skills.loc[(skills["dm_era5c"] < 0) & (skills["dm_p_era5c"] < .05), "name"])
    q4["n_sig_helped_overlap_era5"] = len(sig_r & sig_c)
    sub = pd.concat([skills.set_index("name")["dm_p3b"], li4.set_index("name")["li_dm_h4"]], axis=1).dropna()
    q4["spearman_dm_p3b_vs_li_h4"] = sps.spearmanr(sub["dm_p3b"], sub["li_dm_h4"])[0]
    print(f"\n=== Q4 ===\n{pd.Series(q4).round(3)}")

    merged.to_csv(OUT / "phase6_basin_analysis_summary.csv", index=False)
    pd.Series(q4).to_csv(OUT / "phase6_basin_analysis_q4.csv")
    print("\nsaved: phase6_basin_analysis_summary.csv + spearman/rf_importance/selective/q4 CSVs")


if __name__ == "__main__":
    main()
