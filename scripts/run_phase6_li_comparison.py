"""Phase 6: head-to-head against the Li & Kusche (2026) LSTM hindcast (CSR-FCast).

Places their basin-aggregated forecasts into our evaluation space: subtract OUR
fold-specific climatology (full variant) and a train-window-only mean offset that
absorbs baseline-epoch and decomposition-convention differences, then standardize by
our train std. Scoring is restricted to the matched sample — identical (basin,
target_date) rows for every model at each horizon — so all skill numbers are
directly comparable.

Two Li variants:
  li_lstm_full     their operational product (LSTM non-seasonal + extrapolated
                   seasonal/trend), evaluated as a forecast of our deseasonalized target
  li_lstm_nonseas  their LSTM component alone (interannual + sub-seasonal)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.decompose import fit_climatology  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, diebold_mariano, per_basin_dm_fdr, pooled_monthly_dm  # noqa: E402

OUT_DIR = ROOT / "results"
HORIZONS = range(1, 7)
OUR_MODELS = {
    # file -> models to pull (phase3b is already in std units; phase2 has *_std_units cols)
    "phase3b": ["kalman_corr_top1", "kalman_ar1", "kalman_own_ridge"],
    "phase2": ["ridge_own_perbasin", "damped_persistence_rho", "persistence", "climatology_zero"],
}
PAIRS = [
    ("li_lstm_full", "kalman_corr_top1"),
    ("li_lstm_full", "kalman_ar1"),
    ("li_lstm_full", "ridge_own_perbasin"),
    ("li_lstm_full", "damped_persistence_rho"),
    ("li_lstm_nonseas", "kalman_corr_top1"),
    ("li_lstm_nonseas", "kalman_ar1"),
    ("li_lstm_nonseas", "ridge_own_perbasin"),
    ("li_lstm_nonseas", "damped_persistence_rho"),
]
MIN_TRAIN_OFFSET_MONTHS = 24


def build_li_pred_rows(li: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Li forecasts -> our standardized deseasonalized space, per fold, train-only transforms."""
    out = []
    for fold in DEFAULT_FOLDS:
        train_wide = wide[wide.index < fold.test_start]
        for name in wide.columns:
            sub = li[li["name"] == name]
            if not len(sub):
                continue
            fit = fit_climatology(train_wide[name])
            resid_obs = wide[name] - fit.predict(wide.index)
            train_std = resid_obs[resid_obs.index < fold.test_start].std()
            clim = pd.Series(fit.predict(pd.DatetimeIndex(sub["target_date"].unique())))
            clim.index = pd.DatetimeIndex(sub["target_date"].unique())
            for h in HORIZONS:
                sh = sub[sub["horizon"] == h]
                resid_full = sh["li_full_cm"].values - clim[sh["target_date"]].values
                resid_ns = sh["li_nonseasonal_cm"].values
                obs = resid_obs.reindex(sh["target_date"]).values
                # Train mask is about what was OBSERVABLE before the freeze (target seen);
                # test membership is by ISSUE date, matching evaluate.split_fold
                is_train = (sh["target_date"] < fold.test_start).values
                is_test = ((sh["issue_date"] >= fold.test_start)
                           & (sh["issue_date"] <= fold.test_end)).values
                for model, pred_resid in (("li_lstm_full", resid_full), ("li_lstm_nonseas", resid_ns)):
                    # Offset on train targets only: absorbs baseline-epoch and
                    # decomposition-convention differences, never test information
                    ok = is_train & np.isfinite(pred_resid) & np.isfinite(obs)
                    if ok.sum() < MIN_TRAIN_OFFSET_MONTHS:
                        continue
                    offset = float((pred_resid[ok] - obs[ok]).mean())
                    keep = is_test & np.isfinite(pred_resid) & np.isfinite(obs)
                    if not keep.any():
                        continue
                    df = sh.loc[keep, ["name", "issue_date", "target_date"]].copy()
                    df["target"] = obs[keep] / train_std
                    df["pred"] = (pred_resid[keep] - offset) / train_std
                    df["model"] = model
                    df["fold"] = fold.name
                    df["horizon"] = h
                    out.append(df)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    li = pd.read_csv(ROOT / "data/processed/li2026_csr_basin_forecasts.csv",
                     parse_dates=["issue_date", "target_date"])
    coverage = pd.read_csv(ROOT / "data/processed/li2026_basin_coverage.csv")

    li_rows = build_li_pred_rows(li, wide)
    print(f"li rows: {len(li_rows)} | basins: {li_rows['name'].nunique()}")

    # Our models, all converted to standardized units
    p3b = pd.read_csv(OUT_DIR / "phase3b_predictions.csv", parse_dates=["issue_date", "target_date"])
    p3b = p3b[p3b["model"].isin(OUR_MODELS["phase3b"])]
    p2 = pd.read_csv(OUT_DIR / "phase2_baseline_predictions.csv", parse_dates=["issue_date", "target_date"])
    p2 = p2[p2["model"].isin(OUR_MODELS["phase2"])]
    p2 = p2.drop(columns=["target", "pred"]).rename(
        columns={"target_std_units": "target", "pred_std_units": "pred"})
    ours = pd.concat([p3b[li_rows.columns], p2[li_rows.columns]], ignore_index=True)

    # Consistency guard: our recomputed standardized target must match the stored ones
    chk = li_rows.merge(ours[ours["model"] == "kalman_ar1"],
                        on=["name", "target_date", "horizon"], suffixes=("_li", "_k"))
    gap = (chk["target_li"] - chk["target_k"]).abs().max()
    if not (gap < 1e-6):
        raise AssertionError(f"target mismatch vs phase3b: max |diff| = {gap}")

    # Matched sample per horizon: rows present for every model
    all_rows = pd.concat([li_rows, ours], ignore_index=True)
    n_models = all_rows["model"].nunique()
    counts = all_rows.groupby(["horizon", "name", "target_date"])["model"].nunique()
    matched_keys = counts[counts == n_models].reset_index()[["horizon", "name", "target_date"]]
    matched = all_rows.merge(matched_keys, on=["horizon", "name", "target_date"])
    matched = matched.merge(coverage, on="name")
    matched.to_csv(OUT_DIR / "phase6_li_comparison_predictions.csv", index=False)

    subsets = {"all_matched": matched, "coverage_ge_0.5": matched[matched["li_coverage"] >= 0.5]}
    summary_rows, headline_rows = [], []
    for label, sub in subsets.items():
        for (model, h), grp in sub.groupby(["model", "horizon"]):
            summary_rows.append({
                "subset": label, "model": model, "horizon": h,
                "rmse_std": rmse(grp["target"].values, grp["pred"].values),
                "n": len(grp), "n_months": grp["target_date"].nunique(),
                "n_basins": grp["name"].nunique(),
            })
        for model_a, model_b in PAIRS:
            for h in HORIZONS:
                s = sub[sub["horizon"] == h]
                if not len(s):
                    continue
                skill, lo, hi = block_bootstrap_skill_ci(s, model_a, model_b, h)
                dm, p = pooled_monthly_dm(s, model_a, model_b, h)
                headline_rows.append({
                    "subset": label, "model": model_a, "vs": model_b, "horizon": h,
                    "n_rows": (s["model"] == model_a).sum(),
                    "n_months": s[s["model"] == model_a]["target_date"].nunique(),
                    "skill": skill, "ci_lo": lo, "ci_hi": hi, "dm_stat": dm, "dm_p": p,
                })

    summary = pd.DataFrame(summary_rows)
    # Skill vs damped persistence on the SAME matched rows, the ladder's reference point
    damped = summary[summary["model"] == "damped_persistence_rho"].set_index(["subset", "horizon"])["rmse_std"]
    summary["skill_vs_damped"] = 1 - (summary["rmse_std"] / summary.set_index(["subset", "horizon"]).index.map(damped)) ** 2
    summary = summary.sort_values(["subset", "horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "phase6_li_comparison_summary.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(OUT_DIR / "phase6_li_comparison_headline.csv", index=False)

    # Per-basin: DM per basin with FDR, both key pairs, every horizon
    pb_rows = []
    for model_a, model_b in [("li_lstm_full", "kalman_corr_top1"), ("li_lstm_nonseas", "kalman_corr_top1")]:
        for h in HORIZONS:
            s = matched[matched["horizon"] == h]
            if not len(s):
                continue
            df = per_basin_dm_fdr(s, model_a, model_b, h)
            df["model"], df["vs"], df["horizon"] = model_a, model_b, h
            pb_rows.append(df)
    pd.concat(pb_rows, ignore_index=True).to_csv(OUT_DIR / "phase6_li_comparison_perbasin.csv", index=False)

    for label in subsets:
        print(f"\n===== {label} =====")
        print(summary[summary["subset"] == label].to_string(index=False))
    hl = pd.DataFrame(headline_rows)
    print("\n===== headline (all_matched) =====")
    print(hl[hl["subset"] == "all_matched"].to_string(index=False))


if __name__ == "__main__":
    main()
