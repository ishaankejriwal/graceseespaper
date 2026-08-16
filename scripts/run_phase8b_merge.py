"""Phase 8b: merge the h1-3 and h4-6 stacked-LSTM runs and score h1-6 against Li & Kusche.

Part 1 concatenates the phase 8 (h1-3) and phase 8b (h4-6) summary/headline tables into
one h1-6 comparison table. Part 2 restricts our arms (per-seed and 2-seed ensemble) and
the Li variants to a matched sample — identical (basin, target_date) rows for every model
at each horizon, exactly the phase 6 protocol — and asks the phase 6 crossing question
with the new stack: Li won h3-6 against the Kalman backbone; does lstmres close that gap?

All Li rows come from phase6_li_comparison_predictions.csv, already transformed into our
standardized deseasonalized space with train-only offsets (see run_phase6_li_comparison).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, per_basin_dm_fdr, pooled_monthly_dm  # noqa: E402

OUT = ROOT / "results"
TAG_H13 = "phase8_lstm_combined"
TAG_H46 = "phase8b_lstm_h46"
HORIZONS = range(1, 7)
OUR_MODELS = ["kalman_ar1", "ridge_own", "ridge_own_era5", "ridge_corr_top1_era5",
              "lstm_own_era5_s0", "lstm_own_era5_s1",
              "lstmres_corr_top1_s0", "lstmres_corr_top1_s1"]
ENSEMBLES = ["lstm_own_era5", "lstmres_corr_top1"]
LI_MODELS = ["li_lstm_full", "li_lstm_nonseas", "damped_persistence_rho"]
KEY = ["name", "target_date", "horizon"]
PAIRS = [
    ("lstmres_corr_top1_ens", "li_lstm_full"),
    ("lstmres_corr_top1_s0", "li_lstm_full"),
    ("lstmres_corr_top1_s1", "li_lstm_full"),
    ("lstm_own_era5_ens", "li_lstm_full"),
    ("ridge_corr_top1_era5", "li_lstm_full"),
    ("kalman_ar1", "li_lstm_full"),
    ("lstmres_corr_top1_ens", "li_lstm_nonseas"),
    ("lstmres_corr_top1_ens", "damped_persistence_rho"),
    ("li_lstm_full", "damped_persistence_rho"),
]


def add_ensembles(ours: pd.DataFrame, fams: list[str] | None = None) -> pd.DataFrame:
    # fams defaults to the Li-comparison set; widening it there would change n_models and
    # therefore the matched sample, so extra families are only requested for Part 1b
    parts = [ours]
    for fam in (ENSEMBLES if fams is None else fams):
        s0 = ours[ours["model"] == f"{fam}_s0"]
        s1 = ours[ours["model"] == f"{fam}_s1"][KEY + ["pred"]]
        ens = s0.merge(s1, on=KEY, suffixes=("", "_s1"))
        if len(ens) != len(s0):
            raise AssertionError(f"{fam}: seed row sets differ ({len(ens)} vs {len(s0)})")
        ens["pred"] = (ens["pred"] + ens["pred_s1"]) / 2
        ens = ens.drop(columns=["pred_s1"])
        ens["model"] = f"{fam}_ens"
        parts.append(ens)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    # ---- Part 1: h1-6 merged tables ----
    for piece in ("summary", "headline"):
        merged = pd.concat([pd.read_csv(OUT / f"{TAG_H13}_{piece}.csv"),
                            pd.read_csv(OUT / f"{TAG_H46}_{piece}.csv")], ignore_index=True)
        sort_cols = (["horizon", "rmse_std"] if piece == "summary"
                     else ["challenger", "reference", "horizon"])
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
        merged.to_csv(OUT / f"phase8b_h16_{piece}.csv", index=False)

    # ---- Part 1b: two-seed ensemble within-backbone contrasts ----
    # The reporting rule (audit P0-6) makes the ensemble the headline quantity, but only
    # per-seed contrasts were being shipped (phase 8 audit, 2026-08-15).
    full = pd.concat([pd.read_csv(OUT / f"{t}_predictions.csv", parse_dates=["issue_date", "target_date"])
                      for t in (TAG_H13, TAG_H46)], ignore_index=True)
    # hist12 and oof were added in the 2026-08-15 repair; they need ensemble rows too,
    # because Sect. results_delivery reports two-seed ensembles and the representation
    # and out-of-fold contrasts have to be quoted on the same footing as the headline.
    ens_fams = ["lstmres_corr_top1", "lstm_own_era5", "lstmres_own", "lstm_corr_top1_era5",
                "lstmres_nbrin_corr_top1", "lstmres_corr_top1_hist12",
                "lstmres_oof_corr_top1", "lstmres_oof_own"]
    ens_rows = add_ensembles(
        full[full["model"].str.rsplit("_s", n=1).str[0].isin(ens_fams)], fams=ens_fams)
    ens_only = ens_rows[ens_rows["model"].str.endswith("_ens")]
    ens_pred = pd.concat([ens_only, full[full["model"] == "ridge_corr_top1_era5"]], ignore_index=True)
    ens_contrasts = [
        ("lstmres_corr_top1_ens", "lstm_own_era5_ens"),
        ("lstmres_corr_top1_ens", "lstmres_own_ens"),
        ("lstmres_own_ens", "lstm_own_era5_ens"),
        ("lstmres_corr_top1_ens", "ridge_corr_top1_era5"),
        ("lstm_corr_top1_era5_ens", "lstm_own_era5_ens"),
        # Representation held against delivery: same correction stage, the input arm's
        # 12-month history instead of the propagated scalar
        ("lstmres_corr_top1_hist12_ens", "lstm_own_era5_ens"),
        ("lstmres_corr_top1_hist12_ens", "lstmres_corr_top1_ens"),
        # Out-of-fold stage-2 residuals: does the in-sample-residual caveat bite?
        ("lstmres_oof_corr_top1_ens", "lstm_own_era5_ens"),
        ("lstmres_oof_corr_top1_ens", "lstmres_corr_top1_ens"),
        ("lstmres_oof_own_ens", "lstm_own_era5_ens"),
    ]
    present = set(ens_pred["model"])
    rows = []
    for a, b in ens_contrasts:
        if a not in present or b not in present:
            continue
        for h in sorted(ens_pred["horizon"].unique()):
            point, lo, hi = block_bootstrap_skill_ci(ens_pred, a, b, h)
            stat, p = pooled_monthly_dm(ens_pred, a, b, h)
            rows.append({"challenger": a, "reference": b, "horizon": h,
                         "skill_pct": 100 * point, "ci_lo_pct": 100 * lo,
                         "ci_hi_pct": 100 * hi, "dm_stat": stat, "dm_p": p})
    pd.DataFrame(rows).to_csv(OUT / "phase8b_h16_ensemble_headline.csv", index=False)

    # ---- Part 2: matched-sample comparison against Li ----
    ours = pd.concat([pd.read_csv(OUT / f"{t}_predictions.csv", parse_dates=["issue_date", "target_date"])
                      for t in (TAG_H13, TAG_H46)], ignore_index=True)
    ours = add_ensembles(ours[ours["model"].isin(OUR_MODELS)])
    li = pd.read_csv(OUT / "phase6_li_comparison_predictions.csv",
                     parse_dates=["issue_date", "target_date"])
    li = li[li["model"].isin(LI_MODELS)].drop(columns=["li_coverage"])

    all_rows = pd.concat([ours, li[ours.columns]], ignore_index=True)
    n_models = all_rows["model"].nunique()
    counts = all_rows.groupby(KEY)["model"].nunique()
    matched_keys = counts[counts == n_models].reset_index()[KEY]
    matched = all_rows.merge(matched_keys, on=KEY)

    # Guard: our standardized target must equal the one stored with the Li rows
    chk = (matched[matched["model"] == "kalman_ar1"][KEY + ["target"]]
           .merge(matched[matched["model"] == "li_lstm_full"][KEY + ["target"]],
                  on=KEY, suffixes=("_ours", "_li")))
    gap = (chk["target_ours"] - chk["target_li"]).abs().max()
    if not (gap < 1e-6):
        raise AssertionError(f"target mismatch vs Li rows: max |diff| = {gap}")

    coverage = pd.read_csv(ROOT / "data/processed/li2026_basin_coverage.csv")
    matched = matched.merge(coverage, on="name")
    matched.to_csv(OUT / "phase8b_li_comparison_predictions.csv", index=False)

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
                    "n_rows": int((s["model"] == model_a).sum()),
                    "n_months": s[s["model"] == model_a]["target_date"].nunique(),
                    "skill": skill, "ci_lo": lo, "ci_hi": hi, "dm_stat": dm, "dm_p": p,
                })

    summary = pd.DataFrame(summary_rows)
    damped = summary[summary["model"] == "damped_persistence_rho"].set_index(["subset", "horizon"])["rmse_std"]
    summary["skill_vs_damped"] = 1 - (summary["rmse_std"] / summary.set_index(["subset", "horizon"]).index.map(damped)) ** 2
    summary = summary.sort_values(["subset", "horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT / "phase8b_li_comparison_summary.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(OUT / "phase8b_li_comparison_headline.csv", index=False)

    # Per-basin: who wins where, ensemble vs Li full product
    pb_rows = []
    for h in HORIZONS:
        s = matched[matched["horizon"] == h]
        df = per_basin_dm_fdr(s, "lstmres_corr_top1_ens", "li_lstm_full", h)
        df["model"], df["vs"], df["horizon"] = "lstmres_corr_top1_ens", "li_lstm_full", h
        pb_rows.append(df)
    perbasin = pd.concat(pb_rows, ignore_index=True)
    perbasin.to_csv(OUT / "phase8b_li_comparison_perbasin.csv", index=False)

    for label in subsets:
        print(f"\n===== {label} =====")
        print(summary[summary["subset"] == label].to_string(index=False))
    hl = pd.DataFrame(headline_rows)
    print("\n===== headline (all_matched) =====")
    print(hl[hl["subset"] == "all_matched"].round(4).to_string(index=False))
    wins = perbasin.groupby("horizon").agg(
        we_win=("dm_stat", lambda s: int((s < 0).sum())), n=("name", "nunique"))
    print("\nper-basin win counts (lstmres_ens beats li_full, dm_stat<0):")
    print(wins.to_string())


if __name__ == "__main__":
    main()
