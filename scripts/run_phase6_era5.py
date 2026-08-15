"""Phase 6: do ERA5-Land forcings lift GBM/MLP over ridge, and the neighbor headline?"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import era5_wide_by_var  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_era5 import run_era5_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# Headline contrasts: (challenger, reference) — skill, bootstrap CI, and DM p per horizon
CONTRASTS = [
    ("ridge_own_era5", "ridge_own"),          # does ERA5 add own-basin information at all?
    ("gbm_own_era5", "ridge_own_era5"),       # nonlinearity question on the enriched features
    ("mlp_own_era5_s0", "ridge_own_era5"),
    ("ridge_corr_top1", "ridge_own"),         # Phase 3b headline re-anchored on these rows
    ("ridge_corr_top1_era5", "ridge_own_era5"),  # neighbor conditioned on shared meteorology
    ("ridge_corr_top1_era5", "ridge_corr_top1"),  # does ERA5 improve the headline model?
    ("gbm_corr_top1_era5", "ridge_corr_top1_era5"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default="phase6_era5")
    ap.add_argument("--smoke", action="store_true",
                    help="fold f1, horizon 1, 2 placebo seeds, 1 MLP seed")
    args = ap.parse_args()

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])

    era5_long = pd.read_csv(ROOT / "data/processed/era5_basin_month.csv", parse_dates=["date"])
    era5_wide = era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
    cache = load_params_cache(PARAMS_CACHE, ROOT / "data/processed/basin_month_twsa_global.csv")

    kwargs = {}
    if args.smoke:
        args.placebo, args.tag = 2, args.tag + "_smoke"
        kwargs = {"folds": DEFAULT_FOLDS[:1], "horizons": range(1, 2), "mlp_seeds": (0,)}
    print(f"sample: {wide.shape[1]} basins | era5 vars={len(era5_wide)} | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run_era5_experiment(
        wide, era5_wide, n_placebo=args.placebo, params_cache=cache, **kwargs)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)

    # Per-arm summary with placebo rank, ridge-twin DM, and no-ERA5-twin DM
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))
    models_present = set(pred_rows["model"])
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {"model": model, "horizon": h,
               "rmse_std": rmse(grp["target"].values, grp["pred"].values), "n": len(grp)}
        fam = model.rsplit("_s", 1)[0] if "_s" in model else model
        prefix = f"{fam}_rand"
        dist = plac_pooled.loc[
            plac_pooled.index.get_level_values(0).str.startswith(prefix)
            & (plac_pooled.index.get_level_values(1) == h)]["rmse"].values
        if len(dist):
            row["placebo_n"] = len(dist)
            row["placebo_beaten"] = int((row["rmse_std"] < dist).sum())
            row["p_rank"] = float((1 + (dist <= row["rmse_std"]).sum()) / (1 + len(dist)))
        ref = "ridge_" + fam.split("_", 1)[1] if "_" in fam and not fam.startswith("ridge") else None
        if ref and ref in models_present:
            stat, p = pooled_monthly_dm(pred_rows, model, ref, h)
            row["dm_vs_ridge_twin"], row["dm_p"] = stat, p
        if fam.endswith("_era5"):
            twin = model.replace("_era5", "")
            if twin in models_present:
                stat, p = pooled_monthly_dm(pred_rows, model, twin, h)
                row["dm_vs_no_era5"], row["dm_p_era5"] = stat, p
        rows.append(row)
    summary = pd.DataFrame(rows)
    own = summary[summary["model"] == "ridge_own"].set_index("horizon")["rmse_std"]
    summary["skill_vs_ridge_own"] = 1 - (summary["rmse_std"] / summary["horizon"].map(own)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)

    head_rows = []
    for a, b in CONTRASTS:
        if a not in models_present or b not in models_present:
            continue
        for h in sorted(pred_rows["horizon"].unique()):
            point, lo, hi = block_bootstrap_skill_ci(pred_rows, a, b, h)
            stat, p = pooled_monthly_dm(pred_rows, a, b, h)
            head_rows.append({"challenger": a, "reference": b, "horizon": h,
                              "skill_pct": 100 * point, "ci_lo_pct": 100 * lo,
                              "ci_hi_pct": 100 * hi, "dm_stat": stat, "dm_p": p})
    headline = pd.DataFrame(head_rows)
    headline.to_csv(OUT_DIR / f"{args.tag}_headline.csv", index=False)

    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))
    print("\n=== headline contrasts ===")
    print(headline.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
