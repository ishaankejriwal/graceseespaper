"""Ridge, gradient-boosting and MLP corrections of the Kalman forecast with ERA5-Land inputs.

Not used in the manuscript. run_phase6_basin_analysis.py, run_phase6_hybrid.py and
run_phase6_era5_attribution.py read these outputs. Runs
gracefc.experiment_era5.run_era5_experiment on the kept basins with the cached Kalman
parameters, for own-basin arms and correlation-selected neighbor arms, with and without
ERA5-Land features, plus random-neighbor placebo arms.

Inputs: data/processed/basin_month_twsa_global.csv, basin_meta.csv,
era5_basin_month.csv, and results/kalman_fold_params.pkl.
Outputs (tag phase6_era5 by default): results/<tag>_predictions.csv,
_placebo_monthly.csv, _summary.csv (standardized RMSE, placebo rank, DM against the ridge
twin and the twin without ERA5-Land) and _headline.csv (the CONTRASTS below with
block-bootstrap intervals and DM p-values).

Usage:
  python scripts/run_phase6_era5.py [--placebo 20] [--tag phase6_era5] [--smoke]
  GRACEFC_SOURCE=jpl python scripts/run_phase6_era5.py [--placebo 20] [--tag phase6_era5] [--smoke]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_era5 import run_era5_experiment  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402
from gracefc.runtime import load_era5, load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# (challenger, reference) pairs; each gets skill, a bootstrap CI and a DM p per horizon
CONTRASTS = [
    ("ridge_own_era5", "ridge_own"),          # ERA5-Land added to the own-basin ridge
    ("gbm_own_era5", "ridge_own_era5"),       # nonlinear heads on the same features
    ("mlp_own_era5_s0", "ridge_own_era5"),
    ("ridge_corr_top1", "ridge_own"),         # the phase3b neighbor contrast on these rows
    ("ridge_corr_top1_era5", "ridge_own_era5"),  # neighbor given shared meteorology
    ("ridge_corr_top1_era5", "ridge_corr_top1"),  # ERA5-Land added to the neighbor ridge
    ("gbm_corr_top1_era5", "ridge_corr_top1_era5"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default="phase6_era5")
    ap.add_argument("--smoke", action="store_true",
                    help="fold f1, horizon 1, 2 placebo seeds, 1 MLP seed")
    args = ap.parse_args()

    wide, meta, keep = load_sample(DATA)
    era5_wide = load_era5(keep)
    cache = load_params_cache(PARAMS_CACHE, DATA / "basin_month_twsa_global.csv")

    kwargs = {}
    if args.smoke:
        args.placebo, args.tag = 2, args.tag + "_smoke"
        kwargs = {"folds": DEFAULT_FOLDS[:1], "horizons": range(1, 2), "mlp_seeds": (0,)}
    print(f"sample: {wide.shape[1]} basins | era5 vars={len(era5_wide)} | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run_era5_experiment(
        wide, era5_wide, n_placebo=args.placebo, params_cache=cache, **kwargs)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)

    # Per-arm summary: placebo rank, DM against the ridge twin, DM against the no-ERA5 twin
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
        # Ridge twin: the same features with a ridge head (gbm_own_era5 -> ridge_own_era5)
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
