"""Phase 7 arch 1: the Africa-study two-stage ridge + neighbor-only residual MLP."""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import era5_wide_by_var  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_resmlp import run_resmlp_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.phase7 import summarize_and_write  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# What the two-stage architecture adds (vs one-stage ridge twins on identical info),
# whether a second neighbor helps, and whether the neighbor survives ERA5 conditioning
CONTRASTS = [
    ("resmlp_corr_top1_s0", "ridge_corr_top1"),
    ("resmlp_corr_top1_s0", "ridge_own"),
    ("resmlp_corr_top2_s0", "resmlp_corr_top1_s0"),
    ("resmlp_corr_top1_era5_s0", "ridge_corr_top1_era5"),
    ("resmlp_corr_top1_era5_s0", "resmlp_own_era5_s0"),
    ("resmlp_corr_top2_era5_s0", "resmlp_corr_top1_era5_s0"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default="phase7_resmlp")
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
    print(f"sample: {wide.shape[1]} basins | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run_resmlp_experiment(
        wide, era5_wide, n_placebo=args.placebo, params_cache=cache, **kwargs)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)
    summarize_and_write(pred_rows, plac_monthly, CONTRASTS, OUT_DIR, args.tag)


if __name__ == "__main__":
    main()
