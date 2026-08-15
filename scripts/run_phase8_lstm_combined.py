"""Phase 8: stacked LSTM(own state + ERA5 sequences) + neighbor-only residual MLP."""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import era5_wide_by_var  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_lstm_combined import run_lstm_combined_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.phase7 import summarize_and_write  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# Does the neighbor-only MLP correction still add skill once stage 1 is the LSTM that
# already integrates the ERA5 forcing history — or does sequence-ERA5 absorb the last
# of the neighbor signal the way flat ERA5 absorbed the linear effect?
CONTRASTS = [
    ("lstm_own_era5_s0", "ridge_own_era5"),
    ("lstmres_corr_top1_s0", "lstm_own_era5_s0"),
    ("lstmres_corr_top1_s1", "lstm_own_era5_s1"),
    ("lstmres_corr_top1_s0", "ridge_corr_top1_era5"),
    ("lstm_corr_top1_era5_s0", "lstm_own_era5_s0"),
    ("lstmres_nbrin_corr_top1_s0", "lstm_corr_top1_era5_s0"),
    ("lstmres_nbrin_corr_top1_s0", "ridge_corr_top1_era5"),
    # Neighbor-free stage-2 control (2026-08-14): does ANY second stage explain the
    # correction, or specifically the neighbor's information? The decisive pair is
    # lstmres_corr_top1 vs lstmres_own — same stage-1 net, same stage-2 capacity,
    # only the information source differs.
    ("lstmres_own_s0", "lstm_own_era5_s0"),
    ("lstmres_own_s1", "lstm_own_era5_s1"),
    ("lstmres_corr_top1_s0", "lstmres_own_s0"),
    ("lstmres_corr_top1_s1", "lstmres_own_s1"),
    # Out-of-fold stage-2 residuals (audit repair, 2026-08-15): the correction and the
    # own-state diagnostic re-estimated without the in-sample-residual mechanism
    ("lstmres_oof_corr_top1_s0", "lstm_own_era5_s0"),
    ("lstmres_oof_corr_top1_s1", "lstm_own_era5_s1"),
    ("lstmres_oof_corr_top1_s0", "lstmres_corr_top1_s0"),
    ("lstmres_oof_corr_top1_s1", "lstmres_corr_top1_s1"),
    ("lstmres_oof_own_s0", "lstm_own_era5_s0"),
    ("lstmres_oof_own_s1", "lstm_own_era5_s1"),
    # Delivery-equalized correction: stage 2 fed the input arm's exact 12-month
    # neighbor representation — the clean "same information, different delivery" pair
    ("lstmres_corr_top1_hist12_s0", "lstm_own_era5_s0"),
    ("lstmres_corr_top1_hist12_s1", "lstm_own_era5_s1"),
    ("lstmres_corr_top1_hist12_s0", "lstmres_corr_top1_s0"),
    ("lstmres_corr_top1_hist12_s1", "lstmres_corr_top1_s1"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default="phase8_lstm_combined")
    ap.add_argument("--horizons", default="1-3",
                    help="inclusive lead range, e.g. 4-6 (phase 8b extension)")
    ap.add_argument("--smoke", action="store_true",
                    help="fold f1, first horizon only, 2 placebo seeds, 1 LSTM seed")
    args = ap.parse_args()

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    era5_long = pd.read_csv(ROOT / "data/processed/era5_basin_month.csv", parse_dates=["date"])
    era5_wide = era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
    cache = load_params_cache(PARAMS_CACHE, ROOT / "data/processed/basin_month_twsa_global.csv")

    h_lo, h_hi = (int(x) for x in args.horizons.split("-"))
    kwargs = {"horizons": range(h_lo, h_hi + 1)}
    if args.smoke:
        args.placebo, args.tag = 2, args.tag + "_smoke"
        kwargs = {"folds": DEFAULT_FOLDS[:1], "horizons": range(h_lo, h_lo + 1), "seeds": (0,)}
    print(f"sample: {wide.shape[1]} basins | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run_lstm_combined_experiment(
        wide, era5_wide, n_placebo=args.placebo, params_cache=cache, **kwargs)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)
    summarize_and_write(pred_rows, plac_monthly, CONTRASTS, OUT_DIR, args.tag)


if __name__ == "__main__":
    main()


