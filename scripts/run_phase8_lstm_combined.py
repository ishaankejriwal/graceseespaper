"""Stacked model: LSTM correction of the Kalman forecast, then a neighbor residual MLP.

Not used in the manuscript. Stage 1 is the shared-encoder LSTM over 12-month windows of the
filtered state and the ERA5-Land channels; stage 2 is an MLP fitted to the stage-1
residual from the correlation-selected neighbor's propagated state
(gracefc.experiment_lstm_combined). The manuscript draws no table or figure from these
outputs; compute_conventional_metrics.py adds the two-seed ensemble (stacked_ens) to
its tables when the h1-3 and h4-6 prediction files are both present.

Inputs: data/processed/basin_month_twsa_global.csv, basin_meta.csv,
era5_basin_month.csv; results/kalman_fold_params.pkl.
Outputs: results/<tag>_predictions.csv, _placebo_monthly.csv, _summary.csv and
_headline.csv. The default tag phase8_lstm_combined covers leads 1-3; the h4-6 run
uses --tag phase8b_lstm_h46 --horizons 4-6, and run_phase8b_merge.py combines the two.

Usage:
  python scripts/run_phase8_lstm_combined.py [--horizons 1-3] [--tag TAG] [--placebo 20]
      [--smoke]
  GRACEFC_SOURCE=jpl python scripts/run_phase8_lstm_combined.py [same options]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_lstm_combined import run_lstm_combined_experiment  # noqa: E402
from gracefc.phase7 import summarize_and_write  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402
from gracefc.runtime import load_era5, load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# Whether the neighbor MLP correction adds skill when stage 1 is an LSTM that already
# sees the ERA5-Land history
CONTRASTS = [
    ("lstm_own_era5_s0", "ridge_own_era5"),
    ("lstmres_corr_top1_s0", "lstm_own_era5_s0"),
    ("lstmres_corr_top1_s1", "lstm_own_era5_s1"),
    ("lstmres_corr_top1_s0", "ridge_corr_top1_era5"),
    ("lstm_corr_top1_era5_s0", "lstm_own_era5_s0"),
    ("lstmres_nbrin_corr_top1_s0", "lstm_corr_top1_era5_s0"),
    ("lstmres_nbrin_corr_top1_s0", "ridge_corr_top1_era5"),
    # Second stage without the neighbor: lstmres_corr_top1 against lstmres_own has the
    # same stage-1 network and stage-2 capacity and differs only in the neighbor input
    ("lstmres_own_s0", "lstm_own_era5_s0"),
    ("lstmres_own_s1", "lstm_own_era5_s1"),
    ("lstmres_corr_top1_s0", "lstmres_own_s0"),
    ("lstmres_corr_top1_s1", "lstmres_own_s1"),
    # Stage 2 fitted to out-of-fold stage-1 residuals instead of in-sample residuals
    ("lstmres_oof_corr_top1_s0", "lstm_own_era5_s0"),
    ("lstmres_oof_corr_top1_s1", "lstm_own_era5_s1"),
    ("lstmres_oof_corr_top1_s0", "lstmres_corr_top1_s0"),
    ("lstmres_oof_corr_top1_s1", "lstmres_corr_top1_s1"),
    ("lstmres_oof_own_s0", "lstm_own_era5_s0"),
    ("lstmres_oof_own_s1", "lstm_own_era5_s1"),
    # Stage 2 given the same 12-month neighbor history that the input-channel arm
    # (lstm_corr_top1_era5) receives, instead of the propagated neighbor state
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
                    help="inclusive lead range in months, e.g. 4-6 for the phase8b run")
    ap.add_argument("--smoke", action="store_true",
                    help="fold f1, first horizon only, 2 placebo seeds, 1 LSTM seed")
    args = ap.parse_args()

    wide, meta, keep = load_sample(DATA)
    era5_wide = load_era5(keep)
    cache = load_params_cache(PARAMS_CACHE, DATA / "basin_month_twsa_global.csv")

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


