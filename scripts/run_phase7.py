"""Neural corrections of the Kalman forecast, one architecture per run.

  --arch resmlp   two-stage model: a ridge correction, then an MLP fitted to the
                  ridge's training residual (gracefc.experiment_resmlp)
  --arch lstm     shared-encoder LSTM over 12-month windows of the filtered state,
                  optionally with the ERA5-Land channels (gracefc.experiment_lstm)
  --arch gnn      one-layer graph attention over the correlation graph
                  (gracefc.experiment_gnn; not used in the manuscript)

The lstm run feeds Table 4 (the KF, KF-R1E, KF-R12, KF-R12E, LSTM and MLP rows) and
Fig. 6; the resmlp run feeds the residual MLP rows of Table 4. Each architecture has its
own engine and contrasts (ARCHS); inputs, Kalman parameter cache, placebo count and
output layout are shared.

Inputs: data/processed/basin_month_twsa_global.csv, basin_meta.csv,
era5_basin_month.csv; results/kalman_fold_params.pkl.
Outputs (tag phase7_<arch> by default): results/<tag>_predictions.csv,
_placebo_monthly.csv, _summary.csv and _headline.csv.

Usage:
  python scripts/run_phase7.py --arch {resmlp,lstm,gnn} [--placebo 20] [--tag TAG] [--smoke]
  GRACEFC_SOURCE=jpl python scripts/run_phase7.py [same options]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.phase7 import summarize_and_write  # noqa: E402
from gracefc.cache import load_params_cache  # noqa: E402
from gracefc.runtime import load_era5, load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

# arch -> (engine module.function, seed kwarg, headline contrasts)
ARCHS = {
    # The two-stage model against one-stage ridge twins on the same information, a
    # second neighbor against one, and the neighbor with ERA5-Land conditioning
    "resmlp": ("experiment_resmlp.run_resmlp_experiment", "mlp_seeds", [
        ("resmlp_corr_top1_s0", "ridge_corr_top1"),
        ("resmlp_corr_top1_s0", "ridge_own"),
        ("resmlp_corr_top2_s0", "resmlp_corr_top1_s0"),
        ("resmlp_corr_top1_era5_s0", "ridge_corr_top1_era5"),
        ("resmlp_corr_top1_era5_s0", "resmlp_own_era5_s0"),
        ("resmlp_corr_top2_era5_s0", "resmlp_corr_top1_era5_s0"),
    ]),
    # The LSTM against the ridge on the current state, the neighbor channel against
    # none, and each against the ridge twin on the same information
    "lstm": ("experiment_lstm.run_lstm_experiment", "lstm_seeds", [
        ("lstm_own_s0", "ridge_own"),
        ("lstm_corr_top1_s0", "lstm_own_s0"),
        ("lstm_corr_top1_s0", "ridge_corr_top1"),
        ("lstm_own_era5_s0", "ridge_own_era5"),
        ("lstm_corr_top1_era5_s0", "lstm_own_era5_s0"),
        ("lstm_corr_top1_era5_s0", "ridge_corr_top1_era5"),
    ]),
    # Graph attention against the ridge twin, and two attended neighbors (k=2) against one
    "gnn": ("experiment_gnn.run_gnn_experiment", "gnn_seeds", [
        ("gnn_corr_top1_s0", "ridge_corr_top1"),
        ("gnn_corr_top1_s0", "ridge_own"),
        ("gnn_corr_top2_s0", "gnn_corr_top1_s0"),
        ("gnn_corr_top2_s0", "ridge_corr_top2"),
        ("gnn_corr_top1_era5_s0", "ridge_corr_top1_era5"),
        ("gnn_corr_top2_era5_s0", "gnn_corr_top1_era5_s0"),
    ]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(ARCHS), required=True)
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default=None, help="default phase7_<arch>")
    ap.add_argument("--smoke", action="store_true",
                    help="fold f1, horizon 1, 2 placebo seeds, 1 model seed")
    args = ap.parse_args()
    engine_path, seed_kw, contrasts = ARCHS[args.arch]
    module, func = engine_path.split(".")
    run = getattr(__import__(f"gracefc.{module}", fromlist=[func]), func)
    tag = args.tag or f"phase7_{args.arch}"

    wide, meta, keep = load_sample(DATA)
    era5_wide = load_era5(keep)
    cache = load_params_cache(PARAMS_CACHE, DATA / "basin_month_twsa_global.csv")

    kwargs = {}
    if args.smoke:
        args.placebo, tag = 2, tag + "_smoke"
        kwargs = {"folds": DEFAULT_FOLDS[:1], "horizons": range(1, 2), seed_kw: (0,)}
    print(f"sample: {wide.shape[1]} basins | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run(wide, era5_wide, n_placebo=args.placebo, params_cache=cache, **kwargs)
    pred_rows.to_csv(OUT_DIR / f"{tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{tag}_placebo_monthly.csv", index=False)
    summarize_and_write(pred_rows, plac_monthly, contrasts, OUT_DIR, tag)


if __name__ == "__main__":
    main()
