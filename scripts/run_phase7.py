"""Phase 7: one neural architecture on the shared Kalman-backbone inputs, head to head.

  --arch resmlp   the Africa-study two-stage ridge + neighbor-only residual MLP
  --arch lstm     shared-encoder LSTM over Kalman-filtered state sequences
  --arch gnn      one-layer graph attention over the correlation graph

Each arch has its own engine and headline contrasts; everything else (inputs, cache,
placebo count, outputs, summary) is identical, so they share this one runner.
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
    # What the two-stage architecture adds (vs one-stage ridge twins on identical info),
    # whether a second neighbor helps, and whether the neighbor survives ERA5 conditioning
    "resmlp": ("experiment_resmlp.run_resmlp_experiment", "mlp_seeds", [
        ("resmlp_corr_top1_s0", "ridge_corr_top1"),
        ("resmlp_corr_top1_s0", "ridge_own"),
        ("resmlp_corr_top2_s0", "resmlp_corr_top1_s0"),
        ("resmlp_corr_top1_era5_s0", "ridge_corr_top1_era5"),
        ("resmlp_corr_top1_era5_s0", "resmlp_own_era5_s0"),
        ("resmlp_corr_top2_era5_s0", "resmlp_corr_top1_era5_s0"),
    ]),
    # Does sequence memory beat the AR(1) backbone, does the neighbor channel add anything,
    # and do both survive next to the flat linear twins on the same information?
    "lstm": ("experiment_lstm.run_lstm_experiment", "lstm_seeds", [
        ("lstm_own_s0", "ridge_own"),
        ("lstm_corr_top1_s0", "lstm_own_s0"),
        ("lstm_corr_top1_s0", "ridge_corr_top1"),
        ("lstm_own_era5_s0", "ridge_own_era5"),
        ("lstm_corr_top1_era5_s0", "lstm_own_era5_s0"),
        ("lstm_corr_top1_era5_s0", "ridge_corr_top1_era5"),
    ]),
    # Does learned message passing beat the flat linear twin, and does a second edge
    # (k=2 with attention) add anything beyond the top-1 neighbor?
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
