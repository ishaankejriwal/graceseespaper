"""Fit the pooled ridge corrections to the Kalman filter forecast at leads of 1 to 6 months.

Writes four forecasts (Sect. 3.5): kalman_ar1 (KF); kalman_own_ridge (KF-R1), a ridge
on the current filtered state; ridge_own_flat12 (KF-R12), a ridge on the last 12 months
of filtered state; and ridge_own_era5_flat12 (KF-R12E), which adds the 12-month history
of the 11 ERA5-Land anomaly series. Each correction predicts the error of KF. The
summary gives skill against KF and KF-R1 and a DM test against KF. Does not need torch.
Feeds Table 3 and Fig. 5 directly, and Tables 2, 5, B1 and C1 and Figs. 2 and 7
through build_paper_ladder.py, run_phase6_li_comparison.py and
compute_conventional_metrics.py.

Inputs:  data/processed/basin_month_twsa_global.csv, data/processed/basin_meta.csv
         data/processed/era5_basin_month.csv, results/kalman_fold_params.pkl
         (the cache is fitted and written if missing or stale)
Outputs: results/flat12_ridge_predictions.csv, results/flat12_ridge_summary.csv

Usage:
  python scripts/run_flat12_ridge.py [--horizons 1-6] [--summary-only]
  GRACEFC_SOURCE=jpl python scripts/run_flat12_ridge.py [--horizons 1-6] [--summary-only]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_flat12 import run_flat12_experiment  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.runtime import load_era5, load_sample, processed_dir, results_dir  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

KALMAN_REF = "kalman_ar1"
OWN_RIDGE_REF = "kalman_own_ridge"


def summarize(pred_rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE, skill against KF and KF-R1, and DM test against KF, by model and lead."""
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        # A DM test of the reference against itself is 0/0, so those cells are left NaN
        if model == KALMAN_REF:
            stat, p = float("nan"), float("nan")
        else:
            stat, p = pooled_monthly_dm(pred_rows, model, KALMAN_REF, h)
        rows.append({
            "model": model, "horizon": h,
            "rmse_std": rmse(grp["target"].values, grp["pred"].values),
            "n": len(grp),
            "dm_stat_vs_kalman": stat, "dm_p_vs_kalman": p,
        })
    summary = pd.DataFrame(rows)
    # Skill is the MSE reduction, 1 - (rmse/rmse_ref)^2, on identical rows
    for col, ref in (("skill_vs_kalman", KALMAN_REF), ("skill_vs_ridge_own", OWN_RIDGE_REF)):
        base = summary[summary["model"] == ref].set_index("horizon")["rmse_std"]
        summary[col] = 1 - (summary["rmse_std"] / summary["horizon"].map(base)) ** 2
    cols = ["model", "horizon", "rmse_std", "n", "skill_vs_kalman", "skill_vs_ridge_own",
            "dm_stat_vs_kalman", "dm_p_vs_kalman"]
    return summary[cols].sort_values(["horizon", "rmse_std"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1-6")
    ap.add_argument("--tag", default="flat12_ridge")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild <tag>_summary.csv from the existing "
                         "<tag>_predictions.csv and leave the predictions untouched")
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.horizons.split("-")) if "-" in args.horizons else (
        int(args.horizons), int(args.horizons))

    if args.summary_only:
        pred_path = OUT_DIR / f"{args.tag}_predictions.csv"
        if not pred_path.exists():
            raise SystemExit(f"--summary-only needs {pred_path}, which does not exist")
        summary = summarize(pd.read_csv(pred_path))
        summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
        print(summary.to_string(index=False))
        return

    wide, meta, keep = load_sample(DATA)
    era5_wide = load_era5(keep)
    cache = load_params_cache(PARAMS_CACHE, DATA / "basin_month_twsa_global.csv")
    cached_folds = set(cache)
    print(f"sample: {wide.shape[1]} basins | horizons {lo}-{hi} | "
          f"cached folds: {sorted(cached_folds) or 'none'}", flush=True)

    t0 = time.time()
    pred_rows = run_flat12_experiment(
        wide, era5_wide, horizons=range(lo, hi + 1), folds=DEFAULT_FOLDS, params_cache=cache)
    print(f"experiment wall time: {time.time() - t0:.1f}s", flush=True)
    if set(cache) != cached_folds:
        save_params_cache(PARAMS_CACHE, cache, DATA / "basin_month_twsa_global.csv")

    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    summary = summarize(pred_rows)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
