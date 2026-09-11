"""Flat 12-month ridge corrections on the Kalman backbone, leads 1-6, without torch.

Emits the reference forecast (kalman_ar1), the lag-0 own-state ridge comparator
(kalman_own_ridge, phase 3b's definition) and the two flat-12 ridge corrections
(ridge_own_flat12, ridge_own_era5_flat12). These arms used to exist only inside
run_phase7_lstm.py, at leads 1-3, behind a torch import; this runner is their home.

Inputs : data/processed/basin_month_twsa_global.csv, data/processed/basin_meta.csv
         data/processed/era5_basin_month.csv, results/kalman_fold_params.pkl
Outputs: results/flat12_ridge_predictions.csv, results/flat12_ridge_summary.csv

Run: .venv/Scripts/python.exe scripts/run_flat12_ridge.py
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
from gracefc.era5 import era5_wide_by_var  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402
from gracefc.experiment_flat12 import run_flat12_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.runtime import processed_dir, results_dir, shared_processed_dir  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
SHARED_DATA = shared_processed_dir(ROOT)
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"

KALMAN_REF = "kalman_ar1"
OWN_RIDGE_REF = "kalman_own_ridge"


def summarize(pred_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
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
    args = ap.parse_args()
    lo, hi = (int(v) for v in args.horizons.split("-")) if "-" in args.horizons else (
        int(args.horizons), int(args.horizons))

    long_df = pd.read_csv(DATA / "basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(DATA / "basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    era5_long = pd.read_csv(SHARED_DATA / "era5_basin_month.csv", parse_dates=["date"])
    era5_wide = era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
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
