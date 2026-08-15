"""Phase 3: neighbor-graph treatments vs own-lag backbone, with per-treatment random placebos."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.experiment import run_neighbor_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402

OUT_DIR = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="random-placebo seeds per treatment")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--kinds", nargs="+", default=["corr", "pred_lag1", "geo"])
    ap.add_argument("--tag", default="phase3", help="output filename tag")
    args = ap.parse_args()

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    print(f"sample: {wide.shape[1]} basins | kinds={args.kinds} ks={args.ks} seeds={args.seeds}")

    pred_rows = run_neighbor_experiment(
        wide, meta,
        graph_kinds=tuple(args.kinds), ks=tuple(args.ks),
        n_random_seeds=args.seeds,
    )
    OUT_DIR.mkdir(exist_ok=True)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)

    # Standardized pooled RMSE by model x horizon, with skill vs the shared own-lag backbone
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        rows.append({"model": model, "horizon": h,
                     "rmse_std": rmse(grp["target"].values, grp["pred"].values),
                     "rmse_cm": rmse(grp["target_cm"].values, grp["pred_cm"].values),
                     "n": len(grp)})
    summary = pd.DataFrame(rows)
    base = summary[summary["model"] == "ridge_own_lags"].set_index("horizon")["rmse_std"]
    summary["skill_vs_own"] = 1 - (summary["rmse_std"] / summary["horizon"].map(base)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)

    for h in sorted(summary["horizon"].unique()):
        sub = summary[summary["horizon"] == h]
        print(f"\n=== horizon {h} (top 12 by rmse_std) ===")
        print(sub.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
