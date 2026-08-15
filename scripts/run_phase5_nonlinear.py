"""Phase 5: do nonlinear heads (GBM, MLP) or a 2-hop chain beat ridge on the Kalman backbone?"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.experiment_nonlinear import run_nonlinear_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", type=int, default=20)
    ap.add_argument("--tag", default="phase5_nonlinear")
    args = ap.parse_args()

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    cache = load_params_cache(PARAMS_CACHE, ROOT / "data/processed/basin_month_twsa_global.csv")
    print(f"sample: {wide.shape[1]} basins | placebo seeds={args.placebo}")

    pred_rows, plac_monthly = run_nonlinear_experiment(wide, n_placebo=args.placebo, params_cache=cache)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)

    # Each MLP seed is its own row; p_rank compares each single-seed arm to single-seed placebos
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))
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
        if ref and ref in set(pred_rows["model"]):
            stat, p = pooled_monthly_dm(pred_rows, model, ref, h)
            row["dm_vs_ridge_twin"], row["dm_p"] = stat, p
        rows.append(row)
    summary = pd.DataFrame(rows)
    own = summary[summary["model"] == "ridge_own"].set_index("horizon")["rmse_std"]
    summary["skill_vs_ridge_own"] = 1 - (summary["rmse_std"] / summary["horizon"].map(own)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))


if __name__ == "__main__":
    main()
