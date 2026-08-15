"""Phase 3b: do neighbor filtered states add skill on the Kalman backbone?"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.experiment_kalman import run_kalman_backbone_experiment  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"


def parse_cells(specs: list[str]) -> tuple:
    # Format: kind:k1,k2  e.g. corr:1,2,3 geo:1,2,3 corr_min300:1
    cells = []
    for spec in specs:
        kind, ks = spec.split(":")
        cells += [(kind, int(k)) for k in ks.split(",")]
    return tuple(cells)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--cells", nargs="+",
                    default=["corr:1,2,3", "geo:1,2,3", "corr_min300:1", "corr_min500:1", "corr_min1000:1"])
    ap.add_argument("--tag", default="phase3b")
    ap.add_argument("--condition-indices", nargs="*", default=None,
                    help="index columns (e.g. nino34 dmi) appended to every arm")
    args = ap.parse_args()
    cells = parse_cells(args.cells)

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    print(f"sample: {wide.shape[1]} basins | cells={cells} seeds={args.seeds}")

    indices = None
    if args.condition_indices:
        indices = pd.read_csv(ROOT / "data/processed/indices.csv", parse_dates=["date"]).set_index("date")
        indices = indices[args.condition_indices]
        print(f"conditioning every arm on: {list(indices.columns)}")

    cache = load_params_cache(PARAMS_CACHE, ROOT / "data/processed/basin_month_twsa_global.csv")
    n_before = len(cache)
    pred_rows, plac_monthly, plac_basin = run_kalman_backbone_experiment(
        wide, meta, cells=cells, n_random_seeds=args.seeds, params_cache=cache, indices=indices,
    )
    if len(cache) > n_before:
        save_params_cache(PARAMS_CACHE, cache, ROOT / "data/processed/basin_month_twsa_global.csv")

    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)
    plac_basin.to_csv(OUT_DIR / f"{args.tag}_placebo_basin.csv", index=False)

    # Real-arm summary with rank among the placebo pooled-RMSE distribution
    rows = []
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"]))) if len(plac_monthly) else None
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {"model": model, "horizon": h,
               "rmse_std": rmse(grp["target"].values, grp["pred"].values), "n": len(grp)}
        if model.startswith("kalman_") and model not in ("kalman_ar1", "kalman_own_ridge") and plac_pooled is not None:
            cell = model.removeprefix("kalman_")
            dist = plac_pooled.loc[plac_pooled.index.get_level_values(0).str.startswith(f"{cell}_rand")]
            dist = dist[dist.index.get_level_values(1) == h]["rmse"].values
            if len(dist):
                # Empirical p: rank of the real arm within its placebo null
                row["placebo_n"] = len(dist)
                row["placebo_beaten"] = int((row["rmse_std"] < dist).sum())
                row["p_rank"] = float((1 + (dist <= row["rmse_std"]).sum()) / (1 + len(dist)))
        rows.append(row)
    summary = pd.DataFrame(rows)
    base = summary[summary["model"] == "kalman_ar1"].set_index("horizon")["rmse_std"]
    own = summary[summary["model"] == "kalman_own_ridge"].set_index("horizon")["rmse_std"]
    summary["skill_vs_kalman"] = 1 - (summary["rmse_std"] / summary["horizon"].map(base)) ** 2
    summary["skill_vs_own_ridge"] = 1 - (summary["rmse_std"] / summary["horizon"].map(own)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))


if __name__ == "__main__":
    main()
