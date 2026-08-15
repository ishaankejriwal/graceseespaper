"""Jump-screen robustness for the h1 headline, both variants, committed per the Phase 4 audit.

Full-series screen is test-informed (26/28 flags trigger only on test-period jumps) and must be
reported as a post-hoc diagnostic; the train-only screen is the clean pre-registerable variant.
Exclusion is score-time only: models were fit with flagged basins in the pool.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402

OUT_DIR = ROOT / "results"


def flag_basins(resid: pd.DataFrame, train_end, full_series: bool) -> list[str]:
    flagged = []
    for name in resid.columns:
        s = resid[name] if full_series else resid[name][resid.index < train_end]
        sigma = resid[name][resid.index < train_end].std()
        if (s.diff().abs() / sigma).max() > 6:
            flagged.append(name)
    return flagged


def headline_without(pred: pd.DataFrame, pb: pd.DataFrame, flagged: list[str], h: int = 1) -> dict:
    sub = pred[(pred["horizon"] == h) & (~pred["name"].isin(flagged))]

    def prmse(m):
        g = sub[sub["model"] == m]
        return float(np.sqrt(np.mean((g["target"] - g["pred"]) ** 2)))

    real, own = prmse("kalman_corr_top1"), prmse("kalman_own_ridge")
    pbh = pb[(pb["horizon"] == h) & pb["model"].str.startswith("corr_top1_rand") & (~pb["name"].isin(flagged))]
    plac = pbh.groupby("model").apply(lambda g: np.sqrt(g["sum"].sum() / g["count"].sum()), include_groups=False)
    return {
        "n_flagged": len(flagged), "real_rmse": real, "own_ridge_rmse": own,
        "skill_vs_own": 1 - (real / own) ** 2,
        "beats_placebos": f"{int((real < plac.values).sum())}/{len(plac)}",
        "p_rank": (1 + int((plac.values <= real).sum())) / (1 + len(plac)),
    }


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    resid, _ = deseasonalize_fold(wide, DEFAULT_FOLDS[0])
    train_end = DEFAULT_FOLDS[0].test_start

    pred = pd.read_csv(OUT_DIR / "phase3b_predictions.csv", parse_dates=["target_date"])
    pb = pd.read_csv(OUT_DIR / "phase3b_placebo_basin.csv")

    rows = []
    for label, full in (("full_series_6sigma", True), ("train_only_6sigma", False)):
        flagged = flag_basins(resid, train_end, full)
        row = {"screen": label, **headline_without(pred, pb, flagged)}
        row["flagged_basins"] = ";".join(flagged)
        rows.append(row)
        print(f"{label}: {row['n_flagged']} flagged, skill_vs_own {row['skill_vs_own']:+.4%}, "
              f"beats {row['beats_placebos']} placebos (p={row['p_rank']:.3f})")
    pd.DataFrame(rows).to_csv(OUT_DIR / "phase4_jump_screen.csv", index=False)


if __name__ == "__main__":
    main()
