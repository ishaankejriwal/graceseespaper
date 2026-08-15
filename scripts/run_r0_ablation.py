"""r=0 ablation: how much of the Kalman benchmark's margin is the observation-noise term?

With r fixed at 0 the update gain is 1 at every observed month, so the "filter" tracks
the raw observation and the forecast is rho^h times the last observation — damped
persistence with MLE rho, estimated in the identical model class, on the identical
protocol. Three pairwise contrasts decompose the benchmark margin:

  kalman_ar1 vs kalman_r0            -> the observation-noise (filtering) term
  kalman_r0  vs damped_persistence   -> rho estimation / gap handling beyond tuned damping
  kalman_ar1 vs damped_persistence   -> the full published margin (consistency check)

If the first contrast carries the margin, the paper's noise-filtering attribution is
identified; if not, the title's mechanism claim must stay "state-space filtering"
(audit 2026-08-15, blocker 2). Rows match kalman_predictions.csv exactly (issue-date
fold membership, same keep-list).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.kalman import kalman_predictions  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402

OUT_DIR = ROOT / "results"


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])

    rows = []
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        preds = kalman_predictions(resid_wide, fold.test_start, range(1, 7), fixed_r=0.0)
        for h, df in preds.items():
            df = df.copy()
            tgt = resid_wide.stack().rename("target").reset_index()
            tgt.columns = ["target_date", "name", "target"]
            df = df.merge(tgt, on=["target_date", "name"], how="inner").dropna(subset=["target"])
            df = df[(df["issue_date"] >= fold.test_start) & (df["issue_date"] <= fold.test_end)]
            df["model"] = "kalman_r0"
            df["fold"] = fold.name
            df["horizon"] = h
            rows.append(df)
        print(f"fold {fold.name} done", flush=True)
    r0 = pd.concat(rows, ignore_index=True)
    r0.to_csv(OUT_DIR / "kalman_r0_predictions.csv", index=False)

    kal = pd.read_csv(OUT_DIR / "kalman_predictions.csv", parse_dates=["issue_date", "target_date"])
    ph2 = pd.read_csv(OUT_DIR / "phase2_baseline_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    damped = ph2[ph2["model"] == "damped_persistence_reg"].drop(columns=["target", "pred"]).rename(
        columns={"target_std_units": "target", "pred_std_units": "pred"})
    cols = ["name", "target_date", "horizon", "model", "target", "pred"]
    pool = pd.concat([kal[cols], r0[cols], damped[cols]], ignore_index=True)

    out = []
    for a, b, label in (
        ("kalman_ar1", "kalman_r0", "noise_filtering_term"),
        ("kalman_r0", "damped_persistence_reg", "rho_estimation_term"),
        ("kalman_ar1", "damped_persistence_reg", "full_margin"),
    ):
        for h in range(1, 7):
            point, lo, hi = block_bootstrap_skill_ci(pool, a, b, h)
            stat, p = pooled_monthly_dm(pool, a, b, h)
            sub = pool[pool["horizon"] == h]
            out.append({
                "component": label, "challenger": a, "reference": b, "horizon": h,
                "rmse_a": rmse(sub[sub["model"] == a]["target"].values,
                               sub[sub["model"] == a]["pred"].values),
                "rmse_b": rmse(sub[sub["model"] == b]["target"].values,
                               sub[sub["model"] == b]["pred"].values),
                "skill_pct": 100 * point, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                "dm_stat": stat, "dm_p": p,
            })
    summary = pd.DataFrame(out)
    summary.to_csv(OUT_DIR / "r0_ablation_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
