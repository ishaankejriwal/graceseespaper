"""Refit the Kalman filter with r fixed at zero (the zero-r variant, kalman_r0).

With r = 0 the Kalman gain is 1 at every observed month, so the filtered state equals
the observed residual and the forecast is rho**h times the last residual: damped
persistence with a maximum-likelihood rho, fitted in the same model class on the same
folds (Sect. 3.4). Three contrasts, each with a block-bootstrap CI and DM test,
decompose the margin of KF over damped persistence:

  kalman_ar1 vs kalman_r0                noise_filtering_term (the model-noise term)
  kalman_r0  vs damped_persistence_reg   rho_estimation_term (likelihood fit of rho and
                                         propagation across missing months)
  kalman_ar1 vs damped_persistence_reg   full_margin

Rows match results/kalman_predictions.csv (issue-date fold membership, same basins).
Feeds Fig. 4a and the 1-month decomposition in Sect. 4.2.

Inputs: data/processed/basin_month_twsa_global.csv, basin_meta.csv,
results/kalman_predictions.csv, results/phase2_baseline_predictions.csv.
Outputs: results/kalman_r0_predictions.csv, results/r0_ablation_summary.csv.

Usage:
  python scripts/run_r0_ablation.py
  GRACEFC_SOURCE=jpl python scripts/run_r0_ablation.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.kalman import kalman_predictions  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.runtime import load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)


def main() -> None:
    wide, meta, keep = load_sample(DATA)

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
