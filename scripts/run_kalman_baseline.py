"""Fit the Kalman filter benchmark (kalman_ar1, KF in the paper) and write its forecasts.

For each fold, fits (rho, q, r) per basin by maximum likelihood on the training months
(Sect. 3.4), runs the filter through all months and forecasts rho**h times the filtered
state at leads of 1 to 6 months. Rows are kept by issue date, as in evaluate.split_fold.
Afterwards prints the pooled standardized RMSE of KF, ridge_own_perbasin and
damped_persistence_reg by lead, with a DM test of KF against the per-basin ridge.

Reads data/processed/basin_month_twsa_global.csv, basin_meta.csv and
results/phase2_baseline_predictions.csv. Writes results/kalman_predictions.csv (standardized
units), the KF forecasts read by build_paper_ladder.py (Table 2, Appendix B), the
GRACE-FCast comparison, the centimeter metrics and Fig. 3. The flat12_ridge and phase7
runners write their own kalman_ar1 rows from the cached parameters.

Usage:
  python scripts/run_kalman_baseline.py
  GRACEFC_SOURCE=jpl python scripts/run_kalman_baseline.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.kalman import kalman_predictions  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402
from gracefc.runtime import load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)


def main() -> None:
    wide, meta, keep = load_sample(DATA)

    # The filter is refit on every run (about 20 min) instead of reusing an earlier
    # predictions file, so the output cannot mix data versions or fold definitions.
    rows = []
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        preds = kalman_predictions(resid_wide, fold.test_start, range(1, 7))
        for h, df in preds.items():
            df = df.copy()
            # Attach observed targets and keep rows by issue date, as evaluate.split_fold
            # does, so this file has the same rows as the other forecast files
            tgt = resid_wide.stack().rename("target").reset_index()
            tgt.columns = ["target_date", "name", "target"]
            df = df.merge(tgt, on=["target_date", "name"], how="inner").dropna(subset=["target"])
            df = df[(df["issue_date"] >= fold.test_start) & (df["issue_date"] <= fold.test_end)]
            df["model"] = "kalman_ar1"
            df["fold"] = fold.name
            df["horizon"] = h
            rows.append(df)
    kal = pd.concat(rows, ignore_index=True)
    kal.to_csv(OUT_DIR / "kalman_predictions.csv", index=False)
    compare(kal)


def compare(kal: pd.DataFrame) -> None:
    """Print pooled standardized RMSE of KF, the per-basin ridge and damped persistence by lead."""
    other = pd.read_csv(OUT_DIR / "phase2_baseline_predictions.csv", parse_dates=["issue_date", "target_date"])
    print(f"{'h':>2} {'kalman':>8} {'perbasin':>9} {'damped':>8}  kalman-vs-ridge DM p")
    for h in range(1, 7):
        k = kal[kal["horizon"] == h]
        pb = other[(other["horizon"] == h) & (other["model"] == "ridge_own_perbasin")]
        dp = other[(other["horizon"] == h) & (other["model"] == "damped_persistence_reg")]
        r_k = rmse(k["target"].values, k["pred"].values)
        r_pb = rmse(pb["target_std_units"].values, pb["pred_std_units"].values)
        r_dp = rmse(dp["target_std_units"].values, dp["pred_std_units"].values)
        # Drop the raw-cm columns before renaming, or concat sees duplicate target/pred labels
        pb_std = pb.drop(columns=["target", "pred"]).rename(
            columns={"target_std_units": "target", "pred_std_units": "pred"})
        both = pd.concat([
            k[["name", "target_date", "horizon", "model", "target", "pred"]],
            pb_std[["name", "target_date", "horizon", "model", "target", "pred"]],
        ], ignore_index=True)
        stat, p = pooled_monthly_dm(both, "kalman_ar1", "ridge_own_perbasin", h)
        print(f"{h:>2} {r_k:8.4f} {r_pb:9.4f} {r_dp:8.4f}  stat={stat:+.2f} p={p:.3g}")


if __name__ == "__main__":
    main()
