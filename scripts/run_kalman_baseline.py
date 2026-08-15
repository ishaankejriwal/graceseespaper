"""Side experiment: Kalman AR(1)+noise vs per-basin ridge — noise filtering or real dynamics?

If the Kalman filter matches per-basin ridge, the h1-3 own-lag win is mostly measurement-noise
filtering. If ridge stays ahead, multi-lag structure carries real dynamics beyond AR(1).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.kalman import kalman_predictions  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402

OUT_DIR = ROOT / "results"


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])

    # No resume shortcut: a cached predictions file once silently mixed fold
    # protocols (audit 2026-08-14). The ~20 min MLE refit is the price of
    # knowing the file matches the current data and fold semantics.
    rows = []
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        preds = kalman_predictions(resid_wide, fold.test_start, range(1, 7))
        for h, df in preds.items():
            df = df.copy()
            # Attach observed targets and clip to this fold's ISSUE window, matching
            # evaluate.split_fold — target-date clipping puts this file on a different
            # row set than every other runner and breaks cross-file comparisons
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
    other = pd.read_csv(ROOT / "results/phase2_baseline_predictions.csv", parse_dates=["issue_date", "target_date"])
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
