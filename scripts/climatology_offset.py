"""Share of climatology's 1-month error due to a per-basin, per-fold level shift (Sect. 3.1).

The target is standardized by the training-window standard deviation, yet
climatology's pooled RMSE is about 1.6. This splits its MSE into the mean test
residual of each basin and fold (drift away from the frozen trend and seasonal fit)
and the variance around that mean.

Input: results/phase2_baseline_predictions.csv.
Output: results/climatology_offset.csv. CSR only (fixed CSR paths).

Usage: python scripts/climatology_offset.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

d = pd.read_csv(ROOT / "results/phase2_baseline_predictions.csv",
                usecols=["name", "model", "horizon", "fold", "target_std_units", "pred_std_units"])
y = d.loc[(d.model == "climatology_zero") & (d.horizon == 1)]
assert (y["pred_std_units"] == 0).all()  # climatology forecasts zero, so the target is the error
mean = y.groupby(["name", "fold"])["target_std_units"].transform("mean")
mse = (y["target_std_units"] ** 2).mean()
out = pd.DataFrame([{"mse": mse, "rmse": mse ** 0.5,
                     "offset_share": (mean ** 2).mean() / mse,
                     "within_fold_variance": ((y["target_std_units"] - mean) ** 2).mean(),
                     "n": len(y)}])
out.to_csv(ROOT / "results/climatology_offset.csv", index=False)
print(out.to_string(index=False))
assert len(y) == 19422
