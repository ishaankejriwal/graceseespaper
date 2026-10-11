"""Diebold-Mariano tests and bootstrap intervals on pooled squared error in centimeters.

Sect. 4.5 compares the retained forecasts in cm; this tests those differences with the
same pooled monthly DM and moving-block bootstrap used for the standardized comparisons.
Reuses the row construction of compute_conventional_metrics.py.

Output: results/centimeter_contrasts.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compute_conventional_metrics import HORIZONS, RES, attach_cm_and_full, load_model_rows, load_wide  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402

CM = ("obs_anom_cm", "pred_anom_cm")
PAIRS = [(m, "kalman_ar1") for m in ("ridge_own_era5_flat12", "ridge_own_flat12", "kalman_own_ridge")]
PAIRS.append(("kalman_ar1", "damped_persistence"))

df = attach_cm_and_full(load_model_rows(), load_wide())
rows = []
for a, b in PAIRS:
    for h in HORIZONS:
        _, p = pooled_monthly_dm(df, a, b, h, value_cols=CM)
        skill, lo, hi = block_bootstrap_skill_ci(df, a, b, h, value_cols=CM)
        rows.append({"model": a, "vs": b, "horizon": h, "skill": skill, "ci_lo": lo, "ci_hi": hi, "dm_p": p})
out = pd.DataFrame(rows)
out.to_csv(RES / "centimeter_contrasts.csv", index=False)
print(out.round(4).to_string(index=False))
