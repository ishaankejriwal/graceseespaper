"""Where does the filter gain most? Rank correlations of the per-basin 1-month
KF skill over damped persistence (from RMSE in cm) with basin area, mean annual
ERA5-Land precipitation and the lag-1 autocorrelation of the residual.

Reads saved outputs only; writes results/kalman_gain_correlates.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

conv = pd.read_csv(ROOT / "results/conventional_metrics_perbasin.csv")
rmse = conv[conv.horizon == 1].pivot(index="name", columns="model", values="rmse_cm")
skill = 1 - (rmse["kalman_ar1"] / rmse["damped_persistence"]) ** 2

meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv").set_index("name")
era5 = pd.read_csv(ROOT / "data/processed/era5_basin_month.csv")
precip = era5.groupby("name")["precip_cm"].mean() * 12  # mean annual total, cm

pred = pd.read_csv(ROOT / "results/phase2_baseline_predictions.csv",
                   usecols=["name", "model", "horizon", "target_date", "target"])
pred = (pred[(pred.model == "damped_persistence_rho") & (pred.horizon == 1)]
        .drop_duplicates(["name", "target_date"]).sort_values(["name", "target_date"]))
ac1 = pred.groupby("name")["target"].apply(lambda y: np.corrcoef(y.values[:-1], y.values[1:])[0, 1])

df = pd.DataFrame({"skill": skill,
                   "area_km2": meta["area_km2"].reindex(skill.index),
                   "precip_cm_yr": precip.reindex(skill.index),
                   "resid_ac1": ac1.reindex(skill.index)})
rows = []
for col in ["area_km2", "precip_cm_yr", "resid_ac1"]:
    rho, p = spearmanr(df["skill"], df[col], nan_policy="omit")
    rows.append({"covariate": col, "spearman_rho": rho, "p": p, "n": int(df[col].notna().sum())})
dry = df["precip_cm_yr"] < df["precip_cm_yr"].quantile(0.25)
rows.append({"covariate": "median_skill_driest_quartile", "spearman_rho": df.loc[dry, "skill"].median(),
             "p": np.nan, "n": int(dry.sum())})
rows.append({"covariate": "median_skill_other_basins", "spearman_rho": df.loc[~dry, "skill"].median(),
             "p": np.nan, "n": int((~dry).sum())})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/kalman_gain_correlates.csv", index=False)
print(out.to_string(index=False))
assert len(df) == 234
