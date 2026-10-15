"""Benchmark table on matched rows: Table 2, Fig. 2a, Table 3 and Appendix B.

Per lead time, the script keeps the (basin, issue month, target month) rows present
for every model and recomputes on them the pooled standardized RMSE, the skill
relative to the stronger damped-persistence variant, and DM tests on pooled monthly
losses. This makes the comparison independent of row-count differences between the
per-model files; with the archived CSR files the intersection is 19 422 rows at h1
down to 18 252 at h6, the full row set of each file.

Inputs:  results/phase2_baseline_predictions.csv (standardized-unit columns)
         results/kalman_predictions.csv (kalman_ar1)
         results/flat12_ridge_predictions.csv (kalman_own_ridge, ridge_own_flat12,
         ridge_own_era5_flat12)
Outputs: results/paper_baseline_ladder.csv (one row per model and lead time)
         results/paper_baseline_contrasts.csv (pairwise skill with block-bootstrap CI
         and DM p; every challenger in CONTRAST_CHALLENGERS against both damped
         variants at every lead)

kalman_ar1 and kalman_own_ridge here are bit-identical to their copies in
phase3b_predictions.csv.

Usage:
  python scripts/build_paper_ladder.py
  GRACEFC_SOURCE=jpl python scripts/build_paper_ladder.py   (results/jpl, Table B2)

The archived tables in results/jpl/ were produced at commit 71be5ec of the JPL
pipeline branch (results/jpl/RUN_PROVENANCE.md; archived as
graceseespaper-71be5ec-jpl-run-code.zip). With this revision, run the full JPL
chain (run_chain.py --source jpl) first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.stats import block_bootstrap_skill_ci, diebold_mariano  # noqa: E402
from gracefc.runtime import results_dir  # noqa: E402

RESULTS = results_dir(ROOT)

P2_MODELS = [
    "climatology_zero",
    "persistence",
    "damped_persistence_rho",
    "damped_persistence_reg",
    "ridge_own_lags",
    "ridge_own_plus_indices",
    "ridge_own_perbasin",
]
KALMAN_MODELS = ["kalman_ar1"]
FLAT12_MODELS = ["kalman_own_ridge", "ridge_own_flat12", "ridge_own_era5_flat12"]
OUR_MODELS = KALMAN_MODELS + FLAT12_MODELS

KEY = ["name", "issue_date", "target_date", "horizon"]

DAMPED_MODELS = ["damped_persistence_rho", "damped_persistence_reg"]
# Challengers the paper quotes against damped persistence; each is contrasted with
# both damped variants at every lead time.
CONTRAST_CHALLENGERS = [
    "kalman_ar1",
    "ridge_own_flat12",
    "ridge_own_era5_flat12",
    "kalman_own_ridge",
    "ridge_own_perbasin",
]


def load() -> pd.DataFrame:
    """Prediction rows of every model in standardized units, restricted per lead time
    to the keys present for all models."""
    p2 = pd.read_csv(
        RESULTS / "phase2_baseline_predictions.csv",
        usecols=KEY + ["model", "target_std_units", "pred_std_units"],
    )
    p2 = p2[p2["model"].isin(P2_MODELS)].rename(
        columns={"target_std_units": "target", "pred_std_units": "pred"}
    )
    kal = pd.read_csv(
        RESULTS / "kalman_predictions.csv", usecols=KEY + ["model", "target", "pred"]
    )
    kal = kal[kal["model"].isin(KALMAN_MODELS)]
    flat = pd.read_csv(
        RESULTS / "flat12_ridge_predictions.csv", usecols=KEY + ["model", "target", "pred"]
    )
    flat = flat[flat["model"].isin(FLAT12_MODELS)]
    df = pd.concat([p2, kal, flat], ignore_index=True)

    # Per horizon, keep only (name, issue, target) keys present for every model.
    kept = []
    for h, sub in df.groupby("horizon"):
        counts = sub.groupby(["name", "issue_date", "target_date"])["model"].nunique()
        full = counts[counts == len(P2_MODELS) + len(OUR_MODELS)].index
        sub = sub.set_index(["name", "issue_date", "target_date"])
        kept.append(sub.loc[sub.index.isin(full)].reset_index())
    out = pd.concat(kept, ignore_index=True)

    # Consistency: targets must agree across models within a key.
    chk = out.groupby(KEY)["target"].agg(["min", "max"])
    assert float((chk["max"] - chk["min"]).abs().max()) < 1e-9, "target mismatch"
    return out


def monthly_losses(df: pd.DataFrame, model: str, h: int) -> pd.Series:
    """Squared error averaged over basins for each target month."""
    sub = df[(df["model"] == model) & (df["horizon"] == h)]
    return (
        sub.assign(loss=(sub["target"] - sub["pred"]) ** 2)
        .groupby("target_date")["loss"]
        .mean()
    )


def main() -> None:
    df = load()
    ladder_rows, contrast_rows = [], []
    for h in range(1, 7):
        sub_h = df[df["horizon"] == h]
        # The stronger damped variant at this lead is the one with lower pooled RMSE
        rmse = {
            m: float(
                np.sqrt(
                    ((g["target"] - g["pred"]) ** 2).mean()
                )
            )
            for m, g in sub_h.groupby("model")
        }
        damped = min(DAMPED_MODELS, key=lambda m: rmse[m])
        ml_damped = monthly_losses(df, damped, h)
        for m in P2_MODELS + OUR_MODELS:
            ml = monthly_losses(df, m, h)
            common = ml.index.intersection(ml_damped.index)
            skill = 1 - ml[common].mean() / ml_damped[common].mean()
            stat, p = diebold_mariano(ml[common].values, ml_damped[common].values, horizon=h)
            ladder_rows.append(
                {
                    "model": m,
                    "horizon": h,
                    "n": int((sub_h["model"] == m).sum()),
                    "rmse_std": rmse[m],
                    "damped_ref": damped,
                    "skill_vs_damped": skill,
                    "dm_stat_vs_damped": stat,
                    "dm_p_vs_damped": p,
                }
            )
        # Pairwise contrasts. The order of this first block fixes the row order of the
        # archived contrasts file.
        pairs = [
            ("kalman_ar1", damped),
            ("kalman_ar1", "ridge_own_perbasin"),
            ("kalman_ar1", "ridge_own_lags"),
            ("kalman_own_ridge", "ridge_own_perbasin"),
            ("ridge_own_perbasin", damped),
            ("ridge_own_flat12", "kalman_ar1"),
            ("ridge_own_era5_flat12", "kalman_ar1"),
            ("ridge_own_era5_flat12", damped),
        ]
        # Every quoted challenger against both damped variants (rho is the stronger
        # at h1, reg at h2-6), skipping pairs the block above already holds
        for m in CONTRAST_CHALLENGERS:
            for d in DAMPED_MODELS:
                if (m, d) not in pairs:
                    pairs.append((m, d))
        for a, b in pairs:
            la, lb = monthly_losses(df, a, h), monthly_losses(df, b, h)
            common = la.index.intersection(lb.index)
            skill = 1 - la[common].mean() / lb[common].mean()
            stat, p = diebold_mariano(la[common].values, lb[common].values, horizon=h)
            point, lo, hi = block_bootstrap_skill_ci(
                df[df["model"].isin([a, b])], a, b, h
            )
            contrast_rows.append(
                {
                    "challenger": a,
                    "reference": b,
                    "horizon": h,
                    "skill": skill,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "dm_stat": stat,
                    "dm_p": p,
                }
            )
    pd.DataFrame(ladder_rows).to_csv(RESULTS / "paper_baseline_ladder.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(
        RESULTS / "paper_baseline_contrasts.csv", index=False
    )
    print(pd.DataFrame(ladder_rows).to_string(index=False))
    print()
    print(pd.DataFrame(contrast_rows).to_string(index=False))


if __name__ == "__main__":
    main()
