"""Build the paper's matched-row baseline ladder (Table 1 / Fig 1 source).

Motivation: phase2_baseline_predictions.csv row counts shrink with horizon
(19,656 at h1 down to 18,486 at h6) while kalman_predictions.csv and
phase3b_predictions.csv keep 19,656 rows at every horizon, so summary-file
RMSEs are not directly comparable across the two files at h>=2. This script
intersects the row sets per horizon and recomputes pooled RMSE, skill vs the
stronger damped-persistence variant, and pooled-monthly DM tests on exactly
matched rows.

Inputs : results/phase2_baseline_predictions.csv (std-unit columns)
         results/phase3b_predictions.csv (kalman_ar1, kalman_own_ridge,
         kalman_corr_top1; std units)
Outputs: results/paper_baseline_ladder.csv  (one row per model x horizon)
         results/paper_baseline_contrasts.csv (key pairwise DM/CI contrasts)

Run: .venv/Scripts/python.exe scripts/build_paper_ladder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.stats import block_bootstrap_skill_ci, diebold_mariano  # noqa: E402

RESULTS = ROOT / "results"

P2_MODELS = [
    "climatology_zero",
    "persistence",
    "damped_persistence_rho",
    "damped_persistence_reg",
    "ridge_own_lags",
    "ridge_own_plus_indices",
    "ridge_own_perbasin",
]
P3_MODELS = ["kalman_ar1", "kalman_own_ridge", "kalman_corr_top1"]

KEY = ["name", "issue_date", "target_date", "horizon"]


def load() -> pd.DataFrame:
    p2 = pd.read_csv(
        RESULTS / "phase2_baseline_predictions.csv",
        usecols=KEY + ["model", "target_std_units", "pred_std_units"],
    )
    p2 = p2[p2["model"].isin(P2_MODELS)].rename(
        columns={"target_std_units": "target", "pred_std_units": "pred"}
    )
    p3 = pd.read_csv(
        RESULTS / "phase3b_predictions.csv",
        usecols=KEY + ["model", "target", "pred"],
    )
    p3 = p3[p3["model"].isin(P3_MODELS)]
    df = pd.concat([p2, p3], ignore_index=True)

    # Per horizon, keep only (name, issue, target) keys present for every model.
    kept = []
    for h, sub in df.groupby("horizon"):
        counts = sub.groupby(["name", "issue_date", "target_date"])["model"].nunique()
        full = counts[counts == len(P2_MODELS) + len(P3_MODELS)].index
        sub = sub.set_index(["name", "issue_date", "target_date"])
        kept.append(sub.loc[sub.index.isin(full)].reset_index())
    out = pd.concat(kept, ignore_index=True)

    # Consistency: targets must agree across models within a key.
    chk = out.groupby(KEY)["target"].agg(["min", "max"])
    assert float((chk["max"] - chk["min"]).abs().max()) < 1e-9, "target mismatch"
    return out


def monthly_losses(df: pd.DataFrame, model: str, h: int) -> pd.Series:
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
        # stronger damped variant per horizon (lower pooled RMSE)
        rmse = {
            m: float(
                np.sqrt(
                    ((g["target"] - g["pred"]) ** 2).mean()
                )
            )
            for m, g in sub_h.groupby("model")
        }
        damped = min(
            ["damped_persistence_rho", "damped_persistence_reg"], key=lambda m: rmse[m]
        )
        ml_damped = monthly_losses(df, damped, h)
        for m in P2_MODELS + P3_MODELS:
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
        # key contrasts
        for a, b in [
            ("kalman_ar1", damped),
            ("kalman_ar1", "ridge_own_perbasin"),
            ("kalman_ar1", "ridge_own_lags"),
            ("kalman_own_ridge", "ridge_own_perbasin"),
            ("ridge_own_perbasin", damped),
        ]:
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
