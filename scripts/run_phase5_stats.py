"""Phase 5: one consolidated significance table across every experiment that has run.

Every comparison is computed on the exact intersection of prediction rows (basin, issue,
target date, fold) so arms from different runs are never scored on different row sets.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.stats import block_bootstrap_skill_ci, diebold_mariano, per_basin_dm_fdr  # noqa: E402

OUT_DIR = ROOT / "results"
KEYS = ["name", "issue_date", "target_date", "fold", "horizon"]


def load(tag: str) -> pd.DataFrame | None:
    p = OUT_DIR / f"{tag}_predictions.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["issue_date", "target_date"])


def matched_compare(df_a, model_a, df_b, model_b, horizon) -> dict | None:
    """Skill, bootstrap CI, and DM for A vs B on their common rows only."""
    a = df_a[(df_a["model"] == model_a) & (df_a["horizon"] == horizon)][KEYS + ["target", "pred"]]
    b = df_b[(df_b["model"] == model_b) & (df_b["horizon"] == horizon)][KEYS + ["target", "pred"]]
    m = a.merge(b, on=KEYS, suffixes=("_a", "_b"))
    if len(m) < 100:
        return None
    # Monthly-mean loss equals pooled skill only when every month has the same basin count
    counts = m.groupby("target_date").size()
    assert counts.nunique() == 1, f"unbalanced months in {model_a} vs {model_b}: {counts.unique()}"
    la = m.assign(loss=(m["target_a"] - m["pred_a"]) ** 2).groupby("target_date")["loss"].mean()
    lb = m.assign(loss=(m["target_b"] - m["pred_b"]) ** 2).groupby("target_date")["loss"].mean()
    stat, p = diebold_mariano(la.values, lb.values, horizon=horizon)
    skill = 1 - la.mean() / lb.mean()
    rng = np.random.default_rng(0)
    n, block = len(la), 3
    n_blocks = int(np.ceil(n / block))
    draws = np.empty(2000)
    va, vb = la.values, lb.values
    for i in range(2000):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        draws[i] = 1 - va[idx].mean() / vb[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"model": model_a, "vs": model_b, "horizon": horizon, "n_rows": len(m),
            "n_months": n, "skill": skill, "ci_lo": lo, "ci_hi": hi,
            "dm_stat": stat, "dm_p": p}


def main() -> None:
    p3b = load("phase3b")
    predlag = load("phase5_predlag")
    fusion = load("phase5_fusion")
    coupled = load("phase5_coupled")
    nonlin = load("phase5_nonlinear")

    comparisons = []
    # Headline and its committed comparators, all horizons
    for h in range(1, 7):
        comparisons += [
            (p3b, "kalman_corr_top1", p3b, "kalman_own_ridge", h),
            (p3b, "kalman_corr_top1", p3b, "kalman_ar1", h),
        ]
        if predlag is not None:
            comparisons += [
                (predlag, "kalman_pred_lag1_top1", predlag, "kalman_own_ridge", h),
                (predlag, "kalman_pred_lag1_top1", p3b, "kalman_corr_top1", h),
            ]
        if fusion is not None:
            comparisons += [
                (fusion, "fusion_corr_top1", fusion, "kalman_ar1", h),
                (fusion, "fusion_corr_top1_mle", fusion, "kalman_ar1", h),
                (fusion, "fusion_corr_top1", p3b, "kalman_own_ridge", h),
                (fusion, "fusion_corr_top1", p3b, "kalman_corr_top1", h),
            ]
        if coupled is not None:
            comparisons += [
                (coupled, "coupled_corr_top1", coupled, "kalman_ar1", h),
                (coupled, "coupled_corr_top1", p3b, "kalman_own_ridge", h),
                (coupled, "coupled_corr_top1", p3b, "kalman_corr_top1", h),
            ]
    if nonlin is not None:
        for h in range(1, 4):
            comparisons += [
                (nonlin, "gbm_own", nonlin, "ridge_own", h),
                (nonlin, "gbm_corr_top1", nonlin, "ridge_corr_top1", h),
                (nonlin, "gbm_corr_top1", nonlin, "gbm_own", h),
                (nonlin, "ridge_corr_top1_2hop", nonlin, "ridge_corr_top1", h),
                (nonlin, "gbm_corr_top1_2hop", nonlin, "gbm_corr_top1", h),
                (nonlin, "mlp_corr_top1_s0", nonlin, "ridge_corr_top1", h),
                (nonlin, "mlp_own_s0", nonlin, "ridge_own", h),
            ]

    rows = []
    for df_a, ma, df_b, mb, h in comparisons:
        if df_a is None or df_b is None:
            continue
        r = matched_compare(df_a, ma, df_b, mb, h)
        if r is not None:
            rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "phase5_headline_table.csv", index=False)
    print(table.to_string(index=False))

    # Per-basin FDR at h1 for the two arms that could carry the paper
    fdr_frames = []
    fdr = per_basin_dm_fdr(p3b, "kalman_corr_top1", "kalman_own_ridge", horizon=1)
    fdr["comparison"] = "corr_top1_vs_own_ridge"
    fdr_frames.append(fdr)
    if fusion is not None:
        fdr2 = per_basin_dm_fdr(fusion, "fusion_corr_top1", "kalman_ar1", horizon=1)
        fdr2["comparison"] = "fusion_vs_kalman"
        fdr_frames.append(fdr2)
    if coupled is not None:
        fdr3 = per_basin_dm_fdr(coupled, "coupled_corr_top1", "kalman_ar1", horizon=1)
        fdr3["comparison"] = "coupled_vs_kalman"
        fdr_frames.append(fdr3)
    fdr_all = pd.concat(fdr_frames, ignore_index=True)
    fdr_all.to_csv(OUT_DIR / "phase5_perbasin_fdr_h1.csv", index=False)
    for comp, grp in fdr_all.groupby("comparison"):
        n_sig = int(grp["significant"].sum())
        n_better = int(grp["a_better"].sum())
        print(f"\n{comp}: {n_better}/{len(grp)} basins favor treatment, {n_sig} pass FDR q=0.10")


if __name__ == "__main__":
    main()
