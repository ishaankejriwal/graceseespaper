"""Splice of one of our forecasts at short leads with GRACE-FCast at the longer leads.

Not used in the manuscript. Each hybrid takes a model of
ours at the short leads and the full GRACE-FCast product (li_lstm_full) at the
remaining leads. All models are scored on the matched rows of
results/phase6_li_comparison_predictions.csv (on JPL, the fully covered basin set
only), with skill relative to damped_persistence_rho on the same rows.

  hybrid_kalman_li     kalman_corr_top1 at h1-2, li_lstm_full at h3-6
  hybrid_era5_li       ridge_corr_top1_era5 at h1-2, li_lstm_full at h3-6
  hybrid_kalman_li_h1  kalman_corr_top1 at h1 only, li_lstm_full at h2-6
  hybrid_era5_li_h1    ridge_corr_top1_era5 at h1 only, li_lstm_full at h2-6

The script needs kalman_corr_top1 rows in the comparison predictions file. The
current run_phase6_li_comparison.py does not write that model, so with the current
CSR file the target check below raises (max |diff| = nan); the JPL file, from an
earlier revision, still has those rows. The archived hybrid outputs were produced
from earlier comparison files.

Inputs: results/phase6_li_comparison_predictions.csv, phase6_era5_predictions.csv.
Outputs: results/phase6_hybrid_summary.csv, phase6_hybrid_headline.csv.

Usage:
  python scripts/run_phase6_hybrid.py
  GRACEFC_SOURCE=jpl python scripts/run_phase6_hybrid.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, diebold_mariano, pooled_monthly_dm  # noqa: E402
from gracefc.runtime import results_dir, source  # noqa: E402

OUT_DIR = results_dir(ROOT)
HORIZONS = range(1, 7)
SHORT_H = (1, 2)
HYBRIDS = {
    # name -> (short-lead model, long-lead model, horizons taken from the short model)
    "hybrid_kalman_li": ("kalman_corr_top1", "li_lstm_full", (1, 2)),
    "hybrid_era5_li": ("ridge_corr_top1_era5", "li_lstm_full", (1, 2)),
    "hybrid_kalman_li_h1": ("kalman_corr_top1", "li_lstm_full", (1,)),
    "hybrid_era5_li_h1": ("ridge_corr_top1_era5", "li_lstm_full", (1,)),
}
SOLO_MODELS = ["kalman_corr_top1", "ridge_corr_top1_era5", "li_lstm_full",
               "li_lstm_nonseas", "damped_persistence_rho"]
COLS = ["name", "issue_date", "target_date", "target", "pred", "model", "fold", "horizon"]


def pooled_all_horizon_dm(rows: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float]:
    """DM test on monthly losses averaged across basins and horizons (Newey-West lag 5).

    Descriptive only. Months at the edges of the record carry a partial mix of
    horizons, and for a hybrid against one of its components the loss differential is
    zero at the shared horizons, which dilutes the test. The per-horizon DM tests are
    the ones to read."""
    a = rows[rows["model"] == model_a]
    b = rows[rows["model"] == model_b]
    la = a.assign(loss=(a["target"] - a["pred"]) ** 2).groupby("target_date")["loss"].mean()
    lb = b.assign(loss=(b["target"] - b["pred"]) ** 2).groupby("target_date")["loss"].mean()
    common = la.index.intersection(lb.index)
    return diebold_mariano(la[common].values, lb[common].values, horizon=max(HORIZONS))


def main() -> None:
    matched = pd.read_csv(OUT_DIR / "phase6_li_comparison_predictions.csv",
                          parse_dates=["issue_date", "target_date"])
    if source() == "jpl":
        if "joint_full_cells" not in matched:
            raise ValueError("JPL GRACE-FCast comparison file lacks the joint_full_cells column")
        matched = matched[matched["joint_full_cells"]].copy()

    era5 = pd.read_csv(OUT_DIR / "phase6_era5_predictions.csv",
                       parse_dates=["issue_date", "target_date"])
    era5 = era5[era5["model"] == "ridge_corr_top1_era5"]

    # Consistency guard: both files must score the same standardized target
    chk = matched[matched["model"] == "kalman_corr_top1"].merge(
        era5, on=["name", "target_date", "horizon"], suffixes=("_m", "_e"))
    gap = (chk["target_m"] - chk["target_e"]).abs().max()
    if not (gap < 1e-6):
        raise AssertionError(f"target mismatch matched vs era5: max |diff| = {gap}")

    # Restrict ERA5 rows to the matched keys so every model scores identical rows.
    # The ERA5 run covers h1-3; the splice needs h1-2 fully covered.
    keys = matched[["horizon", "name", "target_date"]].drop_duplicates()
    era5 = era5.merge(keys, on=["horizon", "name", "target_date"])
    n_short = len(keys[keys["horizon"].isin(SHORT_H)])
    if len(era5[era5["horizon"].isin(SHORT_H)]) != n_short:
        raise AssertionError("era5 does not fully cover matched keys at h1-2")
    # Any horizon the ERA5 solo model reports must be fully matched, not partial
    per_h = era5.groupby("horizon").size()
    full = keys.groupby("horizon").size()
    if not (per_h == full[per_h.index]).all():
        raise AssertionError(f"era5 partially covers a reported horizon:\n{per_h}")

    rows = pd.concat([matched[COLS], era5[COLS]], ignore_index=True)

    # Splice: relabel component rows so the hybrid scores as one model
    for hyb, (short_model, long_model, short_h) in HYBRIDS.items():
        part = rows[((rows["model"] == short_model) & rows["horizon"].isin(short_h))
                    | ((rows["model"] == long_model) & ~rows["horizon"].isin(short_h))].copy()
        part["model"] = hyb
        rows = pd.concat([rows, part], ignore_index=True)

    models = SOLO_MODELS + list(HYBRIDS)

    # Per-horizon and pooled RMSE; skill relative to damped persistence on the same rows
    summary = []
    for model in models:
        sub = rows[rows["model"] == model]
        have = set(sub["horizon"])
        for h in HORIZONS:
            if h not in have:
                continue
            g = sub[sub["horizon"] == h]
            summary.append({"model": model, "horizon": h,
                            "rmse_std": rmse(g["target"].values, g["pred"].values),
                            "n": len(g), "n_months": g["target_date"].nunique(),
                            "n_basins": g["name"].nunique()})
        # Pooled row only for models covering every horizon (ERA5 solo stops at h3)
        if have >= set(HORIZONS):
            summary.append({"model": model, "horizon": "all",
                            "rmse_std": rmse(sub["target"].values, sub["pred"].values),
                            "n": len(sub), "n_months": sub["target_date"].nunique(),
                            "n_basins": sub["name"].nunique()})
    summary = pd.DataFrame(summary)
    damped = summary[summary["model"] == "damped_persistence_rho"].set_index("horizon")["rmse_std"]
    summary["skill_vs_damped"] = 1 - (summary["rmse_std"] / summary["horizon"].map(damped)) ** 2
    summary = summary.sort_values(
        ["horizon", "rmse_std"],
        key=lambda s: s.map(str) if s.name == "horizon" else s).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "phase6_hybrid_summary.csv", index=False)

    # Each hybrid against each component and against damped persistence, per horizon
    # and pooled over horizons. A pair is tested only at the horizons where the hybrid
    # differs from the reference.
    headline = []
    pairs = []
    for hyb, (short_model, long_model, short_h) in HYBRIDS.items():
        pairs.append((hyb, short_model, set(HORIZONS) - set(short_h)))
        pairs.append((hyb, long_model, set(short_h)))
        pairs.append((hyb, "damped_persistence_rho", set(HORIZONS)))
    for model_a, model_b, h_test in pairs:
        h_a = set(rows.loc[rows["model"] == model_a, "horizon"])
        h_b = set(rows.loc[rows["model"] == model_b, "horizon"])
        for h in sorted(h_a & h_b & h_test):
            skill, lo, hi = block_bootstrap_skill_ci(rows, model_a, model_b, h)
            dm, p = pooled_monthly_dm(rows, model_a, model_b, h)
            headline.append({"model": model_a, "vs": model_b, "horizon": h,
                             "skill": skill, "ci_lo": lo, "ci_hi": hi,
                             "dm_stat": dm, "dm_p": p})
        if not ((h_a & h_b) >= set(HORIZONS)):
            continue
        a = rows[rows["model"] == model_a]
        b = rows[rows["model"] == model_b]
        skill_all = 1 - (rmse(a["target"].values, a["pred"].values)
                         / rmse(b["target"].values, b["pred"].values)) ** 2
        dm, p = pooled_all_horizon_dm(rows, model_a, model_b)
        headline.append({"model": model_a, "vs": model_b, "horizon": "all",
                         "skill": skill_all, "ci_lo": None, "ci_hi": None,
                         "dm_stat": dm, "dm_p": p})
    pd.DataFrame(headline).to_csv(OUT_DIR / "phase6_hybrid_headline.csv", index=False)

    print(summary.to_string(index=False))
    print("\n===== headline =====")
    print(pd.DataFrame(headline).to_string(index=False))


if __name__ == "__main__":
    main()
