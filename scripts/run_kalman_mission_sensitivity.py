"""Mission-split observation-noise sensitivity: does a separate r for GRACE-FO move the headline?

The published Kalman benchmark fits one observation-noise variance per basin-fold on a record
that spans GRACE (Apr 2002 - Jun 2017) and GRACE-FO (Jun 2018 onward), while every scored month
is GRACE-FO. This runner refits the same filter with two observation-noise variances split at
2018-01, on the same folds, the same deseasonalization, the same keep-list and the same forecast
rule, and reports what changes.

Contrasts:
  kalman_mission vs kalman_ar1   -> the effect of the extra parameter (the question asked)
  kalman_mission vs damped ref   -> the headline margin under the two-r variant
  kalman_ar1     vs damped ref   -> the published margin, recomputed here as a control

Both Kalman variants are generated inside this run so the pair shares every upstream choice;
the regenerated one-r forecasts are then checked against results/kalman_predictions.csv, and
every pairing goes through the row-set assertion in gracefc.stats.

Inputs : data/processed/basin_month_twsa_global.csv, data/processed/basin_meta.csv
         results/kalman_predictions.csv, results/phase2_baseline_predictions.csv
Outputs: results/kalman_mission_predictions.csv
         results/kalman_mission_params.csv
         results/kalman_mission_summary.csv
         results/kalman_mission_perbasin_h1.csv

Run: .venv/Scripts/python.exe scripts/run_kalman_mission_sensitivity.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.kalman_mission import MIN_FO_OBS, MISSION_SPLIT, mission_predictions  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.runtime import processed_dir, results_dir  # noqa: E402
from gracefc.stats import (  # noqa: E402
    _paired_losses,
    block_bootstrap_skill_ci,
    per_basin_dm_fdr,
    pooled_monthly_dm,
)

RESULTS = results_dir(ROOT)
DATA = processed_dir(ROOT)
COLS = ["name", "target_date", "horizon", "model", "target", "pred"]
DAMPED_VARIANTS = ["damped_persistence_rho", "damped_persistence_reg"]


def attach_and_clip(df: pd.DataFrame, resid_wide: pd.DataFrame, fold, model: str,
                    h: int) -> pd.DataFrame:
    """Join observed targets and clip to the fold's ISSUE window, matching evaluate.split_fold."""
    tgt = resid_wide.stack().rename("target").reset_index()
    tgt.columns = ["target_date", "name", "target"]
    out = df.merge(tgt, on=["target_date", "name"], how="inner").dropna(subset=["target"])
    out = out[(out["issue_date"] >= fold.test_start) & (out["issue_date"] <= fold.test_end)]
    return out.assign(model=model, fold=fold.name, horizon=h)


def fit_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_df = pd.read_csv(DATA / "basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(DATA / "basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])

    two_rows, one_rows, param_rows = [], [], []
    for fold in DEFAULT_FOLDS:
        t0 = time.time()
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        preds_two, preds_one, params = mission_predictions(
            resid_wide, fold.test_start, range(1, 7), progress=fold.name)
        for h in preds_two:
            two_rows.append(attach_and_clip(preds_two[h], resid_wide, fold, "kalman_mission", h))
            one_rows.append(attach_and_clip(preds_one[h], resid_wide, fold, "kalman_ar1_refit", h))
        param_rows.append(params.assign(fold=fold.name))
        print(f"fold {fold.name} done in {time.time() - t0:.0f}s "
              f"({len(params)} basins)", flush=True)
    return (pd.concat(two_rows, ignore_index=True),
            pd.concat(one_rows, ignore_index=True),
            pd.concat(param_rows, ignore_index=True))


def check_against_published(one: pd.DataFrame) -> float:
    """The regenerated one-r forecasts must reproduce the published file to float precision."""
    pub = pd.read_csv(RESULTS / "kalman_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    key = ["name", "issue_date", "target_date", "horizon"]
    j = pub[key + ["pred", "target"]].merge(
        one[key + ["pred", "target"]], on=key, suffixes=("_pub", "_new"), validate="one_to_one")
    if len(j) != len(pub) or len(j) != len(one):
        raise AssertionError(
            f"row sets differ from kalman_predictions.csv: published {len(pub)}, "
            f"regenerated {len(one)}, join {len(j)}")
    if not np.allclose(j["target_pub"], j["target_new"], atol=1e-9):
        raise AssertionError("targets differ from kalman_predictions.csv")
    return float((j["pred_pub"] - j["pred_new"]).abs().max())


def load_pool(mission: pd.DataFrame) -> pd.DataFrame:
    kal = pd.read_csv(RESULTS / "kalman_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    ph2 = pd.read_csv(RESULTS / "phase2_baseline_predictions.csv",
                      parse_dates=["issue_date", "target_date"])
    damped = ph2[ph2["model"].isin(DAMPED_VARIANTS)].drop(columns=["target", "pred"]).rename(
        columns={"target_std_units": "target", "pred_std_units": "pred"})
    return pd.concat([mission[COLS], kal[COLS], damped[COLS]], ignore_index=True)


def contrast(pool: pd.DataFrame, a: str, b: str, h: int, label: str) -> dict:
    point, lo, hi = block_bootstrap_skill_ci(pool, a, b, h)
    stat, p = pooled_monthly_dm(pool, a, b, h)
    sub = pool[pool["horizon"] == h]
    sa, sb = sub[sub["model"] == a], sub[sub["model"] == b]
    return {
        "component": label, "challenger": a, "reference": b, "horizon": h,
        "n": len(sa),
        "rmse_a": rmse(sa["target"].values, sa["pred"].values),
        "rmse_b": rmse(sb["target"].values, sb["pred"].values),
        "skill_pct": 100 * point, "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
        "dm_stat": stat, "dm_p": p,
    }


def main() -> None:
    mission, one, params = fit_all()
    mission.to_csv(RESULTS / "kalman_mission_predictions.csv", index=False)

    max_pred_diff = check_against_published(one)
    print(f"\nregenerated one-r vs published kalman_predictions.csv: "
          f"max |dpred| = {max_pred_diff:.3e}", flush=True)

    params["r_ratio"] = params["r_fo"] / params["r_grace"]
    params["r_ratio_raw"] = params["r_fo_raw"] / params["r_grace_raw"]
    cols = ["name", "fold", "n_train_obs", "n_fo_train", "rho", "q", "r_grace", "r_fo",
            "r_ratio", "r_fo_fallback", "r_grace_at_boundary", "r_fo_at_boundary",
            "two_converged", "rho_one", "q_one", "r_one", "r_one_at_boundary",
            "one_converged", "loglik_one", "loglik_two", "lr_stat", "lr_p",
            "rho_two_raw", "q_two_raw", "r_grace_raw", "r_fo_raw", "r_ratio_raw",
            "loglik_two_raw", "lr_stat_raw", "lr_p_raw"]
    params[cols].to_csv(RESULTS / "kalman_mission_params.csv", index=False)

    pool = load_pool(mission)
    # Every contrast must run on identical rows; assert before scoring anything
    for h in range(1, 7):
        sub = pool[pool["horizon"] == h]
        for b in ["kalman_ar1"] + DAMPED_VARIANTS:
            _paired_losses(sub, "kalman_mission", b)

    rows = []
    for h in range(1, 7):
        sub = pool[pool["horizon"] == h]
        # Stronger damped variant per lead, exactly as the paper ladder chooses it
        r_damped = {m: rmse(sub[sub["model"] == m]["target"].values,
                            sub[sub["model"] == m]["pred"].values) for m in DAMPED_VARIANTS}
        ref = min(DAMPED_VARIANTS, key=lambda m: r_damped[m])
        rows.append(contrast(pool, "kalman_mission", "kalman_ar1", h, "mission_split_term"))
        rows.append(contrast(pool, "kalman_mission", ref, h, "mission_vs_damped"))
        rows.append(contrast(pool, "kalman_ar1", ref, h, "published_vs_damped"))
        rows[-2]["damped_ref"] = ref
        rows[-1]["damped_ref"] = ref
    summary = pd.DataFrame(rows)

    fdr = per_basin_dm_fdr(pool, "kalman_mission", "kalman_ar1", 1)
    fdr.to_csv(RESULTS / "kalman_mission_perbasin_h1.csv", index=False)
    helped = int((fdr["significant"] & fdr["a_better"]).sum())
    hurt = int((fdr["significant"] & ~fdr["a_better"]).sum())

    fitted = params[~params["r_fo_fallback"]]
    ratio = fitted["r_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    diag = {
        "n_fits": len(params),
        "n_fallback_r_fo": int(params["r_fo_fallback"].sum()),
        "n_r_grace_at_boundary": int(fitted["r_grace_at_boundary"].sum()),
        "n_r_fo_at_boundary": int(fitted["r_fo_at_boundary"].sum()),
        "n_r_one_at_boundary": int(params["r_one_at_boundary"].sum()),
        "n_two_not_converged": int((~fitted["two_converged"]).sum()),
        "n_one_not_converged": int((~params["one_converged"]).sum()),
        "ratio_n": len(ratio),
        "ratio_median": float(ratio.median()),
        "ratio_q25": float(ratio.quantile(0.25)),
        "ratio_q75": float(ratio.quantile(0.75)),
        "ratio_iqr": float(ratio.quantile(0.75) - ratio.quantile(0.25)),
        "ratio_frac_lt_1": float((ratio < 1).mean()),
        "lr_frac_p_lt_05": float((fitted["lr_p"] < 0.05).mean()),
        "perbasin_h1_helped_bh": helped,
        "perbasin_h1_hurt_bh": hurt,
        "perbasin_h1_n_tested": len(fdr),
        "max_abs_pred_diff_vs_published": max_pred_diff,
        "mission_split": str(MISSION_SPLIT.date()),
        "min_fo_obs": MIN_FO_OBS,
    }
    summary = pd.concat([
        summary,
        pd.DataFrame([{"component": "diagnostic", "stat_name": k, "stat_value": v}
                      for k, v in diag.items()]),
    ], ignore_index=True)
    summary.to_csv(RESULTS / "kalman_mission_summary.csv", index=False)

    print()
    print(summary[summary["component"] != "diagnostic"].drop(
        columns=["stat_name", "stat_value"]).round(4).to_string(index=False))
    print()
    for k, v in diag.items():
        print(f"{k:>34}: {v}")
    print()
    print("r_fo/r_grace by fold (fitted folds only):")
    print(fitted.groupby("fold")["r_ratio"].describe().round(4).to_string())
    print()
    print("boundary + fallback counts by fold:")
    print(params.groupby("fold")[["r_fo_fallback", "r_grace_at_boundary",
                                  "r_fo_at_boundary", "r_one_at_boundary"]].sum().to_string())


if __name__ == "__main__":
    main()
