"""Fit and score the benchmark forecasts on the standardized residual target.

Runs climatology, persistence, the two damped-persistence variants and the pooled,
index and per-basin lag ridges over the five folds (Sect. 3.3). Reads
data/processed/basin_month_twsa_global.csv, basin_meta.csv and indices.csv. Writes
results/phase2_baseline_predictions.csv (forecasts in cm and standardized units),
phase2_baseline_summary.csv, phase2_strata.csv and phase2_skill_by_stratum.csv.
scripts/build_paper_ladder.py turns the predictions into Table 2 and Appendix B.

Usage:
  python scripts/run_phase2_baselines.py
  GRACEFC_SOURCE=jpl python scripts/run_phase2_baselines.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.decompose import fit_climatology, deseasonalize  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS, per_basin_wins, run_baseline_ladder, summarize  # noqa: E402
from gracefc.runtime import load_sample, processed_dir, results_dir, shared_processed_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
SHARED_DATA = shared_processed_dir(ROOT)


def assign_snr_strata(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Stratum per basin from its fold-1 training residual std in cm; glaciated basins form their
    own."""
    train = wide[wide.index < DEFAULT_FOLDS[0].test_start]
    resid_std = {}
    for name in wide.columns:
        fit = fit_climatology(train[name])
        r = deseasonalize(train[name], fit)
        resid_std[name] = r.std()
    s = pd.Series(resid_std)
    glaciated = set(meta[meta["glaciated"]]["name"])
    # Fixed boundaries in cm; the lowest, 2 cm, is roughly the GRACE noise floor
    def stratum(name):
        if name in glaciated:
            return "glaciated"
        v = s[name]
        return "sub_noise" if v < 2 else "low" if v < 4 else "mid" if v < 8 else "high"
    return pd.Series({n: stratum(n) for n in wide.columns}, name="stratum")


def main() -> None:
    wide, meta, keep = load_sample(DATA)
    print(f"hydrology sample: {wide.shape[1]} basins, {wide.shape[0]} months on gapless index")

    strata = assign_snr_strata(wide, meta)
    strata.to_csv(OUT_DIR / "phase2_strata.csv")
    print(f"strata: {strata.value_counts().to_dict()}")

    indices = pd.read_csv(SHARED_DATA / "indices.csv", parse_dates=["date"]).set_index("date")
    # AMM and PDO end before the test window does; dropping them keeps every test month
    indices = indices.drop(columns=[c for c in ("amm", "pdo") if c in indices], errors="ignore")

    pred_rows = run_baseline_ladder(wide, indices=indices, standardize=True)
    pred_rows["stratum"] = pred_rows["name"].map(strata)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_rows.to_csv(OUT_DIR / "phase2_baseline_predictions.csv", index=False)

    summary = summarize(pred_rows)
    summary.to_csv(OUT_DIR / "phase2_baseline_summary.csv", index=False)
    print("\nPooled metrics by horizon (standardized fit, reported in cm and std units):")
    print(summary.to_string(index=False))

    print("\nPer-basin wins: ridge_own_lags vs damped_persistence_reg")
    print(per_basin_wins(pred_rows, "ridge_own_lags", "damped_persistence_reg").to_string(index=False))

    # The sub_noise stratum has little skill to find, so the comparison is also reported per stratum
    print("\nSkill vs damped persistence by stratum (rmse_std ratio, ridge_own_lags):")
    rows = []
    for (stratum, h), grp in pred_rows.groupby(["stratum", "horizon"]):
        ridge = grp[grp["model"] == "ridge_own_lags"]
        damped = grp[grp["model"] == "damped_persistence_reg"]
        if len(ridge) and len(damped):
            from gracefc.models import rmse
            skill = 1 - (rmse(ridge["target_std_units"].values, ridge["pred_std_units"].values)
                         / rmse(damped["target_std_units"].values, damped["pred_std_units"].values)) ** 2
            rows.append({"stratum": stratum, "horizon": h, "skill": round(skill, 4), "n": len(ridge)})
    strat_df = pd.DataFrame(rows).pivot(index="stratum", columns="horizon", values="skill")
    strat_df.to_csv(OUT_DIR / "phase2_skill_by_stratum.csv")
    print(strat_df.to_string())


if __name__ == "__main__":
    main()
