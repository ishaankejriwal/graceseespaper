"""Phase 2: baseline ladder on the deseasonalized target — the go/no-go experiment."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.decompose import fit_climatology, deseasonalize  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS, per_basin_wins, run_baseline_ladder, summarize  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402

OUT_DIR = ROOT / "results"


def assign_snr_strata(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Freeze SNR strata from the fold-1 training window; glaciated basins get their own stratum."""
    train = wide[wide.index < DEFAULT_FOLDS[0].test_start]
    resid_std = {}
    for name in wide.columns:
        fit = fit_climatology(train[name])
        r = deseasonalize(train[name], fit)
        resid_std[name] = r.std()
    s = pd.Series(resid_std)
    glaciated = set(meta[meta["glaciated"]]["name"])
    # Boundaries follow the physical GRACE noise floor (~2 cm), not arbitrary terciles
    def stratum(name):
        if name in glaciated:
            return "glaciated"
        v = s[name]
        return "sub_noise" if v < 2 else "low" if v < 4 else "mid" if v < 8 else "high"
    return pd.Series({n: stratum(n) for n in wide.columns}, name="stratum")


def main() -> None:
    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    print(f"hydrology sample: {wide.shape[1]} basins, {wide.shape[0]} months on gapless index")

    strata = assign_snr_strata(wide, meta)
    strata.to_csv(OUT_DIR / "phase2_strata.csv")
    print(f"strata: {strata.value_counts().to_dict()}")

    indices = pd.read_csv(ROOT / "data/processed/indices.csv", parse_dates=["date"]).set_index("date")
    # AMM/PDO go stale before the record end; drop them so the index ladder covers every test month
    indices = indices.drop(columns=[c for c in ("amm", "pdo") if c in indices], errors="ignore")

    pred_rows = run_baseline_ladder(wide, indices=indices, standardize=True)
    pred_rows["stratum"] = pred_rows["name"].map(strata)
    OUT_DIR.mkdir(exist_ok=True)
    pred_rows.to_csv(OUT_DIR / "phase2_baseline_predictions.csv", index=False)

    summary = summarize(pred_rows)
    summary.to_csv(OUT_DIR / "phase2_baseline_summary.csv", index=False)
    print("\nPooled metrics by horizon (standardized fit, reported in cm and std units):")
    print(summary.to_string(index=False))

    print("\nPer-basin wins: ridge_own_lags vs damped_persistence_reg")
    print(per_basin_wins(pred_rows, "ridge_own_lags", "damped_persistence_reg").to_string(index=False))

    # The go/no-go must hold outside the sub-noise stratum, which shows ~0 skill by construction
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
