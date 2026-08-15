"""Phase 5: neighbor-fused Kalman filter vs the ridge-corrected neighbor model and placebos.

Information set matches Phase 3b exactly — the neighbor's month-t observation was already
available there through its filtered state, so in-filter fusion adds no new data, only a
better way to use it. Placebos refit (c, r_j) per random neighbor, keeping capacity matched.
"""
import argparse
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.features import pivot_wide  # noqa: E402
from gracefc.fusion import fusion_state_matrix  # noqa: E402
from gracefc.graphs import corr_topk, random_degree_matched  # noqa: E402
from gracefc.kalman import filtered_state_wide, fit_fold_params  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402

OUT_DIR = ROOT / "results"
PARAMS_CACHE = OUT_DIR / "kalman_fold_params.pkl"


def emit_rows(state: pd.DataFrame, rho: pd.Series, resid_wide: pd.DataFrame, label: str,
              fold, horizons: range) -> list[pd.DataFrame]:
    """Forecast rho^h * state at every issue date; keep rows whose target lands in the fold test."""
    rows = []
    names = list(state.columns)
    for h in horizons:
        prop = state.values * (rho[names].values[None, :] ** h)
        df = pd.DataFrame({
            "issue_date": np.repeat(state.index.values, len(names)),
            "name": np.tile(names, state.shape[0]),
            "pred": prop.ravel(),
        })
        tgt = resid_wide[names].shift(-h).stack(future_stack=True).rename("target").reset_index()
        tgt.columns = ["issue_date", "name", "target"]
        df = df.merge(tgt, on=["issue_date", "name"]).dropna(subset=["target", "pred"])
        df["target_date"] = df["issue_date"] + pd.DateOffset(months=h)
        te = df[(df["issue_date"] >= fold.test_start) & (df["issue_date"] <= fold.test_end)].copy()
        te["model"], te["fold"], te["horizon"] = label, fold.name, h
        rows.append(te)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--tag", default="phase5_fusion")
    args = ap.parse_args()

    long_df = pd.read_csv(ROOT / "data/processed/basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    wide = pivot_wide(long_df[long_df["name"].isin(keep)])
    cache = load_params_cache(PARAMS_CACHE, ROOT / "data/processed/basin_month_twsa_global.csv")
    print(f"sample: {wide.shape[1]} basins | fusion seeds={args.seeds}")

    out, placebo_monthly, placebo_basin = [], [], []
    horizons = range(1, 7)
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        params = cache.get(fold.name)
        if params is None:
            params = fit_fold_params(resid_wide, fold.test_start)
            cache[fold.name] = params
            save_params_cache(PARAMS_CACHE, cache, ROOT / "data/processed/basin_month_twsa_global.csv")
        rho = params.set_index("name")["rho"]
        fitted = resid_wide[params["name"]]

        train_src = fitted[fitted.index < fold.test_start]
        graph = corr_topk(train_src, 1)

        own = filtered_state_wide(fitted, params)
        out += emit_rows(own, rho, resid_wide, "kalman_ar1", fold, horizons)

        fused = fusion_state_matrix(fitted, params, graph, fold.test_start, mle_polish=False)
        out += emit_rows(fused, rho, resid_wide, "fusion_corr_top1", fold, horizons)
        fused_mle = fusion_state_matrix(fitted, params, graph, fold.test_start, mle_polish=True)
        out += emit_rows(fused_mle, rho, resid_wide, "fusion_corr_top1_mle", fold, horizons)

        # Placebos: random neighbor, same closed-form fit — aggregated losses only
        base = zlib.crc32(b"fusion_corr_top1") % 1_000_000
        for seed in range(args.seeds):
            g_rand = random_degree_matched(graph, base + seed)
            st = fusion_state_matrix(fitted, params, g_rand, fold.test_start, mle_polish=False)
            for te in emit_rows(st, rho, resid_wide, f"fusion_rand{seed}", fold, horizons):
                loss = (te["target"].values - te["pred"].values) ** 2
                ldf = pd.DataFrame({"target_date": te["target_date"].values,
                                    "name": te["name"].values, "loss": loss})
                monthly = ldf.groupby("target_date")["loss"].agg(["sum", "count"]).reset_index()
                monthly["model"], monthly["fold"], monthly["horizon"] = (
                    f"fusion_rand{seed}", fold.name, te["horizon"].iloc[0])
                placebo_monthly.append(monthly)
                basin = ldf.groupby("name")["loss"].agg(["sum", "count"]).reset_index()
                basin["model"], basin["fold"], basin["horizon"] = (
                    f"fusion_rand{seed}", fold.name, te["horizon"].iloc[0])
                placebo_basin.append(basin)
        print(f"{fold.name} done")

    pred_rows = pd.concat(out, ignore_index=True)
    plac_monthly = pd.concat(placebo_monthly, ignore_index=True)
    pred_rows.to_csv(OUT_DIR / f"{args.tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{args.tag}_placebo_monthly.csv", index=False)
    pd.concat(placebo_basin, ignore_index=True).to_csv(
        OUT_DIR / f"{args.tag}_placebo_basin.csv", index=False)

    # Summary: pooled RMSE, skill vs own filter, placebo rank, DM, bootstrap CI
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))
    rows = []
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {"model": model, "horizon": h,
               "rmse_std": rmse(grp["target"].values, grp["pred"].values), "n": len(grp)}
        if model.startswith("fusion_corr"):
            dist = plac_pooled.loc[plac_pooled.index.get_level_values(1) == h]["rmse"].values
            row["placebo_n"] = len(dist)
            row["placebo_beaten"] = int((row["rmse_std"] < dist).sum())
            row["p_rank"] = float((1 + (dist <= row["rmse_std"]).sum()) / (1 + len(dist)))
            stat, p = pooled_monthly_dm(pred_rows, model, "kalman_ar1", h)
            row["dm_vs_kalman"], row["dm_p"] = stat, p
            pt, lo, hi = block_bootstrap_skill_ci(pred_rows, model, "kalman_ar1", h)
            row["skill_ci_lo"], row["skill_ci_hi"] = lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    base_rmse = summary[summary["model"] == "kalman_ar1"].set_index("horizon")["rmse_std"]
    summary["skill_vs_kalman"] = 1 - (summary["rmse_std"] / summary["horizon"].map(base_rmse)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{args.tag}_summary.csv", index=False)
    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))


if __name__ == "__main__":
    main()
