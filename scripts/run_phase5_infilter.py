"""Phase 5: a neighbor entering the Kalman filter itself, vs the own filter and random-pair placebos.

  --model fusion    single latent state, the neighbor as a second noisy sensor of it
                    (closed-form fit plus an MLE-polished twin; see gracefc.fusion)
  --model coupled   two latent states with correlated process noise, one coupling
                    parameter fitted by MLE (see gracefc.coupled); also writes the
                    fitted coupling per basin and fold

Information set matches Phase 3b exactly — the neighbor's month-t observation was already
available there through its filtered state, so in-filter use adds no new data, only a
different way to use it. Placebos refit per random neighbor, keeping capacity matched, and
keep aggregated losses only.
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.coupled import coupled_state_matrix  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold  # noqa: E402
from gracefc.fusion import fusion_state_matrix  # noqa: E402
from gracefc.graphs import corr_topk, random_degree_matched  # noqa: E402
from gracefc.kalman import filtered_state_wide, fit_fold_params  # noqa: E402
from gracefc.models import rmse  # noqa: E402
from gracefc.stats import block_bootstrap_skill_ci, pooled_monthly_dm  # noqa: E402
from gracefc.cache import load_params_cache, save_params_cache  # noqa: E402
from gracefc.runtime import load_sample, processed_dir, results_dir  # noqa: E402

OUT_DIR = results_dir(ROOT)
DATA = processed_dir(ROOT)
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


def real_arms(model: str, fitted, params, graph, test_start) -> tuple[dict, pd.Series | None]:
    """{label: state matrix} for the real arms, plus the coupled model's fitted c per basin."""
    if model == "fusion":
        return {
            "fusion_corr_top1": fusion_state_matrix(fitted, params, graph, test_start, mle_polish=False),
            "fusion_corr_top1_mle": fusion_state_matrix(fitted, params, graph, test_start, mle_polish=True),
        }, None
    state, cs = coupled_state_matrix(fitted, params, graph, test_start)
    return {"coupled_corr_top1": state}, cs


def placebo_state(model: str, fitted, params, graph, test_start) -> pd.DataFrame:
    if model == "fusion":
        return fusion_state_matrix(fitted, params, graph, test_start, mle_polish=False)
    return coupled_state_matrix(fitted, params, graph, test_start)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fusion", "coupled"), required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--tag", default=None, help="default phase5_<model>")
    args = ap.parse_args()
    model = args.model
    tag = args.tag or f"phase5_{model}"

    wide, _, _ = load_sample(DATA)
    cache = load_params_cache(PARAMS_CACHE, DATA / "basin_month_twsa_global.csv")
    print(f"sample: {wide.shape[1]} basins | {model} seeds={args.seeds}", flush=True)

    out, placebo_monthly, placebo_basin, coupling_rows = [], [], [], []
    horizons = range(1, 7)
    for fold in DEFAULT_FOLDS:
        resid_raw, train_std = deseasonalize_fold(wide, fold)
        resid_wide = resid_raw / train_std
        params = cache.get(fold.name)
        if params is None:
            params = fit_fold_params(resid_wide, fold.test_start)
            cache[fold.name] = params
            save_params_cache(PARAMS_CACHE, cache, DATA / "basin_month_twsa_global.csv")
        rho = params.set_index("name")["rho"]
        fitted = resid_wide[params["name"]]
        graph = corr_topk(fitted[fitted.index < fold.test_start], 1)

        own = filtered_state_wide(fitted, params)
        out += emit_rows(own, rho, resid_wide, "kalman_ar1", fold, horizons)

        arms, cs = real_arms(model, fitted, params, graph, fold.test_start)
        for label, state in arms.items():
            out += emit_rows(state, rho, resid_wide, label, fold, horizons)
        if cs is not None:
            coupling_rows.append(pd.DataFrame({"name": cs.index, "coupling": cs.values, "fold": fold.name}))

        base = zlib.crc32(f"{model}_corr_top1".encode()) % 1_000_000
        for seed in range(args.seeds):
            g_rand = random_degree_matched(graph, base + seed)
            st = placebo_state(model, fitted, params, g_rand, fold.test_start)
            label = f"{model}_rand{seed}"
            for te in emit_rows(st, rho, resid_wide, label, fold, horizons):
                loss = (te["target"].values - te["pred"].values) ** 2
                ldf = pd.DataFrame({"target_date": te["target_date"].values,
                                    "name": te["name"].values, "loss": loss})
                for key, acc in (("target_date", placebo_monthly), ("name", placebo_basin)):
                    agg = ldf.groupby(key)["loss"].agg(["sum", "count"]).reset_index()
                    agg["model"], agg["fold"], agg["horizon"] = label, fold.name, te["horizon"].iloc[0]
                    acc.append(agg)
        print(f"{fold.name} done", flush=True)

    pred_rows = pd.concat(out, ignore_index=True)
    plac_monthly = pd.concat(placebo_monthly, ignore_index=True)
    pred_rows.to_csv(OUT_DIR / f"{tag}_predictions.csv", index=False)
    plac_monthly.to_csv(OUT_DIR / f"{tag}_placebo_monthly.csv", index=False)
    pd.concat(placebo_basin, ignore_index=True).to_csv(OUT_DIR / f"{tag}_placebo_basin.csv", index=False)
    if coupling_rows:
        pd.concat(coupling_rows, ignore_index=True).to_csv(OUT_DIR / f"{tag}_coupling.csv", index=False)

    # Summary: pooled RMSE, skill vs own filter, placebo rank, DM, bootstrap CI
    plac_pooled = (plac_monthly.groupby(["model", "horizon"])[["sum", "count"]].sum()
                   .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))
    rows = []
    for (m, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {"model": m, "horizon": h,
               "rmse_std": rmse(grp["target"].values, grp["pred"].values), "n": len(grp)}
        if m.startswith(f"{model}_corr"):
            dist = plac_pooled.loc[plac_pooled.index.get_level_values(1) == h]["rmse"].values
            row["placebo_n"] = len(dist)
            row["placebo_beaten"] = int((row["rmse_std"] < dist).sum())
            row["p_rank"] = float((1 + (dist <= row["rmse_std"]).sum()) / (1 + len(dist)))
            stat, p = pooled_monthly_dm(pred_rows, m, "kalman_ar1", h)
            row["dm_vs_kalman"], row["dm_p"] = stat, p
            pt, lo, hi = block_bootstrap_skill_ci(pred_rows, m, "kalman_ar1", h)
            row["skill_ci_lo"], row["skill_ci_hi"] = lo, hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    base_rmse = summary[summary["model"] == "kalman_ar1"].set_index("horizon")["rmse_std"]
    summary["skill_vs_kalman"] = 1 - (summary["rmse_std"] / summary["horizon"].map(base_rmse)) ** 2
    summary = summary.sort_values(["horizon", "rmse_std"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / f"{tag}_summary.csv", index=False)
    for h in sorted(summary["horizon"].unique()):
        print(f"\n=== horizon {h} ===")
        print(summary[summary["horizon"] == h].to_string(index=False))


if __name__ == "__main__":
    main()
