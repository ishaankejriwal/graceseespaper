"""Conventional per-basin metrics (RMSE in cm, correlation, NSE) for the paper's
comparability section.

Other TWSA-forecasting studies report per-basin/grid RMSE in cm EWH, correlation
against GRACE, and NSE -- mostly on the full signal (seasonal cycle included).
This script computes those metrics for three of our systems so Sect. 5.4 can
bridge to published numbers instead of only explaining non-comparability.

Construction mirrors run_phase6_li_comparison.py exactly: per (basin, fold) the
climatology is fit on the train window only (fit_climatology), the train-window
residual std converts standardized errors to cm, and full-signal series are
frozen-climatology + anomaly. Because both observation and forecast share the
frozen climatology, anomaly-space and full-signal errors are identical rows;
correlation and NSE differ between the two spaces because the observed variance
does (the seasonal cycle is in the full signal only).

Models: damped persistence (the ladder's stronger variant: rho at lead 1,
regression at leads 2-6), the Kalman forecast, and the stacked-system two-seed
ensemble. Rows are asserted identical across models at each lead.

Outputs: results/conventional_metrics_perbasin.csv, results/conventional_metrics_summary.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.decompose import fit_climatology  # noqa: E402
from gracefc.evaluate import DEFAULT_FOLDS  # noqa: E402

RES = ROOT / "results"
HORIZONS = range(1, 7)
# Ladder convention (tab:ladder caption): rho-damping is the stronger variant at
# lead 1, regression damping at leads 2-6.
DAMPED_BY_H = {1: "damped_persistence_rho", **{h: "damped_persistence_reg" for h in range(2, 7)}}


def load_wide() -> pd.DataFrame:
    long = pd.read_csv(ROOT / "data" / "processed" / "basin_month_twsa_global.csv", parse_dates=["date"])
    return long.pivot(index="date", columns="name", values="twsa_cm")


def load_model_rows() -> pd.DataFrame:
    """Return rows: name, issue_date, target_date, fold, horizon, model, pred, target (std units)."""
    frames = []

    ph2 = pd.read_csv(RES / "phase2_baseline_predictions.csv", parse_dates=["issue_date", "target_date"])
    for h, m in DAMPED_BY_H.items():
        sub = ph2[(ph2["model"] == m) & (ph2["horizon"] == h)].copy()
        # phase2 carries both cm and std-unit columns; keep std units plus the cm
        # columns as an independent cross-check on this script's rescaling.
        sub = sub.rename(columns={"target": "target_cm_file", "pred": "pred_cm_file",
                                  "target_std_units": "target", "pred_std_units": "pred"})
        sub["model"] = "damped_persistence"
        frames.append(sub[["name", "issue_date", "target_date", "fold", "horizon",
                           "model", "pred", "target", "pred_cm_file", "target_cm_file"]])

    kal = pd.read_csv(RES / "kalman_predictions.csv", parse_dates=["issue_date", "target_date"])
    kal = kal[kal["model"] == "kalman_ar1"]
    frames.append(kal[["name", "issue_date", "target_date", "fold", "horizon", "model", "pred", "target"]])

    stack = pd.concat([
        pd.read_csv(RES / "phase8_lstm_combined_predictions.csv", parse_dates=["issue_date", "target_date"]),
        pd.read_csv(RES / "phase8b_lstm_h46_predictions.csv", parse_dates=["issue_date", "target_date"]),
    ])
    seeds = stack[stack["model"].isin(["lstmres_corr_top1_s0", "lstmres_corr_top1_s1"])]
    key = ["name", "issue_date", "target_date", "fold", "horizon"]
    ens = seeds.groupby(key, as_index=False).agg(pred=("pred", "mean"), target=("target", "mean"), n=("pred", "size"))
    assert (ens["n"] == 2).all(), "stack ensemble expects exactly two seeds per row"
    ens["model"] = "stacked_ens"
    frames.append(ens[key + ["model", "pred", "target"]])

    rows = pd.concat(frames, ignore_index=True)

    # Matched-row assertion: identical (name, issue_date, fold) sets per horizon.
    for h in HORIZONS:
        sets = {m: frozenset(map(tuple, g[["name", "issue_date", "fold"]].itertuples(index=False)))
                for m, g in rows[rows["horizon"] == h].groupby("model")}
        ref = next(iter(sets.values()))
        assert all(s == ref for s in sets.values()), f"row sets differ across models at h={h}"
    return rows


def attach_cm_and_full(rows: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    out = []
    fold_start = {f.name: f.test_start for f in DEFAULT_FOLDS}
    for (fold, name), grp in rows.groupby(["fold", "name"]):
        train = wide.loc[wide.index < fold_start[fold], name]
        fit = fit_climatology(train)
        resid_obs = wide[name] - fit.predict(wide.index)
        train_std = resid_obs[resid_obs.index < fold_start[fold]].std()
        g = grp.copy()
        g["train_std"] = train_std
        g["obs_anom_cm"] = g["target"].values * train_std
        g["pred_anom_cm"] = g["pred"].values * train_std
        g["clim_cm"] = np.asarray(fit.predict(pd.DatetimeIndex(g["target_date"])))
        g["obs_raw_cm"] = wide[name].reindex(pd.DatetimeIndex(g["target_date"])).values
        # Every scored target month must have a raw observation; a regenerated
        # basin table with holes would otherwise pass the nanmax check below.
        assert g["obs_raw_cm"].notna().all(), f"missing raw observation {name}/{fold}"
        # Consistency: frozen-climatology residual of the raw observation must
        # reproduce the archived standardized target to float precision.
        recon = g["obs_raw_cm"] - g["clim_cm"]
        assert np.nanmax(np.abs(recon - g["obs_anom_cm"])) < 1e-6, f"target reconstruction drift {name}/{fold}"
        out.append(g)
    df = pd.concat(out, ignore_index=True)

    # Independent cross-check against phase2's own cm columns.
    ph2 = df[df["model"] == "damped_persistence"].dropna(subset=["target_cm_file"])
    assert np.nanmax(np.abs(ph2["target_cm_file"] - ph2["obs_anom_cm"])) < 1e-6, "cm rescaling disagrees with phase2 file"
    assert np.nanmax(np.abs(ph2["pred_cm_file"] - ph2["pred_anom_cm"])) < 1e-6, "cm rescaling disagrees with phase2 file (pred)"
    return df


def per_basin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (model, h, name), g in df.groupby(["model", "horizon", "name"]):
        err = g["pred_anom_cm"] - g["obs_anom_cm"]
        obs_a, obs_f = g["obs_anom_cm"], g["obs_raw_cm"]
        pred_f = g["clim_cm"] + g["pred_anom_cm"]
        sse = float((err ** 2).sum())
        recs.append({
            "model": model, "horizon": h, "name": name, "n_months": len(g),
            "rmse_cm": float(np.sqrt((err ** 2).mean())),
            "cc_anom": float(np.corrcoef(g["pred_anom_cm"], obs_a)[0, 1]) if obs_a.std() > 0 else np.nan,
            "cc_full": float(np.corrcoef(pred_f, obs_f)[0, 1]) if obs_f.std() > 0 else np.nan,
            "nse_anom": 1.0 - sse / float(((obs_a - obs_a.mean()) ** 2).sum()),
            "nse_full": 1.0 - sse / float(((obs_f - obs_f.mean()) ** 2).sum()),
        })
    return pd.DataFrame(recs)


def main() -> None:
    wide = load_wide()
    rows = load_model_rows()
    df = attach_cm_and_full(rows, wide)
    per_basin = per_basin_metrics(df)
    per_basin.to_csv(RES / "conventional_metrics_perbasin.csv", index=False)

    qs = per_basin.groupby(["model", "horizon"]).agg(
        n_basins=("name", "size"),
        rmse_cm_med=("rmse_cm", "median"), rmse_cm_q25=("rmse_cm", lambda s: s.quantile(0.25)),
        rmse_cm_q75=("rmse_cm", lambda s: s.quantile(0.75)),
        cc_anom_med=("cc_anom", "median"), cc_full_med=("cc_full", "median"),
        nse_anom_med=("nse_anom", "median"), nse_full_med=("nse_full", "median"),
    ).reset_index()

    pooled = df.groupby(["model", "horizon"]).apply(
        lambda g: np.sqrt(((g["pred_anom_cm"] - g["obs_anom_cm"]) ** 2).mean()),
        include_groups=False).rename("pooled_rmse_cm").reset_index()
    summary = qs.merge(pooled, on=["model", "horizon"])
    summary.to_csv(RES / "conventional_metrics_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
