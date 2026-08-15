"""Rolling-origin evaluation harness: fold-safe decomposition, baseline ladder, leakage guards."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decompose import fit_climatology, deseasonalize
from .features import build_lag_table, interpolate_short_gaps, DEFAULT_LAGS
from .models import fit_ridge, per_basin_ridge_predict, predict_ridge, rmse


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive


def make_folds(spec: list[tuple[str, str, str]]) -> list[Fold]:
    return [Fold(n, pd.Timestamp(a), pd.Timestamp(b)) for n, a, b in spec]


# Test targets 2019-06 onward only: gap-killed lag windows make 2016 to early-2019 nearly
# empty (first fully usable post-gap issue is 2019-05), so earlier folds would test on ~nothing
DEFAULT_FOLDS = make_folds([
    ("f1", "2019-06-01", "2020-10-01"),
    ("f2", "2020-11-01", "2022-03-01"),
    ("f3", "2022-04-01", "2023-08-01"),
    ("f4", "2023-09-01", "2025-01-01"),
    ("f5", "2025-02-01", "2026-05-01"),
])


def deseasonalize_fold(wide: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.Series]:
    """Fit trend+seasonal per basin on pre-test data only; return residual matrix and train std."""
    train_wide = wide[wide.index < fold.test_start]
    resid = {}
    train_std = {}
    for name in wide.columns:
        fit = fit_climatology(train_wide[name])
        r = deseasonalize(wide[name], fit)
        resid[name] = r
        train_std[name] = r[r.index < fold.test_start].std()
    return pd.DataFrame(resid), pd.Series(train_std)


def assert_no_leakage(
    train_rows: pd.DataFrame, fold: Fold, lags: tuple[int, ...],
    test_rows: pd.DataFrame | None = None,
) -> None:
    # Training targets must be observed before the freeze date; test rows must be
    # issued at or after it, so no fitted transform ever sees post-issue data
    if len(train_rows) and train_rows["target_date"].max() >= fold.test_start:
        raise AssertionError(f"leakage: train target {train_rows['target_date'].max()} >= {fold.test_start}")
    if test_rows is not None and len(test_rows) and test_rows["issue_date"].min() < fold.test_start:
        raise AssertionError(f"leakage: test issue {test_rows['issue_date'].min()} < {fold.test_start}")


def split_fold(lag_df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Issue-date fold membership: every transform in this codebase is fit on
    observations strictly before test_start, so a model frozen there is only
    honest for forecasts ISSUED at or after test_start. Splitting on target
    date instead let leads 2-6 use transforms fit up to h-1 months after the
    issue date (audit 2026-08-13, P0-2). Test targets may extend past test_end
    by up to the horizon, as in any rolling-origin design; each fold's own
    train/test separation stays honest because training is capped at the
    freeze date."""
    train = lag_df[lag_df["target_date"] < fold.test_start]
    test = lag_df[(lag_df["issue_date"] >= fold.test_start) & (lag_df["issue_date"] <= fold.test_end)]
    return train, test


def damped_persistence_pred(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> np.ndarray:
    """Two AR(1) forms: 'regression' fits target~lag_0 per basin; 'rho_pow' decays lag-1 acf^h."""
    preds = np.empty(len(test))
    coef = {}
    for name, grp in train.groupby("name"):
        x, y = grp["lag_0"].values, grp["target"].values
        if mode == "regression":
            denom = float(np.dot(x, x))
            coef[name] = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        else:
            x1, y1 = grp["lag_1"].values, grp["lag_0"].values
            denom = float(np.dot(x1, x1))
            rho1 = float(np.dot(x1, y1) / denom) if denom > 0 else 0.0
            h = int(round((grp["target_date"].iloc[0] - grp["issue_date"].iloc[0]) / np.timedelta64(30, "D")))
            coef[name] = np.clip(rho1, -0.999, 0.999) ** max(h, 1)
    for i, row in enumerate(test.itertuples()):
        preds[i] = coef.get(row.name, 0.0) * row.lag_0
    return preds


def run_baseline_ladder(
    wide: pd.DataFrame,
    horizons: range = range(1, 7),
    folds: list[Fold] = DEFAULT_FOLDS,
    indices: pd.DataFrame | None = None,
    index_lags: tuple[int, ...] = (0, 1, 2),
    standardize: bool = False,
    fill_short_gaps: bool = True,
) -> pd.DataFrame:
    """All baselines x folds x horizons on the deseasonalized target. Returns tidy prediction rows."""
    out = []
    for fold in folds:
        resid_wide, train_std = deseasonalize_fold(wide, fold)
        if standardize:
            # Per-basin train-window scaling stops high-amplitude basins from owning the pooled loss
            resid_wide = resid_wide / train_std
        # Short gaps are filled on the feature side only; every model sees identical fills,
        # and anchor masking keeps every filled value causal at its issue date
        feat_wide, anchors = interpolate_short_gaps(resid_wide) if fill_short_gaps else (None, None)
        for h in horizons:
            lag_df = build_lag_table(resid_wide, horizon=h, feature_wide=feat_wide, anchors=anchors)
            train, test = split_fold(lag_df, fold)
            assert_no_leakage(train, fold, DEFAULT_LAGS, test_rows=test)
            if len(test) == 0 or len(train) == 0:
                continue
            own_cols = [f"lag_{k}" for k in DEFAULT_LAGS]

            preds = {
                "climatology_zero": np.zeros(len(test)),
                "persistence": test["lag_0"].values,
                "damped_persistence_reg": damped_persistence_pred(train, test, "regression"),
                "damped_persistence_rho": damped_persistence_pred(train, test, "rho_pow"),
            }
            ridge, scaler = fit_ridge(train, own_cols)
            preds["ridge_own_lags"] = predict_ridge(ridge, scaler, test, own_cols)
            # Strongest baseline per the Phase 2 audit: dodges the pooling penalty entirely
            preds["ridge_own_perbasin"] = per_basin_ridge_predict(train, test, own_cols)

            if indices is not None:
                # Climate indices join on issue date; only lags at or before issue are legal
                idx_feats = pd.concat(
                    {f"{c}_lag{k}": indices[c].shift(k) for c in indices.columns for k in index_lags},
                    axis=1,
                )
                idx_feats.columns = [f"{a}" for a, _ in idx_feats.columns] if isinstance(idx_feats.columns, pd.MultiIndex) else idx_feats.columns
                tr = train.merge(idx_feats, left_on="issue_date", right_index=True, how="left")
                te = test.merge(idx_feats, left_on="issue_date", right_index=True, how="left")
                idx_cols = [c for c in idx_feats.columns]
                keep_tr = tr.dropna(subset=own_cols + idx_cols + ["target"])
                keep_te = te.dropna(subset=own_cols + idx_cols)
                if len(keep_tr) and len(keep_te) == len(te):
                    r2, s2 = fit_ridge(keep_tr, own_cols + idx_cols)
                    preds["ridge_own_plus_indices"] = predict_ridge(r2, s2, keep_te, own_cols + idx_cols)

            for model, p in preds.items():
                df = test[["name", "issue_date", "target_date", "target"]].copy()
                df["pred"] = p
                df["model"] = model
                df["fold"] = fold.name
                df["horizon"] = h
                if standardize:
                    # Report in cm regardless: un-scale so results stay physically interpretable
                    scale = df["name"].map(train_std)
                    df["target"] = df["target"] * scale
                    df["pred"] = df["pred"] * scale
                    df["target_std_units"] = df["target"] / scale
                    df["pred_std_units"] = df["pred"] / scale
                out.append(df)
    return pd.concat(out, ignore_index=True)


def summarize(pred_rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled and per-model metrics by horizon; skill score vs the stronger damped persistence."""
    rows = []
    std_units = "target_std_units" in pred_rows.columns
    for (model, h), grp in pred_rows.groupby(["model", "horizon"]):
        row = {
            "model": model, "horizon": h,
            "rmse_cm": rmse(grp["target"].values, grp["pred"].values),
            "mae_cm": float(np.mean(np.abs(grp["target"] - grp["pred"]))),
            "r": float(np.corrcoef(grp["target"], grp["pred"])[0, 1]) if grp["pred"].std() > 0 else np.nan,
            "n": len(grp),
        }
        if std_units:
            row["rmse_std"] = rmse(grp["target_std_units"].values, grp["pred_std_units"].values)
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Skill relative to the best damped-persistence variant at each horizon, in the fit metric
    key = "rmse_std" if std_units else "rmse_cm"
    damped = summary[summary["model"].str.startswith("damped")].groupby("horizon")[key].min()
    summary["skill_vs_damped"] = 1 - (summary[key] / summary["horizon"].map(damped)) ** 2
    return summary.sort_values(["horizon", key]).reset_index(drop=True)


def per_basin_wins(pred_rows: pd.DataFrame, challenger: str, reference: str) -> pd.DataFrame:
    """Count basins where challenger beats reference on RMSE, by horizon."""
    rows = []
    for h, grp in pred_rows.groupby("horizon"):
        piv = {}
        for model in (challenger, reference):
            sub = grp[grp["model"] == model]
            piv[model] = sub.groupby("name").apply(
                lambda g: rmse(g["target"].values, g["pred"].values), include_groups=False
            )
        wins = (piv[challenger] < piv[reference]).sum()
        rows.append({"horizon": h, "challenger_wins": int(wins), "total": len(piv[reference])})
    return pd.DataFrame(rows)
