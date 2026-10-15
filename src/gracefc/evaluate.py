"""Evaluation folds, the issue-date train/test split, the benchmark forecasts and scoring.

Each row of a lag table has an issue date (when the forecast is made) and a target date
(the month forecast). For a fold whose test window starts at T:

    test  = rows issued at or after T, up to the fold's test_end
    train = rows whose target date is before T

A row issued before T with a target after T belongs to neither set. Splitting on target
date instead would let a lead-h forecast use transforms fit up to h - 1 months after its
issue date; tests/test_regressions.py::test_split_fold_membership_invariants checks the
split. The trend, seasonal cycle, standardization and model coefficients are all fit on
the training window and held fixed over the test window.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .decompose import fit_climatology, deseasonalize
from .features import build_lag_table, interpolate_short_gaps, DEFAULT_LAGS
from .models import fit_ridge, per_basin_ridge_predict, predict_ridge, rmse


@dataclass(frozen=True)
class Fold:
    """Evaluation fold whose test issue months run from test_start through test_end."""
    name: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # inclusive


def make_folds(spec: list[tuple[str, str, str]]) -> list[Fold]:
    return [Fold(n, pd.Timestamp(a), pd.Timestamp(b)) for n, a, b in spec]


# Test issue months start in June 2019. Lag windows that span the 2017-2018 mission gap are
# incomplete (the first issue month after the gap with all lags observed is May 2019), so
# earlier folds would have almost no test rows.
DEFAULT_FOLDS = make_folds([
    ("f1", "2019-06-01", "2020-10-01"),
    ("f2", "2020-11-01", "2022-03-01"),
    ("f3", "2022-04-01", "2023-08-01"),
    ("f4", "2023-09-01", "2025-01-01"),
    ("f5", "2025-02-01", "2026-05-01"),
])


def deseasonalize_fold(wide: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.Series]:
    """Residuals from a per-basin trend and harmonic fit on months before the fold's test_start.

    Returns the residual matrix for the whole record and each basin's training-window
    residual standard deviation, the standardization constant.
    """
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
    """Raise AssertionError if a training target falls on or after test_start, or a test
    row was issued before it, so that no fitted transform uses data after an issue date."""
    if len(train_rows) and train_rows["target_date"].max() >= fold.test_start:
        raise AssertionError(f"leakage: train target {train_rows['target_date'].max()} >= {fold.test_start}")
    if test_rows is not None and len(test_rows) and test_rows["issue_date"].min() < fold.test_start:
        raise AssertionError(f"leakage: test issue {test_rows['issue_date'].min()} < {fold.test_start}")


def split_fold(lag_df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a lag table into (train, test) rows for one fold by issue date.

    Training rows have target dates before test_start. Test rows have issue dates from
    test_start through test_end. Every transform is fit on observations before
    test_start, so it is valid for forecasts issued at or after that date. Test targets
    can extend past test_end by up to the lead time.
    """
    train = lag_df[lag_df["target_date"] < fold.test_start]
    test = lag_df[(lag_df["issue_date"] >= fold.test_start) & (lag_df["issue_date"] <= fold.test_end)]
    return train, test


def damped_persistence_pred(train: pd.DataFrame, test: pd.DataFrame, mode: str) -> np.ndarray:
    """Damped-persistence forecasts for the test rows of one fold and lead time.

    mode='regression' (damped_persistence_reg) regresses the lead-h target on lag_0 per
    basin, without intercept. Any other mode (damped_persistence_rho) multiplies lag_0
    by rho**h, where rho is the per-basin AR(1) coefficient (least squares of lag_0 on
    lag_1, no intercept, clipped to +-0.999). Basins absent from train get zero.
    """
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
    """Benchmark forecasts for every fold and lead time on the deseasonalized residual.

    Returns tidy prediction rows (name, issue_date, target_date, target, pred, model,
    fold, horizon). With standardize=True the models are fit in standardized units,
    target and pred are reported in centimeters, and target_std_units and pred_std_units
    hold the standardized values. ridge_own_plus_indices is produced only when indices
    is given and every test row has all its index lags.
    """
    out = []
    for fold in folds:
        resid_wide, train_std = deseasonalize_fold(wide, fold)
        if standardize:
            # Scaling by the training-window std keeps high-variance basins from
            # dominating the pooled loss
            resid_wide = resid_wide / train_std
        # Short gaps are filled in the features only, never the targets. Every model sees
        # the same fills, and anchor masking hides each filled value from issue dates
        # before the observation that closes its gap.
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
            preds["ridge_own_perbasin"] = per_basin_ridge_predict(train, test, own_cols)

            if indices is not None:
                # Climate indices are joined on issue date at lags of zero or more months,
                # so every value is known at issue time
                idx_feats = pd.concat(
                    {f"{c}_lag{k}": indices[c].shift(k) for c in indices.columns for k in index_lags},
                    axis=1,
                )
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
                    # target and pred go back to centimeters; *_std_units keep the scaled values
                    scale = df["name"].map(train_std)
                    df["target"] = df["target"] * scale
                    df["pred"] = df["pred"] * scale
                    df["target_std_units"] = df["target"] / scale
                    df["pred_std_units"] = df["pred"] / scale
                out.append(df)
    return pd.concat(out, ignore_index=True)


def summarize(pred_rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE, MAE and correlation per model and lead time, and skill over damped persistence.

    skill_vs_damped is 1 - (RMSE / RMSE_damped)**2 against the damped-persistence
    variant with the lower RMSE at that lead time, in standardized units when
    available and in centimeters otherwise.
    """
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
    key = "rmse_std" if std_units else "rmse_cm"
    damped = summary[summary["model"].str.startswith("damped")].groupby("horizon")[key].min()
    summary["skill_vs_damped"] = 1 - (summary[key] / summary["horizon"].map(damped)) ** 2
    return summary.sort_values(["horizon", key]).reset_index(drop=True)


def per_basin_wins(pred_rows: pd.DataFrame, challenger: str, reference: str) -> pd.DataFrame:
    """Number of basins where challenger has a lower RMSE than reference, by lead time."""
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
