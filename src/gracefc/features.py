"""Own-basin lag features indexed by forecast issue date, and the short-gap fill behind them."""
import numpy as np
import pandas as pd

# Lags in months before the issue month t: t, t-1, t-2, t-5, t-11
DEFAULT_LAGS = (0, 1, 2, 5, 11)


def pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Long basin-month table -> wide (date x basin_name) matrix on a complete monthly index."""
    wide = long_df.pivot(index="date", columns="name", values="twsa_cm").sort_index()
    # A complete monthly axis keeps lag shifts aligned with calendar months across gaps
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    return wide.reindex(full)


def interpolate_short_gaps(wide: pd.DataFrame, max_run: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linearly fill interior NaN runs of at most max_run months; for features, never targets.

    Returns the filled frame and, for each filled cell, its right anchor: the first observed
    month after the run. A filled value is known only at issue dates on or after its anchor,
    and build_lag_table masks it at earlier issue dates.
    """
    filled = wide.interpolate(method="linear", limit=max_run, limit_area="inside")
    anchors = pd.DataFrame(pd.NaT, index=wide.index, columns=wide.columns)
    for col in wide.columns:
        isna = wide[col].isna()
        run_id = (~isna).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        # interpolate(limit=...) also fills the first max_run months of longer runs; undo
        # that so long gaps stay missing
        filled.loc[isna & (run_len > max_run), col] = np.nan
        valid_dates = wide.index[~isna]
        for t in wide.index[isna & (run_len <= max_run)]:
            after = valid_dates[valid_dates > t]
            if len(after):
                anchors.loc[t, col] = after[0]
    return filled, anchors


def build_lag_table(
    wide: pd.DataFrame,
    horizon: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    feature_wide: pd.DataFrame | None = None,
    anchors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (issue_date, basin) with own-basin lags and the target horizon months later.

    Lags come from feature_wide when given (a gap-filled copy); targets always come from the
    observed values in wide. With anchors, a filled lag value is masked at issue dates before
    its right anchor, so no feature uses data observed after the issue date. Rows with any
    missing lag or target are dropped.
    """
    src = wide if feature_wide is None else feature_wide
    rows = []
    for name in wide.columns:
        s = src[name]
        df = pd.DataFrame({f"lag_{k}": s.shift(k) for k in lags})
        if anchors is not None:
            a = anchors[name]
            for k in lags:
                illegal = a.shift(k).values > wide.index.values
                df.loc[illegal, f"lag_{k}"] = np.nan
        df["target"] = wide[name].shift(-horizon)
        df["issue_date"] = wide.index
        df["target_date"] = wide.index + pd.DateOffset(months=horizon)
        df["name"] = name
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    # Drop rows where any feature or the target is missing (mission gap, record edges)
    feat_cols = [f"lag_{k}" for k in lags]
    return out.dropna(subset=feat_cols + ["target"]).reset_index(drop=True)
