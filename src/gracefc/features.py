"""Issue-date lag features for forecasting: everything is indexed by when the forecast is made."""
import numpy as np
import pandas as pd

# Issue-date lags carried over from the prior Africa work: t, t-1, t-2, t-5, t-11
DEFAULT_LAGS = (0, 1, 2, 5, 11)


def pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Long basin-month table -> wide (date x basin_name) matrix on a complete monthly index."""
    wide = long_df.pivot(index="date", columns="name", values="twsa_cm").sort_index()
    # Reindex to a gapless monthly axis so lag arithmetic never silently skips missing months
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    return wide.reindex(full)


def interpolate_short_gaps(wide: pd.DataFrame, max_run: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linear-fill NaN runs of at most max_run months. For FEATURE series only, never targets.

    Also returns each filled cell's right-anchor date: a filled value is only causally legal
    at issue dates on or after that anchor, and build_lag_table masks the rest.
    """
    filled = wide.interpolate(method="linear", limit=max_run, limit_area="inside")
    anchors = pd.DataFrame(pd.NaT, index=wide.index, columns=wide.columns)
    for col in wide.columns:
        isna = wide[col].isna()
        run_id = (~isna).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        # limit fills the first max_run of longer runs too; revert those so long gaps stay missing
        filled.loc[isna & (run_len > max_run), col] = np.nan
        # Right anchor of each legally filled cell = the first observed month after its run
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
    """One row per (issue_date, basin) with own-basin lags and the target at issue+horizon.

    feature_wide lets lags come from a gap-interpolated copy while targets stay observed-only.
    anchors enforces causality: an interpolated lag value is masked at issue dates before its
    right anchor, so no feature ever embeds data that postdates the forecast issue.
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


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/10/20 split on unique target dates, matching the prior Africa benchmark convention."""
    dates = np.sort(df["target_date"].unique())
    n = len(dates)
    t_end = dates[int(n * train_frac) - 1]
    v_end = dates[int(n * (train_frac + val_frac)) - 1]
    train = df[df["target_date"] <= t_end]
    val = df[(df["target_date"] > t_end) & (df["target_date"] <= v_end)]
    test = df[df["target_date"] > v_end]
    return train, val, test


def neighbor_features(
    wide: pd.DataFrame,
    graph: dict[str, list[str]],
    lags: tuple[int, ...] = DEFAULT_LAGS,
    anchors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Lagged TWSA of each basin's graph neighbors, ordered by rank so k stays fixed per row."""
    rows = []
    for name, nbrs in graph.items():
        df = pd.DataFrame(index=wide.index)
        for rank, nb in enumerate(nbrs):
            for k in lags:
                col = wide[nb].shift(k)
                if anchors is not None:
                    # Same causality mask as own lags: no interpolated value before its anchor
                    col[anchors[nb].shift(k).values > wide.index.values] = np.nan
                df[f"nbr{rank}_lag_{k}"] = col
        df["issue_date"] = wide.index
        df["name"] = name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
