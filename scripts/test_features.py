from src.config import BASIN_TIMESERIES_FILE
from src.features import (
    load_basin_timeseries,
    get_basin_series,
    build_own_lag_features,
)

df = load_basin_timeseries(
    BASIN_TIMESERIES_FILE
)
print(df.head())

basin = df["basin"].iloc[0]
series = get_basin_series(
    df,
    basin
)
features = build_own_lag_features(
    series,
    n_lags=12
)

print("\nBasin:")
print(basin)
print("\nSeries:")
print(series.head())
print("\nFeatures:")
print(features.head())
print("\nShape:")
print(features.shape)