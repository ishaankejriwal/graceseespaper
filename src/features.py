import pandas as pd

def load_basin_timeseries(path):
    df = pd.read_csv(
        path,
        parse_dates=["time"]
    )

    return df

def get_basin_series(df, basin):
    basin_df = (
        df[df["basin"] == basin]
        .sort_values("time")
        .set_index("time")
    )

    return basin_df["tws_cm"]

def build_own_lag_features(
    series,
    n_lags=12
):
    df = pd.DataFrame({
        "y": series
    })

    for lag in range(1, n_lags + 1):
        df[f"lag{lag}"] = series.shift(lag)

    return df.dropna()

def chronological_split(df, test_fraction=0.20):
    split_index = int(len(df) * (1 - test_fraction))

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    return train, test


def deseasonalize_train_test(
    train_series,
    test_series
):
    # Calculate average TWS for each calendar month
    monthly_climatology = (
        train_series
        .groupby(train_series.index.month)
        .mean()
    )

    # Remove seasonal cycle from training data
    train_deseasonalized = (
        train_series
        - train_series.index.month.map(
            monthly_climatology
        ).to_numpy()
    )

    # Remove the SAME training-derived seasonal cycle from the test data
    test_deseasonalized = (
        test_series
        - test_series.index.month.map(
            monthly_climatology
        ).to_numpy()
    )

    return (
        train_deseasonalized,
        test_deseasonalized,
        monthly_climatology,
    )

def prepare_deseasonalized_features(
    series,
    n_lags=12,
    test_fraction=0.20,
):
    """
    Split a basin time series chronologically,
    calculate seasonal climatology from training
    data only, deseasonalize, and create lag features.
    """

    # Make sure dates are sorted
    series = series.sort_index().dropna()
    split_index = int(
        len(series) * (1 - test_fraction)
    )

    split_date = series.index[split_index]

    raw_train = series.iloc[:split_index]
    raw_test = series.iloc[split_index:]

    # Training-only monthly climatology

    monthly_climatology = (
        raw_train
        .groupby(raw_train.index.month)
        .mean()
    )

    # Deseasonalize ALL observations using training climatology

    seasonal_values = (
        series.index.month.map(
            monthly_climatology
        ).to_numpy()
    )

    deseasonalized = (
        series - seasonal_values
    )
    # Create lag features

    features = build_own_lag_features(
        deseasonalized,
        n_lags=n_lags
    )

    # Split using same date

    train = features[
        features.index < split_date
    ]

    test = features[
        features.index >= split_date
    ]

    return (
        train,
        test,
        monthly_climatology,
        split_date,
    )