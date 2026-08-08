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

def prepare_deseasonalized_basin_data(df, test_fraction=0.20):
    wide = (
        df.pivot(
            index="time",
            columns="basin",
            values="tws_cm",
        )
        .sort_index()
    )

    wide = wide.dropna(how="all")

    split_index = int(
        len(wide) * (1 - test_fraction)
    )

    split_date = wide.index[split_index]

    train = wide[
        wide.index < split_date
    ]

    deseasonalized = wide.copy()

    for basin in wide.columns:

        train_series = train[basin]

        monthly_climatology = (
            train_series
            .groupby(train_series.index.month)
            .mean()
        )

        seasonal_values = (
            wide.index.month.map(
                monthly_climatology
            ).to_numpy()
        )

        deseasonalized[basin] = (
            wide[basin]
            - seasonal_values
        )

    return deseasonalized, split_date

def find_top_correlated_neighbors(
    data,
    target_basin,
    split_date,
    n_neighbors=3,
):
    train = data[
        data.index < split_date
    ]
    correlations = (
        train.corr()[target_basin]
        .drop(target_basin)
        .dropna()
    )
    correlations = correlations.loc[
        correlations.abs()
        .sort_values(ascending=False)
        .index
    ]
    neighbors = (
        correlations
        .head(n_neighbors)
        .index
        .tolist()
    )
    return neighbors, correlations

def build_neighbor_lag_features(
    data,
    target_basin,
    neighbors,
    n_lags=12,
):
    result = pd.DataFrame(
        index=data.index
    )

    result["y"] = data[target_basin]

    # Target basin lags
    for lag in range(1, n_lags + 1):
        result[
            f"target_lag{lag}"
        ] = data[target_basin].shift(lag)

    # Neighbor lags
    for neighbor_number, neighbor in enumerate(
        neighbors,
        start=1,
    ):
        for lag in range(1, n_lags + 1):
            result[
                f"neighbor{neighbor_number}_lag{lag}"
            ] = data[neighbor].shift(lag)

    return result.dropna()


def find_top_lagged_correlated_basins(
    data,
    target_basin,
    split_date,
    n_basins=3,
    max_lag=12,
):
    train = data[
        data.index < split_date
    ]
    target = train[target_basin]

    candidates = []

    for basin in train.columns:
        if basin == target_basin:
            continue

        best_corr = None
        best_lag = None

        for lag in range(1, max_lag + 1):

            lagged_other = (
                train[basin]
                .shift(lag)
            )

            paired = pd.concat(
                [
                    target.rename("target"),
                    lagged_other.rename("other"),
                ],
                axis=1,
            ).dropna()

            if len(paired) < 3:
                continue

            corr = paired[
                "target"
            ].corr(
                paired["other"]
            )

            if pd.isna(corr):
                continue

            if (
                best_corr is None
                or abs(corr) > abs(best_corr)
            ):
                best_corr = corr
                best_lag = lag

        if best_corr is not None:

            candidates.append(
                {
                    "basin": basin,
                    "correlation": best_corr,
                    "lag": best_lag,
                }
            )

    candidates = sorted(
        candidates,
        key=lambda x: abs(
            x["correlation"]
        ),
        reverse=True,
    )

    return candidates[:n_basins]


def build_lagged_correlated_features(
    data,
    target_basin,
    selected_basins,
    target_lags=12,
):

    result = pd.DataFrame(
        index=data.index
    )

    result["y"] = data[target_basin]

    # Target basin's own history
    for lag in range(
        1,
        target_lags + 1
    ):

        result[
            f"target_lag{lag}"
        ] = (
            data[target_basin]
            .shift(lag)
        )

    for i, info in enumerate(
        selected_basins,
        start=1,
    ):

        basin = info["basin"]
        lag = info["lag"]

        result[
            f"correlated{i}_lag{lag}"
        ] = (
            data[basin]
            .shift(lag)
        )

    return result.dropna()