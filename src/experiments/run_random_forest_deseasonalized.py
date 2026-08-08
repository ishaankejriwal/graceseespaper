import pandas as pd

from src.config import (
    BASIN_TIMESERIES_FILE,
    RESULTS_DIR,
)

from src.features import (
    load_basin_timeseries,
    get_basin_series,
    prepare_deseasonalized_features,
)

from src.models.random_forest import (
    train_random_forest,
    predict_random_forest,
)

from src.evaluation import (
    calculate_metrics,
)


N_LAGS = 12
TEST_FRACTION = 0.20

N_ESTIMATORS = 300
MAX_DEPTH = None
RANDOM_STATE = 42


def run_basin(df, basin):
    series = get_basin_series(
        df,
        basin
    )

    (
        train,
        test,
        monthly_climatology,
        split_date,
    ) = prepare_deseasonalized_features(
        series,
        n_lags=N_LAGS,
        test_fraction=TEST_FRACTION,
    )

    feature_cols = [
        col
        for col in train.columns
        if col != "y"
    ]

    X_train = train[feature_cols]
    y_train = train["y"]

    X_test = test[feature_cols]
    y_test = test["y"]

    model = train_random_forest(
        X_train,
        y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    predictions = predict_random_forest(
        model,
        X_test
    )

    metrics = calculate_metrics(
        y_test.values,
        predictions
    )

    return {
        "basin": basin,
        "model": "random_forest_own_lags_deseasonalized",

        "n_lags": N_LAGS,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,

        "n_train": len(train),
        "n_test": len(test),

        "train_start": train.index.min(),
        "train_end": train.index.max(),

        "test_start": test.index.min(),
        "test_end": test.index.max(),

        "split_date": split_date,
        **metrics,
    }


def main():

    print("Loading basin data...")

    df = load_basin_timeseries(
        BASIN_TIMESERIES_FILE
    )

    basins = sorted(
        df["basin"].unique()
    )

    print(
        f"Found {len(basins)} basins."
    )

    results = []

    for i, basin in enumerate(
        basins,
        start=1
    ):

        print(
            f"[{i}/{len(basins)}] "
            f"{basin}"
        )

        try:

            result = run_basin(
                df,
                basin
            )

            results.append(result)

            print(
                f"    RMSE: "
                f"{result['rmse_cm']:.3f} cm"
            )

        except Exception as e:

            print(
                f"    ERROR: {e}"
            )

    results_df = pd.DataFrame(
        results
    )

    output_folder = (
        RESULTS_DIR / "metrics"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder
        / "random_forest_own_lags_deseasonalized_africa.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\nDone.")

    print(
        f"Saved results to:\n"
        f"{output_file}"
    )

    if len(results_df) > 0:

        print("\nAverage metrics:")

        print(
            results_df[
                [
                    "rmse_cm",
                    "mae_cm",
                    "pearson_r",
                    "nse",
                    "r2",
                ]
            ].mean()
        )


if __name__ == "__main__":
    main()