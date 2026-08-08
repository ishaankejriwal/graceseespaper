import pandas as pd

from src.config import (
    BASIN_TIMESERIES_FILE,
    RESULTS_DIR,
)

from src.features import (
    load_basin_timeseries,
    prepare_deseasonalized_basin_data,
    find_top_lagged_correlated_basins,
    build_lagged_correlated_features,
)

from src.models.ridge import (
    train_ridge,
    predict_ridge,
)

from src.evaluation import (
    calculate_metrics,
)

N_LAGS = 12
N_NEIGHBORS = 3
TEST_FRACTION = 0.20
RIDGE_ALPHA = 1.0

def run_basin(
    data,
    target_basin,
    split_date,
):

    selected = (find_top_lagged_correlated_basins(
        data,
        target_basin,
        split_date,
        n_basins=3,
        max_lag=12,
    ))

    if len(selected) < 3:
        raise ValueError(
            "Fewer than 3 valid correlated basins."
        )

    features = build_lagged_correlated_features(
        data,
        target_basin,
        selected,
        target_lags=12,
    )


    features = build_lagged_correlated_features(
        data,
        target_basin,
        selected,
        target_lags=N_LAGS,
    )

    train = features[
        features.index < split_date
    ]

    test = features[
        features.index >= split_date
    ]

    feature_cols = [
        col
        for col in features.columns
        if col != "y"
    ]

    X_train = train[feature_cols]
    y_train = train["y"]

    X_test = test[feature_cols]
    y_test = test["y"]

    model, scaler = train_ridge(
        X_train,
        y_train,
        alpha=RIDGE_ALPHA,
    )

    predictions = predict_ridge(
        model,
        scaler,
        X_test,
    )

    metrics = calculate_metrics(
        y_test.values,
        predictions,
    )

    return {
        "basin": target_basin,
        "model":
            "ridge_top3_lagged_correlated_deseasonalized",

        "correlated_1":
            selected[0]["basin"],
        "correlated_1_lag":
            selected[0]["lag"],
        "correlated_1_corr":
            selected[0]["correlation"],

        "correlated_2":
            selected[1]["basin"],
        "correlated_2_lag":
            selected[1]["lag"],
        "correlated_2_corr":
            selected[1]["correlation"],

        "correlated_3":
            selected[2]["basin"],
        "correlated_3_lag":
            selected[2]["lag"],
        "correlated_3_corr":
            selected[2]["correlation"],

        "n_train": len(train),
        "n_test": len(test),

        **metrics,
    }


def main():

    print("Loading basin data...")

    df = load_basin_timeseries(
        BASIN_TIMESERIES_FILE
    )

    data, split_date = (
        prepare_deseasonalized_basin_data(
            df,
            test_fraction=TEST_FRACTION,
        )
    )

    basins = sorted(
        data.columns
    )

    print(
        f"Found {len(basins)} basins."
    )
    print(
        f"Split date: {split_date.date()}"
    )

    results = []

    for i, basin in enumerate(
        basins,
        start=1,
    ):

        print(
            f"\n[{i}/{len(basins)}] "
            f"{basin}"
        )

        try:
            result = run_basin(
                data,
                basin,
                split_date,
            )
            results.append(result)
            print(
            f"      1. {result['correlated_1']} "
            f"(lag={result['correlated_1_lag']}, "
            f"r={result['correlated_1_corr']:.3f})"
        )

            print(
                f"      2. {result['correlated_2']} "
                f"(lag={result['correlated_2_lag']}, "
                f"r={result['correlated_2_corr']:.3f})"
            )

            print(
                f"      3. {result['correlated_3']} "
                f"(lag={result['correlated_3_lag']}, "
                f"r={result['correlated_3_corr']:.3f})"
            )

            print(
                f"    RMSE: "
                f"{result['rmse_cm']:.3f} cm"
            )

        except Exception as e:
            print(f"    ERROR: {e}")

    results_df = pd.DataFrame(
        results
    )

    output_folder = (
        RESULTS_DIR / "metrics"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "ridge_top3_lagged_deseasonalized_africa.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")

    if len(results_df) > 0:

        print(
            f"Average RMSE: "
            f"{results_df['rmse_cm'].mean():.3f} cm"
        )

        print(
            f"Average MAE: "
            f"{results_df['mae_cm'].mean():.3f} cm"
        )

        print(
            f"Average Pearson r: "
            f"{results_df['pearson_r'].mean():.3f}"
        )

        print(
            f"Average NSE: "
            f"{results_df['nse'].mean():.3f}"
        )

    print(
        f"\nSaved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()