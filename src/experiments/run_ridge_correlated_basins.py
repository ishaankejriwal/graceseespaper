import pandas as pd

from src.config import (
    BASIN_TIMESERIES_FILE,
    RESULTS_DIR,
)

from src.features import (
    load_basin_timeseries,
    prepare_deseasonalized_basin_data,
    find_top_correlated_neighbors,
    build_neighbor_lag_features,
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

    neighbors, correlations = (
        find_top_correlated_neighbors(
            data,
            target_basin,
            split_date,
            n_neighbors=N_NEIGHBORS,
        )
    )

    features = build_neighbor_lag_features(
        data,
        target_basin,
        neighbors,
        n_lags=N_LAGS,
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
        "model": "ridge_top3_lagged_basins_deseasonalized",

        "neighbor_1": neighbors[0],
        "neighbor_1_corr": correlations[neighbors[0]],

        "neighbor_2": neighbors[1],
        "neighbor_2_corr": correlations[neighbors[1]],

        "neighbor_3": neighbors[2],
        "neighbor_3_corr": correlations[neighbors[2]],

        "n_lags": N_LAGS,
        "n_neighbors": N_NEIGHBORS,

        "n_train": len(train),
        "n_test": len(test),

        "train_start": train.index.min(),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),

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
            print("    Neighbors:")
            print(
                f"      1. {result['neighbor_1']} "
                f"(r={result['neighbor_1_corr']:.3f})"
            )
            print(
                f"      2. {result['neighbor_2']} "
                f"(r={result['neighbor_2_corr']:.3f})"
            )
            print(
                f"      3. {result['neighbor_3']} "
                f"(r={result['neighbor_3_corr']:.3f})"
            )
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
        exist_ok=True,
    )

    output_file = (
        output_folder
        / "ridge_top3_neighbors_deseasonalized_africa.csv"
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