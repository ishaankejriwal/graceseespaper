from pathlib import Path
import pandas as pd
from src.config import RESULTS_DIR


METRICS_DIR = RESULTS_DIR / "metrics"

RIDGE_FILE = (
    METRICS_DIR
    / "ridge_own_lags_africa.csv"
)

RF_FILE = (
    METRICS_DIR
    / "random_forest_own_lags_africa.csv"
)

OUTPUT_FILE = (
    METRICS_DIR
    / "ridge_vs_random_forest.csv"
)


def main():
    print("Loading results...")

    ridge = pd.read_csv(RIDGE_FILE)
    rf = pd.read_csv(RF_FILE)

    print(f"Ridge basins: {len(ridge)}")
    print(f"RF basins:    {len(rf)}")

    ridge = ridge[
        [
            "basin",
            "rmse_cm",
            "mae_cm",
            "pearson_r",
            "nse",
            "r2",
        ]
    ].rename(
        columns={
            "rmse_cm": "ridge_rmse",
            "mae_cm": "ridge_mae",
            "pearson_r": "ridge_pearson",
            "nse": "ridge_nse",
            "r2": "ridge_r2",
        }
    )

    rf = rf[
        [
            "basin",
            "rmse_cm",
            "mae_cm",
            "pearson_r",
            "nse",
            "r2",
        ]
    ].rename(
        columns={
            "rmse_cm": "rf_rmse",
            "mae_cm": "rf_mae",
            "pearson_r": "rf_pearson",
            "nse": "rf_nse",
            "r2": "rf_r2",
        }
    )

    comparison = ridge.merge(
        rf,
        on="basin",
        how="inner",
    )

    comparison["rmse_improvement_rf"] = (
        comparison["ridge_rmse"]
        - comparison["rf_rmse"]
    )

    comparison["mae_improvement_rf"] = (
        comparison["ridge_mae"]
        - comparison["rf_mae"]
    )

    # Positive means RF has HIGHER score
    comparison["pearson_improvement_rf"] = (
        comparison["rf_pearson"]
        - comparison["ridge_pearson"]
    )

    comparison["nse_improvement_rf"] = (
        comparison["rf_nse"]
        - comparison["ridge_nse"]
    )

    comparison["winner"] = comparison.apply(
        lambda row:
        "Random Forest"
        if row["rf_rmse"] < row["ridge_rmse"]
        else "Ridge"
        if row["ridge_rmse"] < row["rf_rmse"]
        else "Tie",
        axis=1,
    )

    comparison = comparison.sort_values(
        "rmse_improvement_rf",
        ascending=False,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n==============================")
    print("OVERALL RESULTS")
    print("==============================")

    print(
        f"\nBasins compared: "
        f"{len(comparison)}"
    )

    print("\nAverage RMSE:")

    print(
        f"Ridge:         "
        f"{comparison['ridge_rmse'].mean():.3f} cm"
    )

    print(
        f"Random Forest: "
        f"{comparison['rf_rmse'].mean():.3f} cm"
    )

    print("\nAverage MAE:")

    print(
        f"Ridge:         "
        f"{comparison['ridge_mae'].mean():.3f} cm"
    )

    print(
        f"Random Forest: "
        f"{comparison['rf_mae'].mean():.3f} cm"
    )

    print("\nWins by RMSE:")

    print(
        comparison["winner"]
        .value_counts()
        .to_string()
    )

    avg_improvement = (
        comparison[
            "rmse_improvement_rf"
        ].mean()
    )

    print(
        "\nAverage RF RMSE improvement "
        "over Ridge:"
    )

    print(
        f"{avg_improvement:+.3f} cm"
    )

    print(
        "\nTop 5 basins where "
        "Random Forest improves most:"
    )

    print(
        comparison[
            [
                "basin",
                "ridge_rmse",
                "rf_rmse",
                "rmse_improvement_rf",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )

    print(
        "\nTop 5 basins where "
        "Ridge performs better:"
    )

    print(
        comparison[
            [
                "basin",
                "ridge_rmse",
                "rf_rmse",
                "rmse_improvement_rf",
            ]
        ]
        .tail(5)
        .sort_values(
            "rmse_improvement_rf"
        )
        .to_string(index=False)
    )

    print(
        f"\nFull comparison saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()