import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from scipy.stats import pearsonr


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    if len(y_true) > 1:
        pearson_r = pearsonr(
            y_true,
            y_pred
        )[0]
    else:
        pearson_r = np.nan

    denominator = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if denominator == 0:
        nse = np.nan
    else:
        nse = (
            1
            -
            np.sum(
                (y_true - y_pred) ** 2
            )
            / denominator
        )

    return {
        "rmse_cm": rmse,
        "mae_cm": mae,
        "pearson_r": pearson_r,
        "nse": nse,
        "r2": r2,
    }