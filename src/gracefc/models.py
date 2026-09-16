"""Baselines: pooled and per-basin ridge on own lags, and the RMSE used everywhere."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def fit_ridge(train: pd.DataFrame, feat_cols: list[str], alpha: float = 1.0) -> tuple:
    scaler = StandardScaler().fit(train[feat_cols].values)
    model = Ridge(alpha=alpha).fit(scaler.transform(train[feat_cols].values), train["target"].values)
    return model, scaler


def per_basin_ridge_predict(
    train: pd.DataFrame, test: pd.DataFrame, feat_cols: list[str], alpha: float = 1.0
) -> np.ndarray:
    """One ridge per basin: dodges the pooling penalty under coefficient heterogeneity."""
    preds = np.full(len(test), np.nan)
    test_idx = {name: np.flatnonzero(test["name"].values == name) for name in test["name"].unique()}
    for name, grp in train.groupby("name"):
        if name not in test_idx or len(grp) < 3 * len(feat_cols):
            continue
        model = Ridge(alpha=alpha).fit(grp[feat_cols].values, grp["target"].values)
        rows = test_idx[name]
        preds[rows] = model.predict(test.iloc[rows][feat_cols].values)
    # Basins with too little training history fall back to zero (the climatology forecast)
    return np.nan_to_num(preds)


def predict_ridge(model, scaler, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return model.predict(scaler.transform(df[feat_cols].values))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
