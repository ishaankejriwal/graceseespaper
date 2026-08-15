"""Baselines and the ridge + neighbor-residual-MLP family from the prior Africa benchmark."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def damped_persistence_forecast(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-basin AR(1) rho^h decay applied to lag_0. The honest bar for long-memory signals."""
    preds = np.empty(len(test))
    rho_by_basin = {}
    for name, grp in train.groupby("name"):
        x, y = grp["lag_0"].values, grp["target"].values
        # rho is the lag-h autocorrelation estimated by regression through the origin
        denom = float(np.dot(x, x))
        rho_by_basin[name] = float(np.dot(x, y) / denom) if denom > 0 else 0.0
    for i, row in enumerate(test.itertuples()):
        preds[i] = rho_by_basin.get(row.name, 0.0) * row.lag_0
    return preds


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


def fit_residual_mlp(
    train: pd.DataFrame,
    residual: np.ndarray,
    feat_cols: list[str],
    seed: int = 0,
) -> tuple:
    """MLP learns the ridge residual from extra features; ridge stays the backbone."""
    scaler = StandardScaler().fit(train[feat_cols].values)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        random_state=seed,
    ).fit(scaler.transform(train[feat_cols].values), residual)
    return mlp, scaler


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
