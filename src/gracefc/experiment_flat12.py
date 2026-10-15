"""Ridge corrections to the Kalman filter forecast on a 12-month history of the filtered state.

Builds the 12-month window features and emits the ridge models at every lead time without
a torch dependency. experiment_lstm and experiment_lstm_combined import the window helpers
from here, so the LSTM and the ridge models are built from the same design matrix.

Models emitted (paper names in parentheses):
  kalman_ar1             filter forecast rho^h x(t|t) (KF)
  kalman_own_ridge       ridge on the current filtered state (KF-R1)
  ridge_own_flat12       ridge on the filtered state over a 12-month window (KF-R12)
  ridge_own_era5_flat12  the same window plus the 11 ERA5-Land anomaly series (KF-R12E)

Months before the start of the series are padded with zero, the prior mean of the state.
"""
from functools import partial

import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .phase7 import emit_rows, fold_setup, horizon_frame

LOOKBACK = 12


def _window_channels(t_idx: np.ndarray) -> tuple:
    """(rows, L) month indices of each row's window, clipped at 0, and the mask of valid months."""
    offs = np.arange(LOOKBACK - 1, -1, -1)
    widx = t_idx[:, None] - offs[None, :]
    return np.clip(widx, 0, None), widx >= 0


def _state_channel(mat: np.ndarray, widx: np.ndarray, valid: np.ndarray, node: np.ndarray) -> np.ndarray:
    """(rows, L) history of mat for each row's node; zeros where node or month is absent."""
    ok = valid & (node >= 0)[:, None]
    return np.where(ok, mat[widx, np.clip(node, 0, None)[:, None]], 0.0)


def _era5_state_tensor(era5_feats: pd.DataFrame, era5_wide: dict, filt_index, names) -> np.ndarray:
    """(T, N, V) lag-0 standardized ERA5-Land anomalies aligned to the filter grid; NaN -> 0."""
    mats = []
    for var in era5_wide:
        m = era5_feats.pivot(index="issue_date", columns="name", values=f"{var}_l0")
        mats.append(m.reindex(index=filt_index, columns=names).values)
    return np.nan_to_num(np.stack(mats, axis=-1))


def window_design(F: np.ndarray, E: np.ndarray, frame: dict) -> dict:
    """Return the 12-month window channels and their flattened design matrices.

    own_* are (rows, L) and era_* are (rows, L, V) sequence channels for the LSTM; X12_*
    stacks both flat for the ridge and MLP heads. experiment_lstm builds its LSTM inputs
    and its flat models from the same call.
    """
    widx_tr, valid_tr = _window_channels(frame["t_idx"])
    widx_te, valid_te = _window_channels(frame["e_idx"])
    own_tr = _state_channel(F, widx_tr, valid_tr, frame["tr_pos"])
    own_te = _state_channel(F, widx_te, valid_te, frame["te_pos"])
    # ERA5-Land channels use the same window grid as the state channel
    era_tr = np.where(valid_tr[:, :, None], E[widx_tr, frame["tr_pos"][:, None], :], 0.0)
    era_te = np.where(valid_te[:, :, None], E[widx_te, frame["te_pos"][:, None], :], 0.0)
    return {
        "widx_tr": widx_tr, "valid_tr": valid_tr,
        "widx_te": widx_te, "valid_te": valid_te,
        "own_tr": own_tr, "own_te": own_te,
        "era_tr": era_tr, "era_te": era_te,
        "X12_tr": np.column_stack([own_tr, era_tr.reshape(len(own_tr), -1)]),
        "X12_te": np.column_stack([own_te, era_te.reshape(len(own_te), -1)]),
    }


def run_flat12_experiment(
    wide: pd.DataFrame,
    era5_wide: dict[str, pd.DataFrame],
    horizons: range = range(1, 7),
    folds: list[Fold] = DEFAULT_FOLDS,
    params_cache: dict | None = None,
    era5_lags: tuple[int, ...] = (0, 1, 2),
) -> pd.DataFrame:
    """Prediction rows for the four models above, in the row schema of phase7.emit_rows."""
    out = []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names = setup["names"]
        # Rows without ERA5-Land features are dropped for every model, so kalman_ar1 is
        # scored on the same rows as the ridge models and all comparisons are paired.
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)
        E = _era5_state_tensor(era5_feats, era5_wide, setup["filt"].index, names)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            kal_te = te["kalman"].values
            design = window_design(setup["F"], E, frame)

            emit = partial(emit_rows, out, te, fold, h)
            emit("kalman_ar1", kal_te)
            emit("kalman_own_ridge",
                 kal_te + _fit_head("ridge", tr[["own_state"]].values, ytr,
                                    te[["own_state"]].values, 0))
            emit("ridge_own_flat12",
                 kal_te + _fit_head("ridge", design["own_tr"], ytr, design["own_te"], 0))
            emit("ridge_own_era5_flat12",
                 kal_te + _fit_head("ridge", design["X12_tr"], ytr, design["X12_te"], 0))
            print(f"{fold.name} h{h} done", flush=True)
    return pd.concat(out, ignore_index=True)
