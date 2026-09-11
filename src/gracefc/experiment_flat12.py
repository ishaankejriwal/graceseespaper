"""Flat 12-month ridge corrections on the Kalman backbone — the torch-free reference step.

The strongest own-basin model in this study is a ridge head over a FLAT 12-month history of
the Kalman-filtered state (optionally with the 11 ERA5 standardized anomaly series over the
same window). It was born inside experiment_lstm.py as the LSTM's linear twin, which chained
the paper's headline model to a torch dependency and to horizons 1-3. This module owns the
feature construction and emits the ridge arms on their own, at every lead, with no torch.

experiment_lstm.py imports the window/flattening helpers from here, so there is exactly one
implementation of the design matrix and the two engines agree row for row.

Arms emitted:
  kalman_ar1             the AR(1)+observation-noise filter forecast, rho^h x(t)
  kalman_own_ridge       ridge on the own filtered state at lag 0 (phase 3b's comparator)
  ridge_own_flat12       ridge on the own filtered state over a 12-month window
  ridge_own_era5_flat12  the same window plus the 11 ERA5 anomaly channels over that window

Months before a basin's record start pad with zeros, the state prior mean, so padding is
principled rather than arbitrary.
"""
import numpy as np
import pandas as pd

from .era5 import era5_fold_features
from .evaluate import DEFAULT_FOLDS, Fold
from .experiment_nonlinear import _fit_head
from .phase7 import fold_setup, horizon_frame

LOOKBACK = 12


def _window_channels(t_idx: np.ndarray) -> tuple:
    """Window index grid and validity mask shared by every channel of a row set."""
    offs = np.arange(LOOKBACK - 1, -1, -1)
    widx = t_idx[:, None] - offs[None, :]
    return np.clip(widx, 0, None), widx >= 0


def _state_channel(mat: np.ndarray, widx: np.ndarray, valid: np.ndarray, node: np.ndarray) -> np.ndarray:
    """(rows, L) history of mat for each row's node; zeros where node or month is absent."""
    ok = valid & (node >= 0)[:, None]
    return np.where(ok, mat[widx, np.clip(node, 0, None)[:, None]], 0.0)


def _era5_state_tensor(era5_feats: pd.DataFrame, era5_wide: dict, filt_index, names) -> np.ndarray:
    """(T, N, V) standardized ERA5 anomalies aligned to the filter grid; NaN -> 0."""
    mats = []
    for var in era5_wide:
        m = era5_feats.pivot(index="issue_date", columns="name", values=f"{var}_l0")
        mats.append(m.reindex(index=filt_index, columns=names).values)
    return np.nan_to_num(np.stack(mats, axis=-1))


def window_design(F: np.ndarray, E: np.ndarray, frame: dict) -> dict:
    """The 12-month window channels and their flattened ridge/MLP design matrices.

    Returns own/ERA5 sequence channels (rows, L) and (rows, L, V) for sequence models, and
    the column-stacked flat twins X12 that the ridge heads consume. Single source of truth:
    experiment_lstm.py builds its LSTM inputs and its flat twins from this same call.
    """
    widx_tr, valid_tr = _window_channels(frame["t_idx"])
    widx_te, valid_te = _window_channels(frame["e_idx"])
    own_tr = _state_channel(F, widx_tr, valid_tr, frame["tr_pos"])
    own_te = _state_channel(F, widx_te, valid_te, frame["te_pos"])
    # ERA5 sequence channels come from the same window grid as the state channel
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
    """Prediction rows for the four arms above, in the phase 7 row schema."""
    out = []
    for fold in folds:
        setup = fold_setup(wide, fold, params_cache)
        names = setup["names"]
        # ERA5 rows are dropped up front for EVERY arm, so kalman_ar1 here sits on the
        # same row set as the ridge arms and every contrast below is paired.
        era5_feats, era5_cols = era5_fold_features(era5_wide, fold.test_start, names, era5_lags)
        E = _era5_state_tensor(era5_feats, era5_wide, setup["filt"].index, names)

        for h in horizons:
            frame = horizon_frame(setup, fold, h, era5_feats, era5_cols)
            if frame is None:
                continue
            tr, te, ytr = frame["tr"], frame["te"], frame["ytr"]
            kal_te = te["kalman"].values
            design = window_design(setup["F"], E, frame)

            def emit(label: str, pred: np.ndarray) -> None:
                df = te[["name", "issue_date", "target_date", "target"]].copy()
                df["pred"] = pred
                df["model"], df["fold"], df["horizon"] = label, fold.name, h
                out.append(df)

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
