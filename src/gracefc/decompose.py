"""Trend and seasonal climatology used to deseasonalize storage and ERA5-Land series.

The climatology is a linear trend plus annual and semiannual harmonics (six coefficients),
fit by ordinary least squares on the training window only and then subtracted from the
whole record, test months included.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecompositionFit:
    # coeffs: [intercept, slope, cos1, sin1, cos2, sin2] on time in years since t0,
    # the first fitted month
    coeffs: np.ndarray
    t0: float

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        X = _design_matrix(_frac_year(index) - self.t0)
        return pd.Series(X @ self.coeffs, index=index)


def _frac_year(index: pd.DatetimeIndex) -> np.ndarray:
    return index.year + (index.dayofyear - 1) / 365.25


def _design_matrix(t: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * t
    return np.column_stack([
        np.ones_like(t), t,
        np.cos(w), np.sin(w),
        np.cos(2 * w), np.sin(2 * w),
    ])


def fit_climatology(series: pd.Series) -> DecompositionFit:
    """Fit trend plus annual and semiannual harmonics by least squares, ignoring NaN months.

    Call it on the training window only. Raises ValueError with fewer than 36 observed months.
    """
    s = series.dropna()
    if s.shape[0] < 36:
        # The harmonic fit is unstable on less than three years of data
        raise ValueError(f"only {s.shape[0]} months available, need >= 36 to fit climatology")
    t = _frac_year(s.index)
    t0 = float(t[0])
    X = _design_matrix(t - t0)
    coeffs, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    return DecompositionFit(coeffs=coeffs, t0=t0)


def deseasonalize(series: pd.Series, fit: DecompositionFit) -> pd.Series:
    """Subtract a fitted climatology from series; valid for any window, test months
    included."""
    return series - fit.predict(series.index)
