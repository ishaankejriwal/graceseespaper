"""Removes the boring, already-predictable parts of a water-storage series.

Raw water storage is dominated by two patterns anyone can predict: a long-term drying or
wetting trend, and the annual wet/dry cycle. Leaving them in would make any model look
excellent while predicting nothing of interest. So we fit and subtract both, and forecast
only the leftover — the genuinely hard part.

The fit is a straight line (the trend) plus two sine/cosine pairs (the yearly cycle and
its half-year overtone), estimated by least squares.

Crucially the fit uses TRAINING months only and is then applied forward. Fitting it on
the whole record would leak future information into the test period.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecompositionFit:
    # Coefficients: [intercept, slope, cos1, sin1, cos2, sin2] on fractional-year time
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
    """Fit trend + annual + semiannual by least squares. Call ONLY on the training window."""
    s = series.dropna()
    if s.shape[0] < 36:
        # Under three years of data the harmonic fit is unstable; fail loudly, never silently
        raise ValueError(f"only {s.shape[0]} months available, need >= 36 to fit climatology")
    t = _frac_year(s.index)
    t0 = float(t[0])
    X = _design_matrix(t - t0)
    coeffs, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    return DecompositionFit(coeffs=coeffs, t0=t0)


def deseasonalize(series: pd.Series, fit: DecompositionFit) -> pd.Series:
    """Remove a previously fitted trend+seasonal model. Safe to apply to any window."""
    return series - fit.predict(series.index)
