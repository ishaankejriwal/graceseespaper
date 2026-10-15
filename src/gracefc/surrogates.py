"""IAAFT surrogates: each basin keeps its distribution and spectrum, cross-basin synchrony is lost.

Used by scripts/run_phase4_surrogates.py. Two design choices matter when reading its results:

- Surrogates are built from the full record, test months included. A surrogate is a null
  draw of the data, not a forecast, so no information flows into the real arm, and the
  evaluation needs neighbor values in test months that a training-only surrogate could
  not supply. Because the distribution and spectrum are estimated with test-period
  observations, this is a transductive sensitivity test rather than a prospective control
  within each fold. Sharing the real series' full-record distribution and spectrum keeps
  the null closely matched to the real data, which does not by itself make the test
  conservative.
- The FFT runs on the full monthly grid. Missing months are linearly interpolated before
  the transform and set back to NaN afterwards, so the preserved spectrum is that of the
  true monthly time axis.
"""
import numpy as np
import pandas as pd


def _gap_fill(x: np.ndarray, isnan: np.ndarray) -> np.ndarray:
    """Linear interpolation over gap months on the regular grid; flat at the edges."""
    pos = np.arange(x.size, dtype=float)
    return np.interp(pos, pos[~isnan], x[~isnan])


def iaaft(series: np.ndarray, seed: int, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate of one series with NaN months.

    Works on the gap-filled full-length series, so the spectrum is defined on the true
    monthly axis and the amplitude distribution is that of the gap-filled series (89 %
    observed values for the GRACE record). Gap months are NaN again in the output.
    Raises ValueError for fewer than 24 observed months.
    """
    x = np.asarray(series, dtype=float)
    isnan = np.isnan(x)
    if (~isnan).sum() < 24:
        # Falling back to the real series would put real information into the null arm
        raise ValueError(f"series too short to surrogate ({(~isnan).sum()} obs); exclude it upstream")
    g = _gap_fill(x, isnan)
    rng = np.random.default_rng(seed)
    sorted_vals = np.sort(g)
    target_amp = np.abs(np.fft.rfft(g))
    y = rng.permutation(g)
    for _ in range(n_iter):
        # Impose the spectrum, then restore the exact amplitude distribution
        spec = np.fft.rfft(y)
        phases = np.angle(spec)
        y = np.fft.irfft(target_amp * np.exp(1j * phases), n=g.size)
        rank = np.argsort(np.argsort(y))
        y = sorted_vals[rank]
    out = np.where(isnan, np.nan, y)
    return out


def surrogate_wide(wide: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Independent IAAFT surrogate for each basin column, with its NaN months preserved."""
    out = {}
    for i, name in enumerate(wide.columns):
        out[name] = iaaft(wide[name].values, seed=seed * 100_003 + i)
    return pd.DataFrame(out, index=wide.index)
