"""IAAFT surrogates: keep each series' distribution and spectrum, destroy cross-basin synchrony.

Two properties of this null are deliberate and should be read together with
scripts/run_phase4_surrogates.py:

- The surrogate is built from the FULL record, test months included. It is a null draw
  of the DATA, not a forecast system: no information flows into the real arm, and a
  truncated-support surrogate could not supply the test-month neighbor values the
  evaluation needs. Sharing the real series' full-record marginal and spectrum makes the
  null maximally similar to the real arm, which is the conservative direction for the
  positive claim.
- The FFT runs on the full monthly grid. GRACE gap months are linearly interpolated
  before the transform and the NaN mask is re-imposed after, so the preserved spectrum
  is that of the true monthly time axis. (Before audit 2026-08-15 the transform ran on
  the NaN-compressed vector — 257 of 290 months spliced together — so the "preserved"
  frequencies did not correspond to real months.)
"""
import numpy as np
import pandas as pd


def _gap_fill(x: np.ndarray, isnan: np.ndarray) -> np.ndarray:
    """Linear interpolation over gap months on the regular grid; flat at the edges."""
    pos = np.arange(x.size, dtype=float)
    return np.interp(pos, pos[~isnan], x[~isnan])


def iaaft(series: np.ndarray, seed: int, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate of one series (NaN-safe).

    Operates on the gap-filled full-length series so the spectrum lives on the true
    monthly axis; the amplitude distribution is the gap-filled series' (89% observed
    values for the GRACE record). Gap months are NaN again in the output.
    """
    x = np.asarray(series, dtype=float)
    isnan = np.isnan(x)
    if (~isnan).sum() < 24:
        # Never return the real series into a null arm — that would gift it real information
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
    """Independent IAAFT surrogate per basin column; NaN pattern (gaps) is preserved."""
    out = {}
    for i, name in enumerate(wide.columns):
        out[name] = iaaft(wide[name].values, seed=seed * 100_003 + i)
    return pd.DataFrame(out, index=wide.index)
