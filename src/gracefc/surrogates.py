"""IAAFT surrogates: keep each series' distribution and spectrum, destroy cross-basin synchrony."""
import numpy as np
import pandas as pd


def iaaft(series: np.ndarray, seed: int, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate of one series (NaN-safe)."""
    x = np.asarray(series, dtype=float)
    isnan = np.isnan(x)
    v = x[~isnan]
    if v.size < 24:
        # Never return the real series into a null arm — that would gift it real information
        raise ValueError(f"series too short to surrogate ({v.size} obs); exclude it upstream")
    rng = np.random.default_rng(seed)
    sorted_vals = np.sort(v)
    target_amp = np.abs(np.fft.rfft(v))
    y = rng.permutation(v)
    for _ in range(n_iter):
        # Impose the spectrum, then restore the exact amplitude distribution
        spec = np.fft.rfft(y)
        phases = np.angle(spec)
        y = np.fft.irfft(target_amp * np.exp(1j * phases), n=v.size)
        rank = np.argsort(np.argsort(y))
        y = sorted_vals[rank]
    out = np.full_like(x, np.nan)
    out[~isnan] = y
    return out


def surrogate_wide(wide: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Independent IAAFT surrogate per basin column; NaN pattern (gaps) is preserved."""
    out = {}
    for i, name in enumerate(wide.columns):
        out[name] = iaaft(wide[name].values, seed=seed * 100_003 + i)
    return pd.DataFrame(out, index=wide.index)
