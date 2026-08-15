"""Aggregate ERA5-Land monthly forcings to basin level and build fold-safe exogenous features.

Unit traps handled here (per the download audit): monthly means of accumulated variables
(tp, e, ro, sro, ssro) come as MEAN DAILY totals in m/day, so monthly totals need a
days-in-month multiply; evaporation is sign-flipped (negative = water leaving the surface).
The basin mask grid is 0.25 deg while ERA5 was regridded to 0.5 deg, so each mask cell maps
to its nearest ERA5 cell and duplicate hits act as area weights.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

from .basins import load_basin_masks
from .decompose import deseasonalize, fit_climatology

# var name in file -> (output column, kind, scale). Kinds: 'accum' means mean daily rate in
# m/day (needs days-in-month), 'state' means an instantaneous monthly mean kept as-is.
VAR_SPEC = {
    "tp": ("precip_cm", "accum", 100.0),
    "e": ("evap_cm", "accum", -100.0),  # flip sign so positive = evaporative loss
    "ro": ("runoff_cm", "accum", 100.0),
    "sro": ("surf_runoff_cm", "accum", 100.0),
    "ssro": ("subsurf_runoff_cm", "accum", 100.0),
    "swvl1": ("swvl1", "state", 1.0),
    "swvl2": ("swvl2", "state", 1.0),
    "swvl3": ("swvl3", "state", 1.0),
    "swvl4": ("swvl4", "state", 1.0),
    "t2m": ("t2m_c", "state", 1.0),  # Kelvin offset handled separately
    "sd": ("swe_cm", "state", 100.0),
}

ERA5_FEATURES = [spec[0] for spec in VAR_SPEC.values()]

# Winsorization bound for standardized anomalies (see era5_fold_features docstring)
CLIP_SIGMA = 10.0


def _era5_cell_map(mask_lat: np.ndarray, mask_lon: np.ndarray,
                   era5_lat: np.ndarray, era5_lon: np.ndarray) -> np.ndarray:
    """Flat 0.25-deg mask cell index -> flat 0.5-deg ERA5 cell index, nearest-cell."""
    # ERA5 lat runs 90 -> -90 (step -0.5), lon 0 -> 359.5; mask centers are offset by 0.125
    # so no mask cell is ever equidistant between two ERA5 cells
    lat_idx = np.rint((era5_lat[0] - mask_lat) / 0.5).astype(int)
    lon_idx = np.rint(mask_lon / 0.5).astype(int) % len(era5_lon)
    n_lon = len(era5_lon)
    flat = lat_idx[:, None] * n_lon + lon_idx[None, :]
    return flat.ravel()


def _basin_weight_matrix(bm: dict, cell_map: np.ndarray, n_era5_cells: int) -> sparse.csr_matrix:
    """Sparse (basin x era5_cell) weights: cos-lat mask weights accumulated per target cell."""
    rows, cols, vals = [], [], []
    for b, (idx, w) in enumerate(zip(bm["indices"], bm["weights"])):
        rows.append(np.full(idx.size, b))
        cols.append(cell_map[idx])
        vals.append(w)
    mat = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(bm["indices"]), n_era5_cells),
    )
    return mat.tocsr()


def build_era5_basin_table(era5_dir: Path, mask_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate every ERA5 file to per-basin monthly means. Returns (long_df, coverage).

    long_df: one row per (date, name) with one column per converted variable.
    coverage: per-basin fraction of mask weight landing on valid (land) ERA5 cells —
    ocean-heavy basins aggregate from a sliver of coast and should be screened downstream.
    """
    bm = load_basin_masks(mask_path)
    names = bm["meta"]["name"].values
    mask_lat, mask_lon = _mask_axes(mask_path)
    files = sorted(era5_dir.glob("era5_land_monthly_*.nc"))
    if not files:
        raise FileNotFoundError(f"no ERA5 files under {era5_dir}")
    # A missing interior year is caught by the month-continuity guard below, but a
    # missing FIRST or LAST year silently shrinks the span (the guard builds its
    # expectation from the surviving data). Contiguous filename years close the
    # interior hole at file granularity and make the span visible in the log.
    years = sorted(int(f.stem.rsplit("_", 1)[1]) for f in files)
    gaps = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if gaps:
        raise ValueError(f"ERA5 year files missing from {era5_dir}: {gaps}")
    print(f"ERA5 ingestion span: {years[0]}-{years[-1]} ({len(files)} files)", flush=True)

    W = None
    frames = []
    coverage = None
    for f in files:
        ds = xr.open_dataset(f)
        dates = pd.DatetimeIndex(ds["valid_time"].values).to_period("M").to_timestamp()
        if W is None:
            cell_map = _era5_cell_map(mask_lat, mask_lon,
                                      ds["latitude"].values, ds["longitude"].values)
            n_cells = ds["latitude"].size * ds["longitude"].size
            W = _basin_weight_matrix(bm, cell_map, n_cells)
            w_total = np.asarray(W.sum(axis=1)).ravel()
        days = np.array([pd.Timestamp(d).days_in_month for d in dates], dtype=float)

        cols = {}
        for var, (out_name, kind, scale) in VAR_SPEC.items():
            data = ds[var].values.reshape(len(dates), -1)
            valid = np.isfinite(data)
            num = W @ np.where(valid, data, 0.0).T  # (basin, time)
            den = W @ valid.T.astype(float)
            with np.errstate(invalid="ignore"):
                val = np.where(den > 0, num / den, np.nan)
            if coverage is None:
                coverage = pd.DataFrame({
                    "name": names,
                    "era5_coverage": np.where(w_total > 0, den[:, 0] / w_total, 0.0),
                })
            if kind == "accum":
                val = val * days[None, :] * scale
            else:
                val = val * scale
                if out_name == "t2m_c":
                    val = val - 273.15
            cols[out_name] = val
        ds.close()

        for t, date in enumerate(dates):
            row = {"date": date, "name": names}
            row.update({c: v[:, t] for c, v in cols.items()})
            frames.append(pd.DataFrame(row))
        print(f"{f.name}: {len(dates)} months", flush=True)

    long_df = pd.concat(frames, ignore_index=True).sort_values(["name", "date"]).reset_index(drop=True)
    # Guard against silent month gaps between yearly files: lag arithmetic assumes continuity
    months = pd.DatetimeIndex(sorted(long_df["date"].unique()))
    expect = pd.date_range(months.min(), months.max(), freq="MS")
    missing = expect.difference(months)
    if len(missing):
        raise ValueError(f"ERA5 monthly axis has holes: {list(missing[:5])} ...")
    return long_df, coverage


def _mask_axes(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dm = xr.open_dataset(mask_path, decode_times=False)
    return dm["lat"].values, dm["lon"].values


def era5_wide_by_var(long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long table -> {variable: (date x basin) wide frame} on a gapless monthly index."""
    out = {}
    for col in ERA5_FEATURES:
        wide = long_df.pivot(index="date", columns="name", values=col).sort_index()
        full = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
        out[col] = wide.reindex(full)
    return out


def era5_fold_features(
    era5_wide: dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    names: list[str],
    lags: tuple[int, ...] = (0, 1, 2),
) -> tuple[pd.DataFrame, list[str]]:
    """Deseasonalized, standardized ERA5 lag features keyed by (issue_date, name).

    Mirrors the target treatment exactly: trend+harmonics fit on pre-test months only, then
    per-basin train-window std scaling, so no test-window statistic ever touches a feature.
    Near-constant series (snow in the tropics, desert soil moisture) get a zero feature
    instead of a 0/0 blowup that would silently drop their rows. All standardized values are
    winsorized at +-CLIP_SIGMA: a desert basin whose train-window subsurface runoff is ~0
    can hit tens of thousands of "sigmas" in a single flood month (Libya 2023/24 does exactly
    this), which detonates any linear head; the clip is a constant transform applied
    identically to train and test, so it stays causal.
    """
    pieces = []
    cols = []
    for var, wide in era5_wide.items():
        resid = {}
        for name in names:
            s = wide[name] if name in wide.columns else pd.Series(np.nan, index=wide.index)
            train = s[s.index < test_start]
            if train.notna().sum() < 36:
                resid[name] = pd.Series(np.nan, index=wide.index)
                continue
            r = deseasonalize(s, fit_climatology(train))
            std = r[r.index < test_start].std()
            resid[name] = (r / std).clip(-CLIP_SIGMA, CLIP_SIGMA) if std > 1e-8 else r * 0.0
        rw = pd.DataFrame(resid)
        for k in lags:
            col = f"{var}_l{k}"
            cols.append(col)
            stacked = rw.shift(k).stack(future_stack=True).rename(col)
            pieces.append(stacked)
    feats = pd.concat(pieces, axis=1).reset_index()
    feats.columns = ["issue_date", "name"] + cols
    return feats, cols
