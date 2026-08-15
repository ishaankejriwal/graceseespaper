"""Aggregate Li & Kusche (2026) CSR-FCast 1-degree forecasts to our L3 basin means.

Spatial matching: each 0.25-degree mask cell is assigned to the 1-degree Li cell that
contains it, so basin weights on the coarse grid are exact sums of the fine-grid
cos-lat weights. NaN-aware renormalization mirrors basins.py; per-basin coverage of
Li's land mask is reported so poorly covered coastal basins can be screened.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LI_DIR = ROOT / "data/raw/li2026/CSR-FCast/global_gridded"
OUT = ROOT / "data/processed/li2026_csr_basin_forecasts.csv"
COV_OUT = ROOT / "data/processed/li2026_basin_coverage.csv"


def build_weight_matrix(basin_idx: np.ndarray, li_lat: np.ndarray, li_lon: np.ndarray) -> np.ndarray:
    """(n_basins, n_li_cells) weight matrix on the flattened (lon, lat) Li grid.

    Streams the 1 GB mask file one basin at a time to stay within memory.
    """
    dm = xr.open_dataset(ROOT / "HydroShed+Mascon_Basins_L3.nc", decode_times=False)
    mask_lat = dm["lat"].values  # -89.875..89.875 ascending
    mask_lon = dm["lon"].values  # 0.125..359.875
    coslat = np.cos(np.deg2rad(mask_lat))
    # Containing 1-degree cell: floor(center) + 0.5; Li lat runs 89.5 -> -89.5 (descending)
    li_lat_idx = {v: i for i, v in enumerate(li_lat)}
    li_lon_idx = {v: i for i, v in enumerate(li_lon)}
    lat_map = np.array([li_lat_idx[np.floor(v) + 0.5] for v in mask_lat])
    lon_map = np.array([li_lon_idx[np.floor(v) + 0.5] for v in mask_lon])

    n_cells = len(li_lon) * len(li_lat)  # Li variables are (time, lon, lat)
    W = np.zeros((len(basin_idx), n_cells))
    for row, b in enumerate(basin_idx):
        sel = dm["mask"].isel(mask=int(b)).values > 0  # (lat, lon), ~4 MB
        lat_i, lon_i = np.nonzero(sel)
        li_flat = lon_map[lon_i] * len(li_lat) + lat_map[lat_i]
        np.add.at(W[row], li_flat, coslat[lat_i])
    dm.close()
    return W


def main() -> None:
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")
    # Only the hydrology sample: excluded basins never enter any comparison
    keep_meta = meta[meta["exclude_reason"] == "keep"].reset_index(drop=True)
    files = sorted(LI_DIR.glob("CSR_FCast_gridded_initialized_in_*.nc"))
    print(f"{len(files)} init files | {len(keep_meta)} kept basins")

    ds0 = xr.open_dataset(files[0])
    li_lat, li_lon = ds0["lat"].values, ds0["lon"].values
    W = build_weight_matrix(keep_meta["basin_idx"].values, li_lat, li_lon)
    w_tot = W.sum(axis=1)

    # Static land coverage from the first file: Li's mask is the same in every file
    finite0 = np.isfinite(ds0["TWSC_full"].values[0].reshape(-1))
    ds0.close()
    coverage = (W @ finite0) / w_tot
    keep_meta = keep_meta.assign(li_coverage=coverage)
    keep_meta[["name", "li_coverage"]].to_csv(COV_OUT, index=False)
    print(f"coverage: min={coverage.min():.3f} | <0.9: {(coverage < 0.9).sum()} "
          f"| <0.5: {(coverage < 0.5).sum()}")

    rows = []
    names = keep_meta["name"].values
    for f in files:
        init = pd.Timestamp(re.search(r"initialized_in_(\d{4}-\d{2})", f.name).group(1))
        ds = xr.open_dataset(f)
        leads = ds["lead_time"].values.astype(int)
        full = ds["TWSC_full"].values.reshape(len(leads), -1)
        nonseas = (ds["TWSC_interannual"].values + ds["TWSC_subseasonal"].values).reshape(len(leads), -1)
        finite = np.isfinite(full)
        den = W @ finite.T  # (n_basins, n_leads)
        num_full = W @ np.where(finite, full, 0.0).T
        num_ns = W @ np.where(finite, nonseas, 0.0).T
        with np.errstate(invalid="ignore", divide="ignore"):
            v_full = np.where(den > 0, num_full / den, np.nan)
            v_ns = np.where(den > 0, num_ns / den, np.nan)
        ds.close()
        for j, lead in enumerate(leads):
            target = init + pd.DateOffset(months=int(lead))
            for b in range(len(names)):
                rows.append((names[b], init, target, int(lead), v_full[b, j], v_ns[b, j]))

    df = pd.DataFrame(rows, columns=["name", "issue_date", "target_date", "horizon",
                                     "li_full_cm", "li_nonseasonal_cm"])
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
