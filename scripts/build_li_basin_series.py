"""Basin means of the GRACE-FCast hindcast (Li and Kusche, 2026) for the CSR or JPL member.

Each 0.25-degree basin-mask cell is assigned to the 1-degree GRACE-FCast cell that
contains it, so basin weights on the coarse grid are exact sums of the fine-grid
cosine-latitude weights (Sect. 3.6). Missing cells are handled by renormalizing the
weights, as in gracefc/basins.py. Only kept basins are aggregated.

The coverage table gives, per basin, the fraction of mask weight on valid GRACE-FCast
cells (li_coverage), the number of complete valid GRACE-FCast cells (n_full_li_cells)
and the number of complete native mascons of the product being scored
(n_full_native_mascons). The last two define the fully covered basin set, with the
same rule for CSR and JPL.

Inputs: data/raw/li2026/<CSR|JPL>-FCast/global_gridded/*.nc,
HydroShed+Mascon_Basins_L3.nc, data/processed[/jpl]/basin_meta.csv, and for CSR the
official mascon mapping file read through run_resolution_sensitivity.py.
Outputs: data/processed[/jpl]/li2026_<source>_basin_forecasts.csv (full and
non-seasonal forecasts in cm, per issue month and lead) and li2026_basin_coverage.csv.

Usage:
  python scripts/build_li_basin_series.py
  python scripts/build_li_basin_series.py --source jpl
"""
import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.comparison import fully_contained_group_counts  # noqa: E402


def csr_native_tile_grid() -> np.ndarray:
    """CSR native mascon tile labels on the 0.25-degree basin-mask grid.

    Reuses run_resolution_sensitivity's official-geometry loader so there is one
    reading of CSR_GRACE_GRACE-FO_RL0603_mascons_mapping_file.nc in the repository.
    The loader picks its product from GRACEFC_SOURCE at import time, so the
    environment is pinned to csr across the import and restored afterwards.
    """
    saved = {k: os.environ.get(k) for k in ("GRACEFC_SOURCE", "GRACEFC_MASCON_FILE")}
    os.environ["GRACEFC_SOURCE"] = "csr"
    os.environ.pop("GRACEFC_MASCON_FILE", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "_resolution_geometry", ROOT / "scripts" / "run_resolution_sensitivity.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tile_id, _, _ = module.load_official_geometry()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return tile_id


def native_mascon_counts(source: str, basin_cells: list, meta: pd.DataFrame) -> np.ndarray:
    """Complete native mascons contained in each kept basin, for either product.

    A native tile counts only when every one of its 0.25-degree cells lies inside the
    basin mask, the same definition as the JPL column written by basins.py. For JPL
    the count is recorded at build time from the product's mascon_ID; CSR has no such
    field, so the count is derived here from the official RL06.3 tile mapping.
    """
    if source == "jpl":
        if "n_full_jpl_mascons" not in meta.columns:
            raise ValueError("basin_meta.csv has no n_full_jpl_mascons column")
        return meta["n_full_jpl_mascons"].to_numpy()
    tile_id = csr_native_tile_grid()
    return fully_contained_group_counts(tile_id, basin_cells)


def build_weight_matrix(
    basin_idx: np.ndarray,
    li_lat: np.ndarray,
    li_lon: np.ndarray,
    valid_li_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    """Weights, count of wholly contained valid GRACE-FCast cells, and flat mask-cell
    indices per basin.

    Reads the 1 GB mask file one basin at a time to stay within memory.
    """
    dm = xr.open_dataset(ROOT / "HydroShed+Mascon_Basins_L3.nc", decode_times=False)
    mask_lat = dm["lat"].values  # -89.875..89.875 ascending
    mask_lon = dm["lon"].values  # 0.125..359.875
    coslat = np.cos(np.deg2rad(mask_lat))
    # Containing 1-degree cell: floor(center) + 0.5; GRACE-FCast lat runs 89.5 -> -89.5
    li_lat_idx = {v: i for i, v in enumerate(li_lat)}
    li_lon_idx = {v: i for i, v in enumerate(li_lon)}
    lat_map = np.array([li_lat_idx[np.floor(v) + 0.5] for v in mask_lat])
    lon_map = np.array([li_lon_idx[np.floor(v) + 0.5] for v in mask_lon])

    n_cells = len(li_lon) * len(li_lat)  # GRACE-FCast variables are (time, lon, lat)
    W = np.zeros((len(basin_idx), n_cells))
    basin_cells = []
    for row, b in enumerate(basin_idx):
        sel = dm["mask"].isel(mask=int(b)).values > 0  # (lat, lon), ~4 MB
        lat_i, lon_i = np.nonzero(sel)
        basin_cells.append(np.flatnonzero(sel))
        li_flat = lon_map[lon_i] * len(li_lat) + lat_map[lat_i]
        np.add.at(W[row], li_flat, coslat[lat_i])
    # Each GRACE-FCast cell covers sixteen 0.25-degree mask cells. Count it only when
    # all sixteen belong to the basin and the forecast is finite.
    li_group_grid = lon_map[None, :] * len(li_lat) + lat_map[:, None]
    full_counts = fully_contained_group_counts(
        li_group_grid, basin_cells, valid_groups=valid_li_cells
    )
    dm.close()
    return W, full_counts, basin_cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("csr", "jpl"), default="csr")
    args = ap.parse_args()
    label = args.source.upper()
    data_dir = ROOT / "data" / "processed"
    if args.source != "csr":
        data_dir = data_dir / args.source
    li_dir = ROOT / "data" / "raw" / "li2026" / f"{label}-FCast" / "global_gridded"
    out = data_dir / f"li2026_{args.source}_basin_forecasts.csv"
    cov_out = data_dir / "li2026_basin_coverage.csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data_dir / "basin_meta.csv")
    # Kept basins only; excluded basins enter no comparison
    keep_meta = meta[meta["exclude_reason"] == "keep"].reset_index(drop=True)
    files = sorted(li_dir.glob(f"{label}_FCast_gridded_initialized_in_*.nc"))
    if not files:
        raise FileNotFoundError(f"no {label}-FCast files found in {li_dir}")
    print(f"{len(files)} init files | {len(keep_meta)} kept basins")

    ds0 = xr.open_dataset(files[0])
    li_lat, li_lon = ds0["lat"].values, ds0["lon"].values
    # Land coverage from the first file; the GRACE-FCast land mask is the same in every file
    finite0 = np.isfinite(ds0["TWSC_full"].values[0].reshape(-1))
    W, n_full_li_cells, basin_cells = build_weight_matrix(
        keep_meta["basin_idx"].values, li_lat, li_lon, finite0
    )
    n_full_native = native_mascon_counts(args.source, basin_cells, keep_meta)
    w_tot = W.sum(axis=1)
    ds0.close()
    coverage = (W @ finite0) / w_tot
    keep_meta = keep_meta.assign(
        li_coverage=coverage, n_full_li_cells=n_full_li_cells,
        n_full_native_mascons=n_full_native,
    )
    keep_meta[["name", "li_coverage", "n_full_li_cells",
               "n_full_native_mascons"]].to_csv(cov_out, index=False)
    print(f"coverage: min={coverage.min():.3f} | <0.9: {(coverage < 0.9).sum()} "
          f"| <0.5: {(coverage < 0.5).sum()}")
    print(f"basins containing >=1 complete valid GRACE-FCast cell: {(n_full_li_cells >= 1).sum()}")
    print(f"basins containing >=1 complete native {label} mascon: "
          f"{(np.asarray(n_full_native) >= 1).sum()}")
    joint = (np.asarray(n_full_native) >= 1) & (np.asarray(n_full_li_cells) >= 1)
    print(f"fully covered set (both): {int(joint.sum())} basins")

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
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
