"""Phase 0 inventory: structure, coverage, and alignment of the CSR mascon file and L3 basin mask."""
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
CSR_NC = ROOT / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
MASK_NC = ROOT / "HydroShed+Mascon_Basins_L3.nc"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    section("CSR MASCON FILE")
    ds = xr.open_dataset(CSR_NC, decode_times=False)
    print(ds)
    section("CSR time axis")
    for tvar in ("time", "Time"):
        if tvar in ds:
            t = ds[tvar]
            print(f"units: {t.attrs.get('units', '?')}, n={t.size}")
            print(f"first 5 raw: {t.values[:5]}")
            print(f"last 5 raw: {t.values[-5:]}")
    section("CSR grid")
    for c in ds.coords:
        v = ds[c]
        if v.ndim == 1 and v.size > 1:
            step = float(v[1] - v[0])
            print(f"{c}: size={v.size}, min={float(v.min()):.3f}, max={float(v.max()):.3f}, step={step:.4f}")

    section("BASIN MASK FILE")
    dm = xr.open_dataset(MASK_NC, decode_times=False)
    print(dm)
    section("Mask coverage — the global vs Africa question")
    for c in dm.coords:
        v = dm[c]
        if v.ndim == 1 and v.size > 1:
            step = float(v[1] - v[0])
            print(f"{c}: size={v.size}, min={float(v.min()):.3f}, max={float(v.max()):.3f}, step={step:.4f}")
    for name, var in dm.data_vars.items():
        print(f"\nvariable: {name}, dims={var.dims}, shape={var.shape}, dtype={var.dtype}")
        vals = var.values
        finite = vals[np.isfinite(vals)] if np.issubdtype(vals.dtype, np.floating) else vals
        uniq = np.unique(finite)
        print(f"  unique values: {uniq.size}")
        if uniq.size < 500:
            print(f"  values: {uniq[:50]}{' ...' if uniq.size > 50 else ''}")
        # Where on Earth do labeled cells actually sit — this answers global vs Africa-only
        if var.ndim == 2 and uniq.size > 1:
            lat_name = [d for d in var.dims if "lat" in d.lower()]
            lon_name = [d for d in var.dims if "lon" in d.lower()]
            if lat_name and lon_name:
                labeled = np.isfinite(vals) & (vals > 0) if np.issubdtype(vals.dtype, np.floating) else vals > 0
                lats = dm[lat_name[0]].values
                lons = dm[lon_name[0]].values
                lat_any = lats[labeled.any(axis=1)] if var.dims[0] == lat_name[0] else lats[labeled.any(axis=0)]
                lon_any = lons[labeled.any(axis=0)] if var.dims[0] == lat_name[0] else lons[labeled.any(axis=1)]
                print(f"  labeled-cell lat range: {lat_any.min():.2f} to {lat_any.max():.2f}")
                print(f"  labeled-cell lon range: {lon_any.min():.2f} to {lon_any.max():.2f}")


if __name__ == "__main__":
    main()
