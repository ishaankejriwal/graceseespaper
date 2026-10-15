"""Download global ERA5-Land monthly means at 0.5 deg from the Copernicus Climate Data Store.

Sends one request per year and writes data/raw/era5/era5_land_monthly_<year>.nc.
Years already on disk are skipped, so rerunning after a failure resumes the
download. Requires a CDS API key configured for cdsapi.

Usage: python scripts/download_era5.py
"""
import sys
import zipfile
from pathlib import Path

import cdsapi

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "era5"

VARIABLES = [
    "total_precipitation",
    "total_evaporation",
    "runoff",
    "surface_runoff",
    "sub_surface_runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "2m_temperature",
    "snow_depth_water_equivalent",
]

YEARS = list(range(2001, 2027))
# The final year is incomplete on CDS, and a request that includes an unavailable month fails
# as a whole
FINAL_YEAR_MONTHS = list(range(1, 6))


def post_process(target: Path) -> None:
    """Leave a single flat NetCDF file at `target`.

    Requests that mix accumulated and instantaneous variables can return a zip
    despite data_format=netcdf, sometimes with the variables split across two
    members (data_0.nc, data_1.nc). Ingestion reads only era5_land_monthly_*.nc
    files directly in OUT_DIR, so a zip or a subdirectory would drop that year
    from the basin table without an error. A zip is extracted to a temporary
    directory and its members are merged back into `target`.
    """
    if not zipfile.is_zipfile(target):
        return
    extract_dir = target.with_suffix("")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as z:
        z.extractall(extract_dir)
    target.unlink()
    members = sorted(extract_dir.glob("*.nc"))
    if not members:
        raise RuntimeError(f"zip for {target.name} contained no .nc members")
    if len(members) == 1:
        members[0].replace(target)
    else:
        import xarray as xr
        with xr.open_mfdataset(members, combine="by_coords") as ds:
            ds.load().to_netcdf(target)
        for m in members:
            m.unlink()
    extract_dir.rmdir()
    print(f"  (zip detected, merged {max(len(members), 1)} member(s) into {target.name})")


def year_done(year: int) -> bool:
    """True when the year's file is a flat, non-zip .nc larger than 1 MB, the only form
    ingestion reads."""
    flat = OUT_DIR / f"era5_land_monthly_{year}.nc"
    return flat.exists() and flat.stat().st_size > 1_000_000 and not zipfile.is_zipfile(flat)


def retrieve_year(client: cdsapi.Client, year: int) -> None:
    target = OUT_DIR / f"era5_land_monthly_{year}.nc"
    months = FINAL_YEAR_MONTHS if year == YEARS[-1] else list(range(1, 13))
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "time": ["00:00"],
        # Regrid to 0.5 deg on the server: the data are aggregated to basins, and the native
        # 0.1 deg grid is about 80 GB
        "grid": [0.5, 0.5],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    client.retrieve("reanalysis-era5-land-monthly-means", request, str(target))
    post_process(target)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    failures = []
    for year in YEARS:
        if year_done(year):
            print(f"SKIP {year} (already downloaded)")
            continue
        try:
            print(f"REQUEST {year} ...")
            retrieve_year(client, year)
            print(f"DONE {year}")
        except Exception as e:
            print(f"FAIL {year}: {e}")
            failures.append(year)
    if failures:
        print(f"\nFailed years (rerun to retry): {failures}")
        sys.exit(1)
    print("\nAll years downloaded.")


if __name__ == "__main__":
    main()
