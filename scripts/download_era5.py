"""Download ERA5-Land monthly means, global at 0.5 deg, one year per request with resume."""
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
# The current year is incomplete on CDS; requesting missing months fails the whole request
FINAL_YEAR_MONTHS = list(range(1, 6))


def post_process(target: Path) -> None:
    # Mixed accumulated+instantaneous requests often return a zip despite data_format=netcdf
    if zipfile.is_zipfile(target):
        extract_dir = target.with_suffix("")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as z:
            z.extractall(extract_dir)
        target.unlink()
        print(f"  (zip detected, extracted {len(list(extract_dir.iterdir()))} members to {extract_dir.name}/)")


def year_done(year: int) -> bool:
    # A year counts as done if either the flat file or an extracted zip directory exists
    flat = OUT_DIR / f"era5_land_monthly_{year}.nc"
    extracted = OUT_DIR / f"era5_land_monthly_{year}"
    if extracted.is_dir() and any(extracted.iterdir()):
        return True
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
        # 0.5 deg server-side regrid: we aggregate to basins, native 0.1 deg is ~80 GB of waste
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
