"""Monthly basin means of the 11 ERA5-Land variables (Sect. 2.3).

Aggregates the monthly files in data/raw/era5 to the basins of
HydroShed+Mascon_Basins_L3.nc with gracefc.era5.build_era5_basin_table and writes
data/processed/era5_basin_month.csv and era5_basin_coverage.csv (the fraction of each
basin's mask weight on valid ERA5-Land cells). Both products share these tables, so
there is no --source option.

Usage: python scripts/build_era5_basin_table.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.era5 import ERA5_FEATURES, build_era5_basin_table  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    long_df, coverage = build_era5_basin_table(
        ROOT / "data" / "raw" / "era5",
        ROOT / "HydroShed+Mascon_Basins_L3.nc",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_DIR / "era5_basin_month.csv", index=False)
    coverage.to_csv(OUT_DIR / "era5_basin_coverage.csv", index=False)

    print(f"\nbasins: {long_df['name'].nunique()}")
    print(f"months: {long_df['date'].nunique()} ({long_df['date'].min():%Y-%m} to {long_df['date'].max():%Y-%m})")
    print(f"rows: {long_df.shape[0]}")
    print("\nNaN fraction per variable:")
    print(long_df[ERA5_FEATURES].isna().mean().round(4).to_string())
    print("\nlowest-coverage 15 basins (fraction of mask weight on valid ERA5 land cells):")
    print(coverage.nsmallest(15, "era5_coverage").to_string(index=False))
    print("\nsanity ranges:")
    print(long_df[ERA5_FEATURES].describe().loc[["min", "mean", "max"]].T.round(3).to_string())


if __name__ == "__main__":
    main()
