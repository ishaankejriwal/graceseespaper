"""Phase 1: build the global basin-month TWSA table and basin metadata from CSR + L3 masks."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.basins import build_basin_series  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    long_df, meta = build_basin_series(
        ROOT / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
        ROOT / "HydroShed+Mascon_Basins_L3.nc",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(OUT_DIR / "basin_month_twsa_global.csv", index=False)
    meta.to_csv(OUT_DIR / "basin_meta.csv", index=False)

    print(f"basins: {meta.shape[0]}")
    print(f"months: {long_df['date'].nunique()} ({long_df['date'].min():%Y-%m} to {long_df['date'].max():%Y-%m})")
    print(f"rows: {long_df.shape[0]}, NaN twsa rows: {long_df['twsa_cm'].isna().sum()}")
    print("\ncontinent counts:")
    print(meta["continent"].value_counts().to_string())
    print("\nafrica basins:")
    afr = meta[meta["continent"] == "africa"].sort_values("area_km2", ascending=False)
    print(afr[["name", "area_km2", "n_cells"]].to_string(index=False))
    print("\nsmallest 10 basins globally (GRACE resolvability check):")
    print(meta.nsmallest(10, "area_km2")[["name", "area_km2", "continent"]].to_string(index=False))


if __name__ == "__main__":
    main()
