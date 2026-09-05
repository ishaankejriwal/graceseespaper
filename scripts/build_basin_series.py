"""Phase 1: build global basin-month TWSA from CSR or JPL mascons."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.basins import build_basin_series  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("csr", "jpl"), default="csr")
    ap.add_argument("--mascon-file", type=Path)
    ap.add_argument("--no-scale-factors", action="store_true",
                    help="do not apply the JPL scale_factor field")
    args = ap.parse_args()
    defaults = {
        "csr": ROOT / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
        "jpl": ROOT / "data" / "raw" /
               "GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc",
    }
    mascon_file = args.mascon_file or defaults[args.source]
    out_dir = OUT_DIR if args.source == "csr" else OUT_DIR / args.source
    long_df, meta = build_basin_series(
        mascon_file,
        ROOT / "HydroShed+Mascon_Basins_L3.nc",
        product=args.source,
        apply_scale_factors=False if args.no_scale_factors else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_dir / "basin_month_twsa_global.csv", index=False)
    meta.to_csv(out_dir / "basin_meta.csv", index=False)

    print(f"basins: {meta.shape[0]}")
    print(f"months: {long_df['date'].nunique()} ({long_df['date'].min():%Y-%m} to {long_df['date'].max():%Y-%m})")
    print(f"rows: {long_df.shape[0]}, NaN twsa rows: {long_df['twsa_cm'].isna().sum()}")
    print(f"source: {args.source}; input: {mascon_file}; output: {out_dir}")
    print("\ncontinent counts:")
    print(meta["continent"].value_counts().to_string())
    print("\nafrica basins:")
    afr = meta[meta["continent"] == "africa"].sort_values("area_km2", ascending=False)
    print(afr[["name", "area_km2", "n_cells"]].to_string(index=False))
    print("\nsmallest 10 basins globally (GRACE resolvability check):")
    print(meta.nsmallest(10, "area_km2")[["name", "area_km2", "continent"]].to_string(index=False))


if __name__ == "__main__":
    main()
