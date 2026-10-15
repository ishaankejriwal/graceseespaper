"""Data and results paths for the mascon product selected by GRACEFC_SOURCE.

GRACEFC_SOURCE is csr (the default) or jpl. CSR files live directly in data/processed and
results; JPL files live in a jpl subdirectory of each. ERA5-Land and climate-index tables
are shared by both products.
"""
import os
from pathlib import Path

import pandas as pd


def source() -> str:
    """Mascon product named by GRACEFC_SOURCE: 'csr' (default) or 'jpl'."""
    value = os.environ.get("GRACEFC_SOURCE", "csr").lower()
    if value not in {"csr", "jpl"}:
        raise ValueError(f"unsupported GRACEFC_SOURCE: {value!r}")
    return value


def processed_dir(root: Path) -> Path:
    base = root / "data" / "processed"
    return base if source() == "csr" else base / source()


def shared_processed_dir(root: Path) -> Path:
    """Directory of the ERA5-Land and climate-index tables, which do not depend on the product."""
    return root / "data" / "processed"


def results_dir(root: Path) -> Path:
    base = root / "results"
    return base if source() == "csr" else base / source()


def load_sample(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(wide TWSA matrix of kept basins in cm, basin_meta table, kept basin names)."""
    from .features import pivot_wide
    long_df = pd.read_csv(data_dir / "basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(data_dir / "basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    return pivot_wide(long_df[long_df["name"].isin(keep)]), meta, keep


def load_era5(keep: pd.Series, root: Path | None = None) -> dict[str, pd.DataFrame]:
    """ERA5-Land basin series of the kept basins, as {variable: wide date x basin frame}."""
    from .era5 import era5_wide_by_var
    root = root or Path(__file__).resolve().parents[2]
    era5_long = pd.read_csv(shared_processed_dir(root) / "era5_basin_month.csv", parse_dates=["date"])
    return era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
