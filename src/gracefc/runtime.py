"""Source-aware paths shared by command-line experiment runners.

The archived CSR run keeps its historical locations. Alternative mascon products use
namespaced processed/results/figure directories selected by ``GRACEFC_SOURCE``.
"""
import os
from pathlib import Path

import pandas as pd


def source() -> str:
    value = os.environ.get("GRACEFC_SOURCE", "csr").lower()
    if value not in {"csr", "jpl"}:
        raise ValueError(f"unsupported GRACEFC_SOURCE: {value!r}")
    return value


def processed_dir(root: Path) -> Path:
    base = root / "data" / "processed"
    return base if source() == "csr" else base / source()


def shared_processed_dir(root: Path) -> Path:
    """Source-independent ERA5 and index tables remain in the canonical directory."""
    return root / "data" / "processed"


def results_dir(root: Path) -> Path:
    base = root / "results"
    return base if source() == "csr" else base / source()


def load_sample(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(wide TWSA matrix of kept basins, basin_meta, kept names) from a processed directory."""
    from .features import pivot_wide
    long_df = pd.read_csv(data_dir / "basin_month_twsa_global.csv", parse_dates=["date"])
    meta = pd.read_csv(data_dir / "basin_meta.csv")
    keep = meta[meta["exclude_reason"] == "keep"]["name"]
    return pivot_wide(long_df[long_df["name"].isin(keep)]), meta, keep


def load_era5(keep: pd.Series, root: Path | None = None) -> dict[str, pd.DataFrame]:
    """{variable: wide frame} of the shared ERA5 basin table, kept basins only."""
    from .era5 import era5_wide_by_var
    root = root or Path(__file__).resolve().parents[2]
    era5_long = pd.read_csv(shared_processed_dir(root) / "era5_basin_month.csv", parse_dates=["date"])
    return era5_wide_by_var(era5_long[era5_long["name"].isin(keep)])
