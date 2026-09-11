"""Source-aware paths shared by command-line experiment runners.

The archived CSR run keeps its historical locations. Alternative mascon products use
namespaced processed/results/figure directories selected by ``GRACEFC_SOURCE``.
"""
import os
from pathlib import Path


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


def figures_dir(root: Path) -> Path:
    base = root / "figures"
    return base if source() == "csr" else base / source()
