"""Invariants of the processed CSR basin tables and prediction files.

Each test is skipped when its input file is absent, for example in a checkout
without the data archive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASIN_TABLE = ROOT / "data" / "processed" / "basin_month_twsa_global.csv"
KALMAN_PRED = ROOT / "results" / "kalman_predictions.csv"
META = ROOT / "data" / "processed" / "basin_meta.csv"


def _need(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"{p.name} not present")


def test_month_axis_has_257_solutions():
    """The basin table has 257 unique solution months, including November 2011 and May
    2015.

    Assigning each solution to the month of its time-span midpoint puts these two
    solutions in neighboring months, which leaves 255 distinct months with duplicates.
    """
    _need(BASIN_TABLE)
    df = pd.read_csv(BASIN_TABLE, usecols=["date"], parse_dates=["date"])
    months = pd.DatetimeIndex(sorted(df["date"].unique()))
    assert len(months) == 257
    assert pd.Timestamp("2011-11-01") in months
    assert pd.Timestamp("2015-05-01") in months
    assert months.is_unique


def test_kalman_predictions_keys_and_windows():
    """Kalman prediction rows have unique keys, targets h months after issue, and
    the expected issue windows."""
    _need(KALMAN_PRED)
    df = pd.read_csv(KALMAN_PRED, parse_dates=["issue_date", "target_date"])
    keys = ["model", "fold", "horizon", "name", "issue_date"]
    assert not df.duplicated(subset=keys).any()
    expect = df["issue_date"] + df["horizon"].map(lambda h: pd.DateOffset(months=h))
    assert (df["target_date"] == expect).all()
    # The issue window shrinks with lead time: 83 issue months at 1 month down to 78 at
    # 6 months
    counts = df.groupby("horizon")["issue_date"].nunique()
    assert counts.to_dict() == {1: 83, 2: 82, 3: 81, 4: 80, 5: 79, 6: 78}
    # No target falls after the last month of the record
    assert df["target_date"].max() == pd.Timestamp("2026-05-01")


def test_meta_continent_counts():
    """The sample has 234 kept basins (199 of at least 90 000 km2), and Indonesian
    basins are in Asia."""
    _need(META)
    m = pd.read_csv(META)
    keep = m[m["exclude_reason"] == "keep"]
    assert len(keep) == 234
    assert (keep["area_km2"] >= 90_000).sum() == 199
    for b in ("C_Java", "C_Sulawesi", "C_Maluku", "C_Nusa_Tenggara"):
        row = m[m["name"] == b]
        if len(row):
            assert row["continent"].iat[0] == "asia", b


def test_cross_model_target_agreement():
    """Two prediction files must record the same observed target for a shared (name,
    target_date, horizon)."""
    _need(KALMAN_PRED)
    r0 = ROOT / "results" / "kalman_r0_predictions.csv"
    _need(r0)
    a = pd.read_csv(KALMAN_PRED, parse_dates=["target_date"])
    b = pd.read_csv(r0, parse_dates=["target_date"])
    merged = a.merge(b, on=["name", "target_date", "horizon"], suffixes=("_a", "_b"))
    assert len(merged) > 0
    assert np.allclose(merged["target_a"], merged["target_b"], atol=1e-8)
