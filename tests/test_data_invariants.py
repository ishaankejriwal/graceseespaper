"""Data-dependent invariants on the live processed tables and prediction files.

These skip cleanly when the underlying file is absent (fresh clone without data),
and pin the repaired month-assignment, pairing, and window invariants when present.
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
    """P0-1 (2026-08-13): midpoint binning collapsed Nov 2011 / May 2015 into
    neighbors, giving 255 distinct months with duplicates. The corrected axis
    carries 257 unique solution months including both recovered ones."""
    _need(BASIN_TABLE)
    df = pd.read_csv(BASIN_TABLE, usecols=["date"], parse_dates=["date"])
    months = pd.DatetimeIndex(sorted(df["date"].unique()))
    assert len(months) == 257
    assert pd.Timestamp("2011-11-01") in months
    assert pd.Timestamp("2015-05-01") in months
    assert months.is_unique


def test_kalman_predictions_keys_and_windows():
    _need(KALMAN_PRED)
    df = pd.read_csv(KALMAN_PRED, parse_dates=["issue_date", "target_date"])
    # No duplicate (model, fold, horizon, name, issue) keys
    keys = ["model", "fold", "horizon", "name", "issue_date"]
    assert not df.duplicated(subset=keys).any()
    # Date arithmetic: target = issue + h months, exactly
    expect = df["issue_date"] + df["horizon"].map(lambda h: pd.DateOffset(months=h))
    assert (df["target_date"] == expect).all()
    # Issue windows shrink with lead: 83 issue months at h1 down to 78 at h6
    counts = df.groupby("horizon")["issue_date"].nunique()
    assert counts.to_dict() == {1: 83, 2: 82, 3: 81, 4: 80, 5: 79, 6: 78}
    # Targets never exceed the record end
    assert df["target_date"].max() == pd.Timestamp("2026-05-01")


def test_meta_continent_counts():
    _need(META)
    m = pd.read_csv(META)
    keep = m[m["exclude_reason"] == "keep"]
    assert len(keep) == 234
    assert (keep["area_km2"] >= 90_000).sum() == 199
    # Continent repair (audit 2026-08-15): Indonesian basins are not australia
    for b in ("C_Java", "C_Sulawesi", "C_Maluku", "C_Nusa_Tenggara"):
        row = m[m["name"] == b]
        if len(row):
            assert row["continent"].iat[0] == "asia", b


def test_cross_model_target_agreement():
    """Any two models sharing (name, target_date, horizon) must agree on the target."""
    _need(KALMAN_PRED)
    r0 = ROOT / "results" / "kalman_r0_predictions.csv"
    _need(r0)
    a = pd.read_csv(KALMAN_PRED, parse_dates=["target_date"])
    b = pd.read_csv(r0, parse_dates=["target_date"])
    j = a.merge(b, on=["name", "target_date", "horizon"], suffixes=("_a", "_b"))
    assert len(j) > 0
    assert np.allclose(j["target_a"], j["target_b"], atol=1e-8)
