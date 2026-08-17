"""Regression tests for the invariants repaired in the 2026-08 audits.

Each test pins a defect that actually occurred: if it fails, one of the repaired
bugs has been reintroduced. Fast, no heavy compute; data-dependent checks live in
test_data_invariants.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc import stats  # noqa: E402
from gracefc.basins import _assign_continent  # noqa: E402
from gracefc.evaluate import Fold, split_fold  # noqa: E402
from gracefc.kalman import fit_kalman_ar1, kalman_forecast_series  # noqa: E402
from gracefc.surrogates import iaaft  # noqa: E402


# ---------------------------------------------------------------- fold membership
def test_split_fold_membership_invariants():
    """P0-2 (2026-08-13): test membership is by ISSUE date (a model frozen at
    test_start is only honest for forecasts issued at or after it); training rows
    must have their target OBSERVED before the freeze. A row issued pre-freeze
    whose target lands inside the test window belongs to neither set."""
    fold = Fold("f", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))
    df = pd.DataFrame({
        "issue_date": pd.to_datetime(["2019-07-01", "2019-12-01", "2020-01-01",
                                      "2020-06-01", "2020-07-01"]),
        "target_date": pd.to_datetime(["2019-12-01", "2020-05-01", "2020-07-01",
                                       "2020-11-01", "2020-08-01"]),
        "x": range(5),
    })
    tr, te = split_fold(df, fold)
    assert list(te["issue_date"]) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")]
    # Train: target observed strictly before the freeze
    assert list(tr["issue_date"]) == [pd.Timestamp("2019-07-01")]
    assert (tr["target_date"] < fold.test_start).all()
    # The straddling row (issued pre-freeze, target in the test window) is in
    # neither set, and post-window issues are dropped too
    in_any = set(tr["issue_date"]) | set(te["issue_date"])
    assert pd.Timestamp("2019-12-01") not in in_any
    assert pd.Timestamp("2020-07-01") not in in_any


# ---------------------------------------------------------------- paired-row stats
def test_paired_losses_rejects_target_disagreement():
    """P1-5: pairing must assert target equality, not silently intersect."""
    rows = pd.DataFrame({
        "model": ["a", "b"], "name": ["X", "X"],
        "target_date": [pd.Timestamp("2020-01-01")] * 2,
        "target": [1.0, 2.0], "pred": [0.0, 0.0],
    })
    with pytest.raises(AssertionError, match="targets differ"):
        stats._paired_losses(rows, "a", "b")


def test_paired_losses_rejects_row_set_mismatch():
    """Audit 2026-08-17: both models present but on different rows must raise,
    not silently score each on the intersection."""
    rows = pd.DataFrame({
        "model": ["a", "a", "b"], "name": ["X", "Y", "X"],
        "target_date": [pd.Timestamp("2020-01-01")] * 3,
        "target": [1.0, 1.0, 1.0], "pred": [0.0, 0.0, 0.0],
    })
    with pytest.raises(AssertionError, match="row sets differ"):
        stats._paired_losses(rows, "a", "b")


def test_paired_losses_empty_model_is_not_an_error():
    """A model absent from the slice (not run at this horizon) yields an empty
    frame — that is coverage, not a pairing defect — and every stats entry
    point degrades to NaN/empty on it rather than crashing (audit 2026-08-17)."""
    rows = pd.DataFrame({
        "model": ["a"], "name": ["X"], "horizon": [1],
        "target_date": [pd.Timestamp("2020-01-01")],
        "target": [1.0], "pred": [0.0],
    })
    j = stats._paired_losses(rows, "a", "b")
    assert len(j) == 0
    stat, p = stats.pooled_monthly_dm(rows, "a", "b", 1)
    assert np.isnan(stat) and np.isnan(p)
    point, lo, hi = stats.block_bootstrap_skill_ci(rows, "a", "b", 1, n_boot=10)
    assert np.isnan(point)
    fdr = stats.per_basin_dm_fdr(rows, "a", "b", 1)
    assert len(fdr) == 0 and "significant" in fdr.columns


def test_bootstrap_block_scales_with_horizon():
    """Audit 2026-08-15: CI block length must cover lag-(h-1) overlap dependence."""
    import inspect
    src = inspect.getsource(stats.block_bootstrap_skill_ci)
    assert "max(3, horizon)" in src
    sig = inspect.signature(stats.block_bootstrap_skill_ci)
    assert sig.parameters["block"].default is None


# ---------------------------------------------------------------- placebo seeding
def test_no_offset_placebo_model_seeds():
    """Audit 2026-08-15 blocker 3: placebo heads must reuse the real arm's model
    seed; '1000 + seed' offsets change init/shuffle/early-stop with the graph."""
    engines = list((ROOT / "src" / "gracefc").glob("experiment_*.py"))
    assert engines, "no experiment engines found"
    offenders = [e.name for e in engines if "1000 + seed" in e.read_text(encoding="utf-8")]
    assert not offenders, f"placebo seed offset reintroduced in: {offenders}"


def test_placebo_draws_seeded_per_cell():
    """Placebo graph draws must vary by (fold, horizon) so leads are independent."""
    for name in ("experiment_lstm_combined", "experiment_resmlp", "experiment_lstm",
                 "experiment_gnn", "experiment_era5", "experiment_nonlinear"):
        src = (ROOT / "src" / "gracefc" / f"{name}.py").read_text(encoding="utf-8")
        assert "{fold.name}:h{h}" in src, f"{name}: per-cell placebo seeding removed"


# ---------------------------------------------------------------- kalman
def test_fixed_r_zero_degenerates_to_damped_persistence():
    """r=0 forces gain 1: the filtered state must track the raw observation."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal(120)
    y[40:43] = np.nan
    rho, q, r = fit_kalman_ar1(y, fixed_r=0.0)
    assert r == 0.0
    filt = kalman_forecast_series(y, rho, q, 0.0)
    obs = ~np.isnan(y)
    np.testing.assert_allclose(filt[obs], y[obs], atol=1e-12)
    # Through a gap the state propagates by rho
    assert filt[41] == pytest.approx(rho * filt[40])


def test_free_r_actually_filters():
    """With observation noise present, the MLE filter must NOT track raw obs."""
    rng = np.random.default_rng(1)
    n, rho_true = 400, 0.8
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho_true * x[i - 1] + rng.standard_normal() * 0.5
    y = x + rng.standard_normal(n) * 1.0
    rho, q, r = fit_kalman_ar1(y)
    assert r > 0.1, "observation noise not recovered on a noisy series"
    filt = kalman_forecast_series(y, rho, q, r)
    assert np.mean((filt - y) ** 2) > 0.05, "filter is tracking raw observations"


# ---------------------------------------------------------------- surrogates
def test_iaaft_preserves_mask_and_monthly_axis():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(290)
    gaps = rng.choice(290, 33, replace=False)
    x[gaps] = np.nan
    y = iaaft(x, seed=3)
    assert np.array_equal(np.isnan(x), np.isnan(y))
    obs = ~np.isnan(x)
    assert not np.allclose(x[obs], y[obs])
    # The transform must run on the full-length grid: output length == input length
    # and the surrogate is NOT a permutation of only the observed values compressed
    assert y.size == x.size


def test_iaaft_refuses_short_series():
    x = np.full(300, np.nan)
    x[:20] = 1.0
    with pytest.raises(ValueError, match="too short"):
        iaaft(x, seed=0)


# ---------------------------------------------------------------- continents
def test_indonesian_basins_are_asia():
    assert _assign_continent(-7.3, 110.4, "C_Java") == "asia"
    assert _assign_continent(-2.2, 121.4, "C_Sulawesi") == "asia"
    assert _assign_continent(-0.44, 101.7, "C_Sumatra") == "asia"
    # New Guinea and mainland Australia stay in the (Oceania-inclusive) group
    assert _assign_continent(-5.3, 140.5, "C_Papua_New_Guinea_Island") == "australia"
    assert _assign_continent(-25.0, 135.0, "E_Lake_Eyre_Basin") == "australia"


# ---------------------------------------------------------------- era5 download
def test_year_done_requires_flat_file(tmp_path, monkeypatch):
    """Audit 2026-08-15: an extracted directory must NOT count as downloaded —
    ingestion only sees flat era5_land_monthly_<year>.nc files."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import download_era5
    monkeypatch.setattr(download_era5, "OUT_DIR", tmp_path)
    d = tmp_path / "era5_land_monthly_2003"
    d.mkdir()
    (d / "data_0.nc").write_bytes(b"x" * 10)
    assert not download_era5.year_done(2003)
    flat = tmp_path / "era5_land_monthly_2004.nc"
    flat.write_bytes(b"\x00" * 1_100_000)
    assert download_era5.year_done(2004)
