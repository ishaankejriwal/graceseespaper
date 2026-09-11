"""Regression tests for the invariants repaired in the 2026-08 audits.

Each test pins a defect that actually occurred: if it fails, one of the repaired
bugs has been reintroduced. Fast, no heavy compute; data-dependent checks live in
test_data_invariants.py.
"""
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc import stats  # noqa: E402
from gracefc.basins import (  # noqa: E402
    _assign_continent, _nearest_grid_indices, assign_solution_months,
    build_basin_series,
)
from gracefc.comparison import (  # noqa: E402
    fully_contained_group_counts, li_joint_support_names,
)
from gracefc.evaluate import Fold, split_fold  # noqa: E402
from gracefc.kalman import fit_kalman_ar1, kalman_forecast_series  # noqa: E402
from gracefc.surrogates import iaaft  # noqa: E402

# Archive comparisons need result files a bare checkout does not carry, so they skip
# by default. Set GRACEFC_REQUIRE_ARCHIVES=1 (release checks, any run that claims the
# archived numbers still reproduce) and a missing file fails instead of skipping —
# otherwise deleting a results file would silently turn the pin green.
REQUIRE_ARCHIVES = os.environ.get("GRACEFC_REQUIRE_ARCHIVES") == "1"


def _need_result(path: Path) -> None:
    if path.exists():
        return
    message = f"{path.name} not present"
    if REQUIRE_ARCHIVES:
        pytest.fail(f"{message} and GRACEFC_REQUIRE_ARCHIVES=1")
    pytest.skip(message)


def test_fully_contained_group_counts_are_literal():
    groups = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ])
    basins = [
        np.array([0, 1, 4, 5]),       # all of group 0
        np.array([0, 1, 2, 3]),       # half of both groups
        np.array([2, 3, 6, 7]),       # all of group 1
    ]
    np.testing.assert_array_equal(
        fully_contained_group_counts(groups, basins), [1, 0, 1]
    )
    np.testing.assert_array_equal(
        fully_contained_group_counts(groups, basins, valid_groups=[True, False]),
        [1, 0, 0],
    )


def test_li_comparison_requires_complete_cells_from_both_products():
    meta = pd.DataFrame({
        "name": ["good", "no_jpl", "excluded", "low_coverage"],
        "exclude_reason": ["keep", "keep", "jpl_unavailable", "keep"],
        "n_full_jpl_mascons": [1, 0, 4, 2],
    })
    coverage = pd.DataFrame({
        "name": meta["name"],
        "li_coverage": [0.9, 0.9, 0.9, 0.1],
        "n_full_li_cells": [1, 3, 5, 2],
    })
    # Coverage is diagnostic only: literal complete-cell containment determines
    # spatial support, so low_coverage still qualifies in this synthetic case.
    assert list(li_joint_support_names(meta, coverage)) == ["good", "low_coverage"]

    # 2026-09-10: the same rule now runs on CSR too, reading the product-neutral
    # native count out of the coverage table. It must win over the JPL column.
    both = coverage.assign(n_full_native_mascons=[1, 0, 4, 2])
    meta_no_jpl = meta.drop(columns=["n_full_jpl_mascons"])
    assert list(li_joint_support_names(meta_no_jpl, both)) == ["good", "low_coverage"]
    # Precedence, with BOTH columns present and DISAGREEING: the product-neutral
    # native count decides and the stale JPL column is ignored. Under the JPL column
    # this frame would qualify good + low_coverage; under the native column it is
    # no_jpl + low_coverage, so the assertion cannot pass by accident.
    disagreeing = coverage.assign(n_full_native_mascons=[0, 1, 4, 2])
    assert list(li_joint_support_names(meta, disagreeing)) == ["no_jpl", "low_coverage"]
    none_contained = coverage.assign(n_full_native_mascons=[0, 0, 0, 0])
    assert list(li_joint_support_names(meta, none_contained)) == []
    with pytest.raises(ValueError, match="native-mascon containment count"):
        li_joint_support_names(meta_no_jpl, coverage)


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


# ---------------------------------------------------------------- mascon products
def test_solution_months_use_metadata_for_jpl_without_bounds():
    ds = xr.Dataset({"time": ("time", [15.0, 74.0])})
    ds["time"].attrs["units"] = "days since 2002-01-01T00:00:00Z"
    ds.attrs.update({
        "time_coverage_start": "2002-01-01T00:00:00Z",
        "time_coverage_end": "2002-03-31T23:59:59Z",
        "months_missing": "2002-02",
    })
    got = assign_solution_months(ds, product="jpl")
    assert list(got) == [pd.Timestamp("2002-01-01"), pd.Timestamp("2002-03-01")]


def test_nearest_longitude_mapping_wraps_at_dateline():
    source = np.array([-179.75, -0.25, 0.25, 179.75])
    target = np.array([180.1, 359.9, 0.1])
    got = _nearest_grid_indices(source, target, circular=True)
    np.testing.assert_array_equal(got, [0, 1, 2])


def test_jpl_basin_aggregation_maps_grid_and_applies_scale(tmp_path):
    mask = xr.Dataset(
        {
            "mask": (("basin", "lat", "lon"), np.ones((1, 2, 4), dtype=np.int8)),
            "Name": ("basin", ["C_Test"]),
            "ID": ("basin", ["1"]),
        },
        coords={"lat": [-0.1, 0.1], "lon": [0.1, 0.2, 0.8, 0.9]},
    )
    mascon = xr.Dataset(
        {
            "lwe_thickness": (("time", "lat", "lon"), [[[1.0, 2.0]], [[2.0, 4.0]]]),
            "scale_factor": (("lat", "lon"), [[2.0, 3.0]]),
            "mascon_ID": (("lat", "lon"), [[7, 7]]),
        },
        coords={"time": [15.0, 45.0], "lat": [0.0], "lon": [0.15, 0.85]},
    )
    mascon["time"].attrs["units"] = "days since 2002-01-01T00:00:00Z"
    mascon.attrs.update({
        "time_coverage_start": "2002-01-01T00:00:00Z",
        "time_coverage_end": "2002-02-28T23:59:59Z",
        "months_missing": "",
    })
    mask_path, mascon_path = tmp_path / "mask.nc", tmp_path / "jpl.nc"
    mask.to_netcdf(mask_path)
    mascon.to_netcdf(mascon_path)
    long, meta = build_basin_series(mascon_path, mask_path, product="jpl")
    np.testing.assert_allclose(long["twsa_cm"], [4.0, 8.0])
    assert list(long["date"]) == [pd.Timestamp("2002-01-01"), pd.Timestamp("2002-02-01")]
    assert meta.loc[0, "mascon_product"] == "jpl"
    assert bool(meta.loc[0, "scale_factors_applied"])
    assert meta.loc[0, "n_full_jpl_mascons"] == 1

    # The expert non-CRI JPL file has no scale_factor and must remain supported.
    unscaled_path = tmp_path / "jpl_non_cri.nc"
    mascon.drop_vars("scale_factor").to_netcdf(unscaled_path)
    unscaled, unscaled_meta = build_basin_series(unscaled_path, mask_path, product="jpl")
    np.testing.assert_allclose(unscaled["twsa_cm"], [1.5, 3.0])
    assert not bool(unscaled_meta.loc[0, "scale_factors_applied"])

    # A JPL basin with no finite CRI-scaled observations remains in the audit
    # table but must not enter models that require a fitted climatology.
    unavailable = mascon.copy(deep=True)
    unavailable["lwe_thickness"][:] = np.nan
    unavailable_path = tmp_path / "jpl_unavailable.nc"
    unavailable.to_netcdf(unavailable_path)
    _, unavailable_meta = build_basin_series(unavailable_path, mask_path, product="jpl")
    assert unavailable_meta.loc[0, "product_valid_months"] == 0
    assert unavailable_meta.loc[0, "exclude_reason"] == "jpl_unavailable"


def test_jpl_chain_paths_are_isolated():
    spec = importlib.util.spec_from_file_location("run_chain_test", ROOT / "scripts/run_chain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    steps, default, results = module.steps_for_source("jpl")
    by_name = {step[0]: step for step in steps}
    assert results == ROOT / "results" / "jpl"
    assert "figures" not in default
    assert by_name["phase2"][3][0].parent == results
    assert by_name["phase2"][2][0].parent == ROOT / "data" / "processed" / "jpl"
    # Source-independent forcings remain shared rather than duplicated.
    assert by_name["phase6_era5"][2][2] == ROOT / "data" / "processed" / "era5_basin_month.csv"


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


# ---------------------------------------------------------------- flat-12 ridge extraction
FLAT12_PRED = ROOT / "results" / "flat12_ridge_predictions.csv"
LSTM_PRED = ROOT / "results" / "phase7_lstm_predictions.csv"
FLAT12_MODELS = ["kalman_ar1", "ridge_own_flat12", "ridge_own_era5_flat12"]


def test_flat12_module_is_torch_free():
    """The reference forecast and its ridge correction must not need torch: the
    default chain runs without it, so a stray import here would break the chain."""
    import ast
    for rel in ("src/gracefc/experiment_flat12.py", "scripts/run_flat12_ridge.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported, f"{rel} imports torch"
    # Importing the module must not pull torch in transitively either.
    import subprocess
    code = ("import sys; import gracefc.experiment_flat12; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    rc = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "src").returncode
    assert rc == 0, "importing experiment_flat12 pulled torch into sys.modules"


def test_flat12_extraction_matches_archived_lstm_predictions():
    """WP1 acceptance: the torch-free runner must reproduce the ridge arms that
    used to be emitted inside the LSTM script, row for row, at leads 1-3."""
    for p in (FLAT12_PRED, LSTM_PRED):
        _need_result(p)
    cols = ["name", "issue_date", "horizon", "model", "pred", "target"]
    new = pd.read_csv(FLAT12_PRED, usecols=cols)
    old = pd.read_csv(LSTM_PRED, usecols=cols)
    key = ["name", "issue_date", "horizon"]
    for model in FLAT12_MODELS:
        for h in (1, 2, 3):
            a = new[(new["model"] == model) & (new["horizon"] == h)][key + ["pred", "target"]]
            b = old[(old["model"] == model) & (old["horizon"] == h)][key + ["pred", "target"]]
            assert len(a) and len(a) == len(b), f"{model} h{h}: {len(a)} vs {len(b)} rows"
            j = a.merge(b, on=key, suffixes=("_new", "_old"), validate="one_to_one")
            assert len(j) == len(a), f"{model} h{h}: row keys differ"
            np.testing.assert_allclose(j["pred_new"], j["pred_old"], atol=1e-8, rtol=0)
            np.testing.assert_allclose(j["target_new"], j["target_old"], atol=1e-8, rtol=0)


# ---------------------------------------------------------------- ladder / Li rewiring
# WP2 acceptance (2026-09-10): the paper ladder and the Li comparison used to read
# their Kalman arms out of phase3b_predictions.csv, a neighbor experiment. They now
# read kalman_ar1 from the Kalman baseline and the ridge arms from the flat-12 step.
# These are the ARCHIVED values for the retained models; the rewiring must reproduce
# them exactly, because the underlying arms are bit-identical across the two files.
ARCHIVED_LADDER = {
    # model -> [(rmse_std, skill_vs_damped)] for horizons 1..6
    "damped_persistence_rho": [
        (1.076129198596155, 0.0), (1.2552401046550832, -0.0057624621016578),
        (1.3518525457519814, -0.0636242442254606), (1.43540857540113, -0.1184117688055377),
        (1.4953382020967751, -0.1196241782655471), (1.547782497187057, -0.1004865890563322)],
    "ridge_own_perbasin": [
        (1.059568079239761, 0.0305422164570878), (1.2139260711983206, 0.0593538394396406),
        (1.2825821956983448, 0.0425855112395269), (1.3432858089224078, 0.0205381861631602),
        (1.410494400554887, 0.0038238401059575), (1.4775972053827655, -0.0029447609509638)],
    "kalman_ar1": [
        (1.048995163580747, 0.0497931843097244), (1.1953584266583883, 0.0879091382607659),
        (1.2733980750571514, 0.056247837707178), (1.336299347051353, 0.0307001000885953),
        (1.395072928851735, 0.025487904017458), (1.4483805890107582, 0.0363256852628769)],
    "kalman_own_ridge": [
        (1.0498958690826776, 0.0481607193158049), (1.201365758370325, 0.0787185886995848),
        (1.270614535742653, 0.0603692513185488), (1.3204410384757217, 0.0535695983851837),
        (1.3764233749233163, 0.0513686126585214), (1.4368516855847977, 0.0516060483838594)],
}
ARCHIVED_CONTRASTS = {
    # (challenger, reference) -> [(horizon, skill, dm_p)]
    ("kalman_ar1", "damped_persistence_rho"): [(1, 0.0497931843097244, 3.5377160107071306e-12)],
    ("kalman_ar1", "damped_persistence_reg"): [
        (2, 0.0879091382607659, 6.33828330177373e-19), (3, 0.056247837707178, 5.664523568455729e-19),
        (4, 0.0307001000885953, 2.5680437700158505e-11), (5, 0.025487904017458, 7.866022927196087e-07),
        (6, 0.0363256852628769, 1.9887363373338423e-09)],
    ("kalman_own_ridge", "ridge_own_perbasin"): [
        (1, 0.0181735637774032, 0.0013313877080129), (2, 0.0205866457249006, 9.243725272687047e-05),
        (3, 0.0185747555398351, 0.0015277802390436), (4, 0.033724042893137, 2.033746048887521e-06),
        (5, 0.0477272740170985, 7.386236888497145e-07), (6, 0.0543906418964686, 3.163147502194595e-08)],
}
ARCHIVED_LI_RMSE = {  # subset all_matched, horizons 1..6
    "li_lstm_full": [1.128222592752525, 1.1579726333267129, 1.1886229174059446,
                     1.2200981521043466, 1.2547089234545583, 1.2776530815164593],
    "li_lstm_nonseas": [1.214204697212348, 1.234656699115299, 1.2523084916940863,
                        1.2669542592718506, 1.294252761178086, 1.3131849274930818],
    "kalman_ar1": [1.0424413687684628, 1.1959930241380576, 1.2818218740378262,
                   1.3513670094066286, 1.4208547964854656, 1.4779969051680628],
    "kalman_own_ridge": [1.043587739065226, 1.2030691206869095, 1.278748393471348,
                         1.333685692502684, 1.4003892728466276, 1.4657323499211594],
    "ridge_own_perbasin": [1.0560139553657455, 1.2156395066596395, 1.2956163400528848,
                           1.361945636374058, 1.4399120553657152, 1.5119718872683063],
    "damped_persistence_rho": [1.0704341667623205, 1.2542369577744898, 1.3572383044988974,
                               1.4476936499868291, 1.518176815169077, 1.574389703057876],
}
ARCHIVED_LI_HEAD = {  # (model, vs) -> [(horizon, skill, dm_p)] on all_matched
    ("li_lstm_full", "kalman_ar1"): [
        (1, -0.1713489940250443, 0.0039293787607197), (2, 0.0625690276768606, 0.1543067031131342),
        (3, 0.1401299110696139, 0.0008599954541566), (4, 0.1848398999450645, 3.3905027173023625e-05),
        (5, 0.2201939899970535, 1.83774799175164e-06), (6, 0.2527277686679133, 4.87093126351688e-08)],
    ("li_lstm_nonseas", "kalman_ar1"): [
        (1, -0.3566897431099953, 2.4684477478869477e-10), (2, -0.0657004306263175, 0.0344420571749366),
        (3, 0.0455189861241499, 0.0562758330807755), (4, 0.1210275824863984, 1.207929156636512e-06),
        (5, 0.1702661558223406, 2.5202865144227686e-09), (6, 0.2105861724781291, 4.1307747895727135e-10)],
}


def _load_result(name: str) -> pd.DataFrame:
    p = ROOT / "results" / name
    _need_result(p)
    return pd.read_csv(p)


def test_ladder_reproduces_archived_rows_without_phase3b():
    lad = _load_result("paper_baseline_ladder.csv").set_index(["model", "horizon"])
    for model, values in ARCHIVED_LADDER.items():
        for h, (rmse_std, skill) in enumerate(values, start=1):
            row = lad.loc[(model, h)]
            assert abs(row["rmse_std"] - rmse_std) < 1e-8, f"{model} h{h} rmse"
            assert abs(row["skill_vs_damped"] - skill) < 1e-8, f"{model} h{h} skill"
    # the neighbor arm is gone from the ladder entirely
    assert "kalman_corr_top1" not in lad.index.get_level_values("model")


def test_ladder_contrasts_reproduce_archived_rows():
    con = _load_result("paper_baseline_contrasts.csv").set_index(
        ["challenger", "reference", "horizon"])
    for (a, b), values in ARCHIVED_CONTRASTS.items():
        for h, skill, dm_p in values:
            row = con.loc[(a, b, h)]
            assert abs(row["skill"] - skill) < 1e-8, f"{a} vs {b} h{h} skill"
            assert abs(row["dm_p"] - dm_p) < 1e-8, f"{a} vs {b} h{h} dm_p"
    # the two flat-12 contrasts the reframed paper needs are present at every lead
    for a in ("ridge_own_flat12", "ridge_own_era5_flat12"):
        got = sorted(con.loc[(a, "kalman_ar1")].index)
        assert got == [1, 2, 3, 4, 5, 6], f"{a} vs kalman_ar1: leads {got}"


def test_li_comparison_all_matched_reproduces_archived_rows():
    summary = _load_result("phase6_li_comparison_summary.csv")
    summary = summary[summary["subset"] == "all_matched"].set_index(["model", "horizon"])
    for model, values in ARCHIVED_LI_RMSE.items():
        for h, rmse_std in enumerate(values, start=1):
            assert abs(summary.loc[(model, h), "rmse_std"] - rmse_std) < 1e-8, f"{model} h{h}"
    head = _load_result("phase6_li_comparison_headline.csv")
    head = head[head["subset"] == "all_matched"].set_index(["model", "vs", "horizon"])
    for (a, b), values in ARCHIVED_LI_HEAD.items():
        for h, skill, dm_p in values:
            row = head.loc[(a, b, h)]
            assert abs(row["skill"] - skill) < 1e-8, f"{a} vs {b} h{h} skill"
            assert abs(row["dm_p"] - dm_p) < 1e-8, f"{a} vs {b} h{h} dm_p"
    # The archived ci_lo/ci_hi at leads 4-6 predate the 2026-08-15 block-length
    # repair in stats.block_bootstrap_skill_ci (block = max(3, horizon)) and are
    # deliberately NOT pinned here; skill and DM are unaffected.
    assert "kalman_corr_top1" not in summary.index.get_level_values("model")
