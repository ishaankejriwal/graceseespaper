"""Resolution stratification on OFFICIAL CSR mascon geometry (rebuilt 2026-08-15).

History: the first version of this analysis believed no official mascon-assignment
product existed and recovered native tiles empirically (identical 8-month fingerprints
across a tile's 0.25 deg cells). The audit of 2026-08-15 found that CSR's RL06.3 page
does distribute both a native-mascon-ID mapping file and v02 land/ocean masks
(doi:10.15781/cgq9-nh24), and recommends the masks for basin definitions to minimize
coastal leakage. This script now:

1. Uses the OFFICIAL mapping file as the primary tile assignment and the OFFICIAL land
   mask as the land definition for contamination.
2. Retains the fingerprint recovery as a cross-check and reports partition agreement
   (the validation the fingerprint approach previously could not have).
3. Sweeps area thresholds as before, and repeats the decisive 2x2 at BOTH the 90,000 km2
   convention and CSR's published ~200,000 km2 caution threshold.

Outputs: resolution_diagnostics.csv (official geometry, plus *_fp fingerprint columns),
mascon_tile_inventory.csv, csr_geometry_validation.csv, resolution_sweep_area.csv,
resolution_sweep_contamination.csv, resolution_cross_2x2.csv (90k),
resolution_cross_2x2_200k.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.basins import load_basin_masks  # noqa: E402
from gracefc.stats import pooled_monthly_dm  # noqa: E402

MASCON_NC = ROOT / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
MASK_NC = ROOT / "HydroShed+Mascon_Basins_L3.nc"
CSR_DIR = ROOT / "data" / "raw" / "csr_ancillary"
MAPPING_NC = CSR_DIR / "CSR_GRACE_GRACE-FO_RL0603_mascons_mapping_file.nc"
LANDMASK_NC = CSR_DIR / "CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc"
OUT_DIR = ROOT / "results"

# Months used as the tile fingerprint for the cross-check. Eight float32 values
# agreeing exactly across two different mascons is not a credible collision.
FINGERPRINT_TIMES = (5, 40, 80, 120, 160, 200, 230, 250)
MODEL_A, MODEL_B = "kalman_corr_top1", "kalman_own_ridge"


def _align_to(ds: xr.Dataset, lat_ref: np.ndarray, lon_ref: np.ndarray) -> xr.Dataset:
    """Flip/verify a 0.25-deg global dataset onto the solutions grid axes."""
    if not np.allclose(ds["lat"].values, lat_ref):
        ds = ds.reindex(lat=lat_ref, method="nearest", tolerance=1e-3)
    if not np.allclose(ds["lon"].values, lon_ref):
        ds = ds.reindex(lon=lon_ref, method="nearest", tolerance=1e-3)
    assert np.allclose(ds["lat"].values, lat_ref) and np.allclose(ds["lon"].values, lon_ref), \
        "CSR ancillary grid does not align with the solutions grid"
    return ds


def load_official_geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tile_id 2D int, cell_km2 2D, land 2D bool) on the solutions grid, from CSR files."""
    sols = xr.open_dataset(MASCON_NC, decode_times=False)
    lat, lon = sols["lat"].values, sols["lon"].values
    sols.close()

    mp = _align_to(xr.open_dataset(MAPPING_NC), lat, lon)
    raw = mp["mascon_id"].values
    assert np.isfinite(raw).all(), "official mapping has non-finite mascon ids"
    ids = raw.astype(np.int64)
    assert np.abs(raw - ids).max() < 1e-9, "official mascon ids are not integral"
    # Compact to 0..K-1 so bincount-based aggregation stays dense
    _, tile_id = np.unique(ids.ravel(), return_inverse=True)
    tile_id = tile_id.reshape(ids.shape)
    mp.close()

    lm = _align_to(xr.open_dataset(LANDMASK_NC), lat, lon)
    land = lm["LO_val"].values >= 0.5
    lm.close()

    cell_km2_eq = (111.32 * 0.25) ** 2
    cell_km2 = np.broadcast_to(
        (np.cos(np.deg2rad(lat)) * cell_km2_eq)[:, None], tile_id.shape
    ).copy()
    return tile_id, cell_km2, land


def recover_mascon_tiles() -> np.ndarray:
    """Fingerprint recovery (cross-check only): label cells by identical 8-month series."""
    ds = xr.open_dataset(MASCON_NC, decode_times=False)
    fp = ds["lwe_thickness"].isel(time=list(FINGERPRINT_TIMES)).values  # (T, lat, lon)
    n_lat, n_lon = fp.shape[1], fp.shape[2]
    ds.close()
    flat = fp.reshape(len(FINGERPRINT_TIMES), -1).T  # (cells, T)
    contiguous = np.ascontiguousarray(flat)
    keys = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1]).ravel()
    _, tile_id = np.unique(keys, return_inverse=True)
    return tile_id.reshape(n_lat, n_lon)


def validate_partitions(off_id: np.ndarray, fp_id: np.ndarray) -> pd.DataFrame:
    """Cell-level agreement between the official and fingerprint partitions."""
    a, b = off_id.ravel(), fp_id.ravel()
    df = pd.DataFrame({"off": a, "fp": b})
    # Purity in both directions: does each cluster of one partition sit inside one
    # cluster of the other? 1.0 both ways means the partitions are identical.
    fp_major = df.groupby("fp")["off"].agg(lambda s: s.value_counts().iat[0] / len(s))
    off_major = df.groupby("off")["fp"].agg(lambda s: s.value_counts().iat[0] / len(s))
    pair_counts = df.value_counts()
    n = len(df)
    # Adjusted Rand index from the contingency counts (exact, no sklearn needed)
    sum_comb_c = float((pair_counts * (pair_counts - 1) / 2).sum())
    ai = df["off"].value_counts()
    bi = df["fp"].value_counts()
    sum_comb_a = float((ai * (ai - 1) / 2).sum())
    sum_comb_b = float((bi * (bi - 1) / 2).sum())
    total_comb = n * (n - 1) / 2
    expected = sum_comb_a * sum_comb_b / total_comb
    max_index = (sum_comb_a + sum_comb_b) / 2
    ari = (sum_comb_c - expected) / (max_index - expected)
    out = pd.DataFrame([{
        "n_official_tiles": int(df["off"].nunique()),
        "n_fingerprint_tiles": int(df["fp"].nunique()),
        "cell_weighted_purity_fp_in_off": float((fp_major * df["fp"].value_counts(normalize=False)
                                                 .reindex(fp_major.index)).sum() / n),
        "cell_weighted_purity_off_in_fp": float((off_major * df["off"].value_counts(normalize=False)
                                                 .reindex(off_major.index)).sum() / n),
        "adjusted_rand_index": float(ari),
    }])
    out.to_csv(OUT_DIR / "csr_geometry_validation.csv", index=False)
    return out


def tile_report(tile_id: np.ndarray, cell_km2: np.ndarray) -> pd.DataFrame:
    flat_id = tile_id.ravel()
    flat_km2 = cell_km2.ravel()
    n_tiles = int(flat_id.max()) + 1
    counts = np.bincount(flat_id, minlength=n_tiles)
    areas = np.bincount(flat_id, weights=flat_km2, minlength=n_tiles)
    return pd.DataFrame({"tile": np.arange(n_tiles), "n_cells": counts, "area_km2": areas})


def basin_diagnostics(tile_id, cell_km2, masks, land_cells, suffix="") -> pd.DataFrame:
    """Per-basin mascon-footprint diagnostics; `land_cells` is a flat bool array."""
    meta = masks["meta"]
    flat_id = tile_id.ravel()
    flat_km2 = cell_km2.ravel()
    n_tiles = int(flat_id.max()) + 1
    tile_total = np.bincount(flat_id, weights=flat_km2, minlength=n_tiles)
    tile_land = np.bincount(flat_id[land_cells], weights=flat_km2[land_cells], minlength=n_tiles)

    rows = []
    for b, name in enumerate(meta["name"]):
        idx = masks["indices"][b]
        if idx.size == 0:
            continue
        t = flat_id[idx]
        km2 = flat_km2[idx]
        w = np.bincount(t, weights=km2, minlength=n_tiles)
        used = np.flatnonzero(w)
        wt = w[used]
        total = wt.sum()
        fill_all = wt / tile_total[used]           # share of the whole tile inside this basin
        fill_land = wt / np.maximum(tile_land[used], 1e-9)  # share of the tile's LAND inside it
        rows.append({
            "name": name,
            f"n_tiles{suffix}": int(used.size),
            # Effective mascon count: 1.0 means the basin lives inside a single tile
            f"n_eff_tiles{suffix}": float(total ** 2 / np.sum(wt ** 2)),
            # Share of the basin's signal contributed by land outside the basin
            f"contamination{suffix}": float(np.sum((wt / total) * (1 - np.clip(fill_land, 0, 1)))),
            f"contamination_all{suffix}": float(np.sum((wt / total) * (1 - fill_all))),
            f"max_tile_fill_land{suffix}": float(np.clip(fill_land, 0, 1).max()),
            f"mascon_area_km2{suffix}": float(total),
        })
    return pd.DataFrame(rows)


def contrast(pred: pd.DataFrame, names, horizon: int) -> dict:
    sub = pred[pred["name"].isin(names)]
    # Horizon must be filtered BEFORE the pairing merge: several horizons share a
    # target_date (different issue dates), so joining on (name, target_date) alone
    # fans out across leads and silently compares h1 against h2-6.
    h = sub[sub["horizon"] == horizon]
    keys = ["name", "target_date"]
    a = h[h["model"] == MODEL_A][keys + ["target", "pred"]]
    b = h[h["model"] == MODEL_B][keys + ["target", "pred"]]
    j = a.merge(b, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(j) < 10:
        return {"n_basins": len(set(names)), "skill_pct": np.nan, "p": np.nan, "n_rows": len(j)}
    if not np.allclose(j["target_a"], j["target_b"], atol=1e-8):
        raise AssertionError("paired targets differ")
    mse_a = float(np.mean((j["target_a"] - j["pred_a"]) ** 2))
    mse_b = float(np.mean((j["target_b"] - j["pred_b"]) ** 2))
    stat, p = pooled_monthly_dm(sub, MODEL_A, MODEL_B, horizon)
    return {
        "n_basins": int(j["name"].nunique()),
        "skill_pct": 100 * (1 - mse_a / mse_b),
        "dm_stat": stat,
        "p": p,
        "n_rows": len(j),
    }


def cross_2x2(keep: pd.DataFrame, pred: pd.DataFrame, area_cut: float) -> pd.DataFrame:
    q_hi = keep["contamination"].quantile(2 / 3)
    rows = []
    for a_lab, a_sel in (("resolved", keep["area_km2"] >= area_cut),
                         ("sub_resolution", keep["area_km2"] < area_cut)):
        for c_lab, c_sel in (("cont_high", keep["contamination"] > q_hi),
                             ("cont_lowmid", keep["contamination"] <= q_hi)):
            names = keep[a_sel & c_sel]["name"]
            rows.append({"area": a_lab, "area_cut": area_cut, "contamination": c_lab,
                         **contrast(pred, names, horizon=1)})
    return pd.DataFrame(rows)


def main() -> None:
    print("loading OFFICIAL CSR mascon geometry ...")
    tile_id, cell_km2, land = load_official_geometry()
    tiles = tile_report(tile_id, cell_km2)
    print(f"  {len(tiles)} official tiles over the global grid")
    print("  tile area km2 percentiles:",
          {q: round(float(np.percentile(tiles['area_km2'], q))) for q in (1, 25, 50, 75, 99)})
    tiles.to_csv(OUT_DIR / "mascon_tile_inventory.csv", index=False)

    print("recovering fingerprint tiles for the cross-check ...")
    fp_id = recover_mascon_tiles()
    val = validate_partitions(tile_id, fp_id)
    print("  partition agreement vs official mapping:")
    print(val.to_string(index=False))

    masks = load_basin_masks(MASK_NC)
    meta = masks["meta"]
    flat_land = land.ravel()
    diag = basin_diagnostics(tile_id, cell_km2, masks, flat_land)
    # Fingerprint variant kept for the delta report ("what did the official switch change")
    legacy_land = np.zeros(tile_id.size, dtype=bool)
    for idx in masks["indices"]:
        legacy_land[idx] = True
    diag_fp = basin_diagnostics(fp_id, cell_km2, masks, legacy_land, suffix="_fp")
    meta = meta.merge(diag, on="name", how="left").merge(diag_fp, on="name", how="left")
    keep = meta[meta["exclude_reason"] == "keep"].copy()

    both = keep.dropna(subset=["contamination", "contamination_fp"])
    print(f"\ncontamination official vs fingerprint: Spearman "
          f"{both[['contamination', 'contamination_fp']].corr(method='spearman').iloc[0, 1]:.4f}, "
          f"max abs diff {float((both['contamination'] - both['contamination_fp']).abs().max()):.4f}")

    keep["area_ratio"] = keep["mascon_area_km2"] / keep["area_km2"]
    print(f"area self-consistency (should be 1.0): "
          f"min={keep.area_ratio.min():.6f} max={keep.area_ratio.max():.6f}")

    keep.to_csv(OUT_DIR / "resolution_diagnostics.csv", index=False)

    pred = pd.read_csv(ROOT / "results/phase3b_predictions.csv")

    # Sweep: area threshold (200k = CSR's published caution threshold)
    rows = []
    for cut in (40_000, 60_000, 80_000, 90_000, 100_000, 120_000, 150_000, 200_000):
        small = keep[keep["area_km2"] < cut]["name"]
        large = keep[keep["area_km2"] >= cut]["name"]
        for label, names in (("sub_resolution", small), ("resolved", large)):
            r = contrast(pred, names, horizon=1)
            rows.append({"split": "area_km2", "cut": cut, "stratum": label, **r})
    area_sweep = pd.DataFrame(rows)
    area_sweep.to_csv(OUT_DIR / "resolution_sweep_area.csv", index=False)

    # Sweep: contamination — the physical quantity, cut at fixed shares
    rows = []
    for cut in (0.10, 0.20, 0.30, 0.40, 0.50):
        hi = keep[keep["contamination"] >= cut]["name"]
        lo = keep[keep["contamination"] < cut]["name"]
        for label, names in (("contaminated", hi), ("clean", lo)):
            r = contrast(pred, names, horizon=1)
            rows.append({"split": "contamination", "cut": cut, "stratum": label, **r})
    # and by tercile, for a split that cannot be accused of threshold shopping
    q1, q2 = keep["contamination"].quantile([1 / 3, 2 / 3])
    for label, names in (
        ("tercile_low", keep[keep["contamination"] <= q1]["name"]),
        ("tercile_mid", keep[(keep["contamination"] > q1) & (keep["contamination"] <= q2)]["name"]),
        ("tercile_high", keep[keep["contamination"] > q2]["name"]),
    ):
        rows.append({"split": "contamination_tercile", "cut": np.nan, "stratum": label,
                     **contrast(pred, names, horizon=1)})
    # single-tile basins: the sharpest possible statement of the leakage worry
    for label, names in (
        ("n_eff_tiles<2", keep[keep["n_eff_tiles"] < 2]["name"]),
        ("n_eff_tiles>=2", keep[keep["n_eff_tiles"] >= 2]["name"]),
    ):
        rows.append({"split": "n_eff_tiles", "cut": 2, "stratum": label,
                     **contrast(pred, names, horizon=1)})
    cont_sweep = pd.DataFrame(rows)
    cont_sweep.to_csv(OUT_DIR / "resolution_sweep_contamination.csv", index=False)

    # The decisive 2x2 at both thresholds
    cross90 = cross_2x2(keep, pred, 90_000)
    cross90.to_csv(OUT_DIR / "resolution_cross_2x2.csv", index=False)
    cross200 = cross_2x2(keep, pred, 200_000)
    cross200.to_csv(OUT_DIR / "resolution_cross_2x2_200k.csv", index=False)

    print("\n=== h1 neighbor contrast, AREA x CONTAMINATION (2x2, 90k) ===")
    print(cross90.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== h1 neighbor contrast, AREA x CONTAMINATION (2x2, 200k = CSR guidance) ===")
    print(cross200.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nmost contaminated keep-basins:")
    print(keep.nlargest(12, "contamination")[
        ["name", "area_km2", "n_eff_tiles", "contamination"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== h1 neighbor contrast by AREA cut ===")
    print(area_sweep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== h1 neighbor contrast by MASCON CONTAMINATION ===")
    print(cont_sweep.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\ncorrelation of the two stratifiers (Spearman):",
          round(float(keep[["area_km2", "contamination"]].corr(method="spearman").iloc[0, 1]), 3))


if __name__ == "__main__":
    main()
