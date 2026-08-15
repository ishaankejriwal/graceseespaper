"""Resolution stratification, audited three ways (2026-08-15).

The manuscript's primary sensitivity splits basins at 90,000 km2 ("one GRACE
resolution element"). Both inputs are ours, not CSR's: the area is our own
cos-lat integration of the mask file, and the threshold is a convention. This
script tests whether the P0-3 stratification survives that.

1. Cross-check  - the mask file carries no area or mascon-ID variable, so there
   is nothing external to validate against. Instead we recover the NATIVE mascon
   tiles empirically: the gridded product replicates one tile value across its
   0.25 deg cells, so cells sharing a tile have bit-identical time series.
   Tile-size statistics are the validation that the recovery worked.
2. Mascon sharing - the physical leakage quantity is not area but how much of a
   basin's signal comes from land outside it. With tiles recovered we can
   compute that directly (`contamination`).
3. Threshold sweep - the h1 neighbor contrast at a range of area cuts, and the
   same contrast stratified by contamination, to show whether 35 is a knife edge.
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
OUT_DIR = ROOT / "results"

# Months used as the tile fingerprint. Eight float32 values agreeing exactly
# across two different mascons is not a credible collision.
FINGERPRINT_TIMES = (5, 40, 80, 120, 160, 200, 230, 250)
MODEL_A, MODEL_B = "kalman_corr_top1", "kalman_own_ridge"


def recover_mascon_tiles() -> tuple[np.ndarray, np.ndarray]:
    """Label every 0.25 deg cell with its native mascon id. Returns (tile_id 2D, cell km2 2D)."""
    ds = xr.open_dataset(MASCON_NC, decode_times=False)
    fp = ds["lwe_thickness"].isel(time=list(FINGERPRINT_TIMES)).values  # (T, lat, lon)
    lat = ds["lat"].values
    n_lat, n_lon = fp.shape[1], fp.shape[2]
    flat = fp.reshape(len(FINGERPRINT_TIMES), -1).T  # (cells, T)
    # Exact-equality grouping; view the row bytes as one opaque key so unique is O(n log n)
    contiguous = np.ascontiguousarray(flat)
    keys = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1]).ravel()
    _, tile_id = np.unique(keys, return_inverse=True)
    tile_id = tile_id.reshape(n_lat, n_lon)

    cell_km2_eq = (111.32 * 0.25) ** 2
    cell_km2 = np.broadcast_to(
        (np.cos(np.deg2rad(lat)) * cell_km2_eq)[:, None], (n_lat, n_lon)
    ).copy()
    return tile_id, cell_km2


def tile_report(tile_id: np.ndarray, cell_km2: np.ndarray) -> pd.DataFrame:
    flat_id = tile_id.ravel()
    flat_km2 = cell_km2.ravel()
    n_tiles = int(flat_id.max()) + 1
    counts = np.bincount(flat_id, minlength=n_tiles)
    areas = np.bincount(flat_id, weights=flat_km2, minlength=n_tiles)
    return pd.DataFrame({"tile": np.arange(n_tiles), "n_cells": counts, "area_km2": areas})


def basin_diagnostics(tile_id, cell_km2, masks) -> pd.DataFrame:
    """Per-basin mascon-footprint diagnostics on the full 284-unit mask set."""
    meta = masks["meta"]
    flat_id = tile_id.ravel()
    flat_km2 = cell_km2.ravel()
    n_tiles = int(flat_id.max()) + 1
    tile_total = np.bincount(flat_id, weights=flat_km2, minlength=n_tiles)

    # Land = any cell claimed by any of the 284 mask units; a coastal tile's ocean
    # half is not "another basin's water", so report tile fill both ways.
    land_cells = np.zeros(flat_id.size, dtype=bool)
    for idx in masks["indices"]:
        land_cells[idx] = True
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
            "n_tiles": int(used.size),
            # Effective mascon count: 1.0 means the basin lives inside a single tile
            "n_eff_tiles": float(total ** 2 / np.sum(wt ** 2)),
            # Share of the basin's signal contributed by land outside the basin
            "contamination": float(np.sum((wt / total) * (1 - np.clip(fill_land, 0, 1)))),
            "contamination_all": float(np.sum((wt / total) * (1 - fill_all))),
            "max_tile_fill_land": float(np.clip(fill_land, 0, 1).max()),
            "mascon_area_km2": float(total),
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


def main() -> None:
    print("recovering native mascon tiles from the gridded product ...")
    tile_id, cell_km2 = recover_mascon_tiles()
    tiles = tile_report(tile_id, cell_km2)
    print(f"  {len(tiles)} distinct tiles over the global grid")
    print("  tile area km2 percentiles:",
          {q: round(float(np.percentile(tiles['area_km2'], q))) for q in (1, 25, 50, 75, 99)})
    print("  tile cell-count percentiles:",
          {q: float(np.percentile(tiles['n_cells'], q)) for q in (1, 25, 50, 75, 99)})
    tiles.to_csv(OUT_DIR / "mascon_tile_inventory.csv", index=False)

    masks = load_basin_masks(MASK_NC)
    meta = masks["meta"]
    diag = basin_diagnostics(tile_id, cell_km2, masks)
    meta = meta.merge(diag, on="name", how="left")
    keep = meta[meta["exclude_reason"] == "keep"].copy()

    # Cross-check 1: our cos-lat area vs the area of the same cells summed per tile
    keep["area_ratio"] = keep["mascon_area_km2"] / keep["area_km2"]
    print(f"\narea self-consistency (should be 1.0): "
          f"min={keep.area_ratio.min():.6f} max={keep.area_ratio.max():.6f}")

    keep.to_csv(OUT_DIR / "resolution_diagnostics.csv", index=False)

    pred = pd.read_csv(ROOT / "results/phase3b_predictions.csv")

    # Sweep 2: area threshold
    rows = []
    for cut in (40_000, 60_000, 80_000, 90_000, 100_000, 120_000, 150_000, 200_000):
        small = keep[keep["area_km2"] < cut]["name"]
        large = keep[keep["area_km2"] >= cut]["name"]
        for label, names in (("sub_resolution", small), ("resolved", large)):
            r = contrast(pred, names, horizon=1)
            rows.append({"split": "area_km2", "cut": cut, "stratum": label, **r})
    area_sweep = pd.DataFrame(rows)
    area_sweep.to_csv(OUT_DIR / "resolution_sweep_area.csv", index=False)

    # Sweep 3: contamination — the physical quantity, cut at fixed shares
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

    # The decisive 2x2: area and contamination are only weakly related, so cross
    # them. If contamination separates the effect INSIDE the resolved basins,
    # area is the wrong stratifier and the paper should say so.
    q_hi = keep["contamination"].quantile(2 / 3)
    rows = []
    for a_lab, a_sel in (("resolved", keep["area_km2"] >= 90_000),
                         ("sub_resolution", keep["area_km2"] < 90_000)):
        for c_lab, c_sel in (("cont_high", keep["contamination"] > q_hi),
                             ("cont_lowmid", keep["contamination"] <= q_hi)):
            names = keep[a_sel & c_sel]["name"]
            rows.append({"area": a_lab, "contamination": c_lab,
                         **contrast(pred, names, horizon=1)})
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT_DIR / "resolution_cross_2x2.csv", index=False)

    print("\n=== h1 neighbor contrast, AREA x CONTAMINATION (2x2) ===")
    print(cross.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
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
