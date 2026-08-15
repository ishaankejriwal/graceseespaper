"""Build basin-month TWSA series from CSR mascons and the HydroSHEDS+Mascon L3 mask."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Rough continent boxes on basin centroids, checked in order — first match wins.
# Arabia carve-out and Europe both precede Africa so Gulf basins and Sicily don't leak in.
CONTINENT_BOXES = [
    ("antarctica", -90, -60, -180, 180),
    ("greenland", 59, 84, -75, -10),
    ("asia", 12, 40, 35, 60),
    ("europe", 36, 72, -25, 60),
    ("africa", -35, 38, -18, 52),
    ("asia", 0, 82, 60, 180),
    ("asia", 36, 82, 25, 60),
    ("australia", -50, 0, 110, 180),
    ("south_america", -56, 13, -82, -33),
    ("north_america", 7, 84, -170, -50),
]


# Basins whose centroids sit in box-ambiguous strips (Red Sea, Mediterranean rim, Sinai).
# Greenland ice sheet hides under NASA GIAG drainage-system names, not "Greenland".
NAME_OVERRIDES = {
    "Antarctic": "antarctica",
    "NASA_GIAG": "greenland",
    "Iceland": "europe",
    "Baffin": "north_america",
    "West_Red_Sea": "africa",
    "South_Gulf_of_Aden": "africa",
    "Southwest_Mediterranean": "africa",
    "South_Mediterranean": "africa",
    "East_Mediterranean": "asia",
    "East_Red_Sea": "asia",
    "Sicily": "europe",
    "Svalbard": "europe",
    "Sumatra": "asia",
    "Chukchi": "asia",
}


def _assign_continent(lat: float, lon: float, name: str) -> str:
    # Name-based overrides beat boxes: coastline names are unambiguous where centroids are not
    for key, cont in NAME_OVERRIDES.items():
        if key in name:
            return cont
    for cont, la0, la1, lo0, lo1 in CONTINENT_BOXES:
        if la0 <= lat <= la1 and lo0 <= lon <= lo1:
            return cont
    return "other"


def load_basin_masks(mask_path: Path) -> dict:
    """Return per-basin flat cell indices, cos-lat weights, and a metadata table."""
    dm = xr.open_dataset(mask_path, decode_times=False)
    masks = dm["mask"].values  # (basin, lat, lon), binary
    names = dm["Name"].values
    ids = dm["ID"].values
    lat = dm["lat"].values
    lon = dm["lon"].values
    # Wrap 0-360 longitudes to -180..180 so centroids and distances behave near the meridian
    lon_wrapped = np.where(lon > 180, lon - 360, lon)
    coslat = np.cos(np.deg2rad(lat))
    w2d = np.broadcast_to(coslat[:, None], masks.shape[1:])

    cell_indices, cell_weights, meta_rows = [], [], []
    # Quarter-degree cell area at the equator, scaled by cos(lat) per cell
    cell_km2_eq = (111.32 * 0.25) * (111.32 * 0.25)
    for b in range(masks.shape[0]):
        sel = masks[b] > 0
        flat = np.flatnonzero(sel)
        w = w2d[sel]
        cell_indices.append(flat)
        cell_weights.append(w)
        lat_cells = np.broadcast_to(lat[:, None], masks.shape[1:])[sel]
        lon_cells = np.broadcast_to(lon_wrapped[None, :], masks.shape[1:])[sel]
        c_lat = float(np.average(lat_cells, weights=w))
        # Longitude needs a weighted circular mean or dateline basins land thousands of km off
        rad = np.deg2rad(lon_cells)
        c_lon = float(np.rad2deg(np.arctan2(
            np.average(np.sin(rad), weights=w), np.average(np.cos(rad), weights=w)
        )))
        area = float(np.sum(w) * cell_km2_eq)
        meta_rows.append({
            "basin_idx": b,
            "id": str(ids[b]),
            "name": str(names[b]),
            "prefix": str(names[b]).split("_")[0],
            "centroid_lat": c_lat,
            "centroid_lon": c_lon,
            "area_km2": area,
            "n_cells": int(flat.size),
        })
    meta = pd.DataFrame(meta_rows)
    meta["continent"] = [
        _assign_continent(r.centroid_lat, r.centroid_lon, r.name) for r in meta.itertuples()
    ]
    # Sample-definition columns: every exclusion is one auditable filter, never an implicit box
    reasons = []
    for r in meta.itertuples():
        if "Antarctic" in r.name or "NASA_GIAG" in r.name:
            reasons.append("ice_sheet")
        elif r.prefix == "W":
            reasons.append("water_body")
        else:
            # Explicit sentinel: empty strings round-trip through CSV as NaN and break filters
            reasons.append("keep")
    meta["exclude_reason"] = reasons
    # One ~300 km GRACE resolution element is ~90,000 km2; smaller basins are leakage receivers
    meta["below_resolution"] = (meta["area_km2"] < 90_000) & (meta["exclude_reason"] == "keep")
    # De facto ice-dominated units inside the hydrology sample: report as a separate stratum
    glaciated_keys = (
        "New_Siberian", "Svalbard", "Novaya_Zemlya", "Severnaya_Zemlya", "Iceland",
        "Franz_Josef", "Banks_Island", "Aleutians_South_Alaska", "Patagonia",
    )
    meta["glaciated"] = meta["name"].apply(lambda n: any(k in n for k in glaciated_keys))
    return {"indices": cell_indices, "weights": cell_weights, "meta": meta}


def decode_csr_time(ds: xr.Dataset) -> pd.DatetimeIndex:
    # CSR stores the units under 'Units' (capital U), which defeats xarray's auto-decoding
    units = ds["time"].attrs.get("Units", "days since 2002-01-01T00:00:00Z")
    origin = pd.Timestamp(units.split("since")[1].strip().replace("Z", ""))
    return origin + pd.to_timedelta(ds["time"].values.astype("float64"), unit="D")


def assign_solution_months(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Map each raw solution to its official calendar month, one solution per month.

    Midpoint binning mislabels irregular-span solutions: the arcs centered
    2011-10-31 and 2015-04-27 are CSR's November 2011 and May 2015 solutions,
    and averaging them into the prior month drops two real months. The file's
    own months_missing attribute enumerates exactly which months lack a
    solution, so the coverage span minus that list IS the month sequence;
    every assignment is then asserted against the solution's data span.
    """
    units = ds["time"].attrs.get("Units", "days since 2002-01-01T00:00:00Z")
    origin = pd.Timestamp(units.split("since")[1].strip().replace("Z", ""))
    missing = {
        pd.Period(m, freq="M")
        for m in re.findall(r"\d{4}-\d{2}", str(ds.attrs["months_missing"]))
    }
    start = pd.Period(pd.Timestamp(str(ds.attrs["time_coverage_start"]).replace("Z", "")), freq="M")
    end = pd.Period(pd.Timestamp(str(ds.attrs["time_coverage_end"]).replace("Z", "")), freq="M")
    months = [m for m in pd.period_range(start, end, freq="M") if m not in missing]
    n_sol = ds["time"].shape[0]
    if len(months) != n_sol:
        raise ValueError(
            f"CSR metadata inconsistent: {len(months)} expected months vs {n_sol} solutions; "
            "verify months_missing/time_coverage attributes before rebuilding"
        )
    bounds = ds["time_bounds"].values.astype("float64")
    t0 = origin + pd.to_timedelta(bounds[:, 0], unit="D")
    t1 = origin + pd.to_timedelta(bounds[:, 1], unit="D")
    for i, m in enumerate(months):
        if t1[i] < m.start_time or t0[i] > m.end_time:
            raise ValueError(
                f"solution {i} span {t0[i]:%Y-%m-%d}..{t1[i]:%Y-%m-%d} does not overlap "
                f"its assigned month {m}"
            )
    return pd.DatetimeIndex([m.to_timestamp() for m in months])


def build_basin_series(csr_path: Path, mask_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate CSR lwe_thickness into per-basin monthly means. Returns (long_df, meta)."""
    bm = load_basin_masks(mask_path)
    ds = xr.open_dataset(csr_path, decode_times=False)
    dates = assign_solution_months(ds)
    lwe = ds["lwe_thickness"]
    n_time = lwe.shape[0]

    n_basins = len(bm["indices"])
    out = np.full((n_time, n_basins), np.nan)
    # Chunk over time to keep memory bounded: one chunk is ~170 MB
    chunk = 40
    for t0 in range(0, n_time, chunk):
        t1 = min(t0 + chunk, n_time)
        block = lwe.isel(time=slice(t0, t1)).values.reshape(t1 - t0, -1)
        for b in range(n_basins):
            idx, w = bm["indices"][b], bm["weights"][b]
            vals = block[:, idx]
            # NaN-aware weighted mean: renormalize weights over valid cells per timestep
            valid = np.isfinite(vals)
            wsum = (valid * w).sum(axis=1)
            num = np.nansum(vals * w, axis=1)
            out[t0:t1, b] = np.where(wsum > 0, num / wsum, np.nan)

    df = pd.DataFrame(out, index=dates, columns=bm["meta"]["name"].values)
    # One solution per official month by construction; duplicates mean the mapping broke
    if not df.index.is_unique:
        raise ValueError("duplicate solution months after official assignment")
    long_df = df.reset_index(names="date").melt(id_vars="date", var_name="name", value_name="twsa_cm")
    long_df = long_df.merge(bm["meta"][["name", "basin_idx"]], on="name")
    return long_df.sort_values(["basin_idx", "date"]).reset_index(drop=True), bm["meta"]
