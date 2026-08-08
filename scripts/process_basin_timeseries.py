from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import (
    GRACE_FILE,
    MASK_FOLDER,
    PROCESSED_DIR,
)


OUTPUT_FILE = PROCESSED_DIR / "basin_timeseries.csv"


def load_grace():
    print("Loading GRACE...")
    ds = xr.open_dataset(GRACE_FILE)

    lwe = ds["lwe_thickness"]

    print(f"GRACE shape: {lwe.shape}")

    return ds, lwe


def load_masks():
    masks = {}

    mask_files = sorted(MASK_FOLDER.glob("*.npz"))

    print(f"Found {len(mask_files)} masks")

    for path in mask_files:
        with np.load(path) as data:
            masks[path.stem] = data["indices"]

    return masks


def calculate_cell_areas(latitudes, lon_count):
    earth_radius = 6_371_008.8

    lat_step = 0.25
    lon_step = 0.25

    lon_step_rad = np.radians(lon_step)

    lat_min = np.radians(
        latitudes - lat_step / 2
    )

    lat_max = np.radians(
        latitudes + lat_step / 2
    )

    area_by_lat = (
        earth_radius**2
        * lon_step_rad
        * (
            np.sin(lat_max)
            - np.sin(lat_min)
        )
    )

    # Expand from:
    # 720 latitude areas
    #
    # to:
    # 720 * 1440 grid-cell areas

    areas = np.repeat(
        area_by_lat,
        lon_count
    )

    return np.asarray(areas)


def calculate_basin_timeseries(
    lwe,
    masks,
    cell_areas,
):
    time_values = lwe["time"].values

    dates = (
        pd.Timestamp("2002-01-01")
        + pd.to_timedelta(time_values, unit="D")
    )

    basin_names = sorted(masks)

    results = {
        basin: []
        for basin in basin_names
    }

    # Precalculate weights
    basin_weights = {
        basin: cell_areas[indices]
        for basin, indices in masks.items()
    }

    for t, date in enumerate(dates):

        print(
            f"[{t + 1}/{len(dates)}] "
            f"{date.date()}"
        )

        grid = np.asarray(
            lwe.isel(time=t).values
        ).ravel()

        for basin in basin_names:

            indices = masks[basin]

            values = grid[indices]
            weights = basin_weights[basin]

            valid = np.isfinite(values)

            if not np.any(valid):
                tws_cm = np.nan

            else:
                tws_cm = np.average(
                    values[valid],
                    weights=weights[valid],
                )

            results[basin].append(
                tws_cm
            )

    records = []

    for basin in basin_names:

        basin_df = pd.DataFrame(
            {
                "time": dates,
                "basin": basin,
                "tws_cm": results[basin],
            }
        )

        records.append(basin_df)

    return pd.concat(
        records,
        ignore_index=True
    )

def main():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds, lwe = load_grace()

    masks = load_masks()

    latitudes = ds["lat"].values
    lon_count = len(ds["lon"])

    print("Calculating grid-cell areas...")

    cell_areas = calculate_cell_areas(
        latitudes,
        lon_count
    )

    print("Calculating basin time series...")

    result = calculate_basin_timeseries(
        lwe,
        masks,
        cell_areas,
    )

    print("\nSaving...")

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Saved {len(result):,} rows to:"
    )

    print(OUTPUT_FILE)

    print("\nPreview:")
    print(result.head())

    ds.close()


if __name__ == "__main__":
    main()