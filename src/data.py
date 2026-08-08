import numpy as np
import pandas as pd
import xarray as xr
from src.config import GRACE_FILE, MASK_FOLDER

def load_grace():
    print(f"Loading GRACE from {GRACE_FILE}")

    ds = xr.open_dataset(GRACE_FILE)

    print("GRACE loaded:")
    print(ds.sizes)

    return ds

def load_mask_indices(mask_path):
    with np.load(mask_path) as data:
        return data["indices"]

def load_all_masks():
    masks = {}

    files = sorted(MASK_FOLDER.glob("*.npz"))

    for path in files:
        basin_name = path.stem

        masks[basin_name] = load_mask_indices(path)

    print(f"Loaded {len(masks)} masks")

    return masks