from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

GRACE_FILE = (
    RAW_DIR
    / "grace"
    / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
)

MASK_FOLDER = (
    PROCESSED_DIR
    / "masks"
    / "africa_l3"
)

BASIN_TIMESERIES_FILE = (
    PROCESSED_DIR
    / "basin_timeseries.csv"
)

RESULTS_DIR = ROOT / "results"

LOOKBACK = 12
RANDOM_STATE = 42