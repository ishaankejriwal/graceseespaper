"""Download monthly climate indices from NOAA PSL/CPC and build one tidy CSV."""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "indices"
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "indices.csv"

# Each index maps to candidate URLs tried in order, so one dead link never kills the run
INDEX_URLS = {
    # Anomaly version, not raw SST: raw nina34 carries an annual cycle into deseasonalized models
    "nino34": ["https://psl.noaa.gov/data/correlation/nina34.anom.data"],
    "oni": ["https://psl.noaa.gov/data/correlation/oni.data"],
    "soi": ["https://psl.noaa.gov/data/correlation/soi.data"],
    "mei": ["https://psl.noaa.gov/enso/mei/data/meiv2.data"],
    "dmi": ["https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"],
    "tna": ["https://psl.noaa.gov/data/correlation/tna.data"],
    "tsa": ["https://psl.noaa.gov/data/correlation/tsa.data"],
    "amm": ["https://psl.noaa.gov/data/timeseries/monthly/AMM/ammsst.data"],
    "nao": ["https://psl.noaa.gov/data/correlation/nao.data"],
    "pna": ["https://psl.noaa.gov/data/correlation/pna.data"],
    "ao": [
        "https://psl.noaa.gov/data/correlation/ao.data",
        "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",
    ],
    "pdo": [
        "https://psl.noaa.gov/data/correlation/pdo.data",
        "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    ],
}


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "grace-research/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_psl(text: str) -> pd.Series:
    # PSL .data layout: header "year1 year2", then one row per year with 12 values, then footer lines
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split()
    y0, y1 = int(header[0]), int(header[1])
    records = {}
    missing_flag = None
    for ln in lines[1:]:
        parts = ln.split()
        # A lone numeric footer line is the missing-value flag; anything else is metadata
        if len(parts) == 1:
            try:
                missing_flag = float(parts[0])
            except ValueError:
                pass
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if not (y0 <= year <= y1) or len(parts) < 13:
            continue
        records[year] = [float(v) for v in parts[1:13]]
    ser = {}
    for year, vals in records.items():
        for m, v in enumerate(vals, start=1):
            ser[pd.Timestamp(year=year, month=m, day=1)] = v
    s = pd.Series(ser).sort_index()
    # Apply the file's own missing flag, plus common sentinels that show up unflagged
    sentinels = {missing_flag} if missing_flag is not None else set()
    sentinels |= {-99.0, -99.9, -99.99, -999.0, -999.9, -9999.0}
    s = s.where(~s.isin(list(sentinels)))
    return s


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    columns = {}
    for name, urls in INDEX_URLS.items():
        got = False
        for url in urls:
            try:
                text = fetch(url)
                (RAW_DIR / f"{name}.txt").write_text(text, encoding="utf-8")
                s = parse_psl(text)
                # Reject parses that clearly failed rather than silently keeping junk
                if s.dropna().shape[0] < 120:
                    raise ValueError(f"only {s.dropna().shape[0]} valid months")
                columns[name] = s
                print(f"OK   {name:8s} {s.dropna().index.min():%Y-%m} to {s.dropna().index.max():%Y-%m} ({s.dropna().shape[0]} months) from {url}")
                got = True
                break
            except Exception as e:
                print(f"WARN {name:8s} failed {url}: {e}")
        if not got:
            print(f"FAIL {name:8s} all sources failed — continuing without it")
    df = pd.DataFrame(columns)
    df.index.name = "date"
    df.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV} with {df.shape[1]} indices x {df.shape[0]} months")


if __name__ == "__main__":
    main()
