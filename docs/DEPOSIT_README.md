# Data and code for "A Kalman filter benchmark for data-driven forecasts of GRACE terrestrial water storage"

Ishaan Kejriwal and Ian Wang. Deposit accompanying the manuscript submitted to
Hydrology and Earth System Sciences.

## Contents

`graceseespaper-<commit>-code.zip` is the analysis repository at the commit cited
in the manuscript: the `gracefc` package (`src/`), the experiment and figure
scripts (`scripts/`), every result table behind the manuscript (`results/*.csv`,
`results/jpl/*.csv`), the figures, and the LaTeX source. `README.md` in that
archive describes the pipeline, and `docs/ARCHIVE_MANIFEST.md` maps every
manuscript table and figure to the result file and column it is drawn from.

`graceseespaper-<commit>-data.zip` holds the files the repository excludes for size:

| Path | Content |
|---|---|
| `results/*_predictions.csv` | Row-level forecasts on the CSR mascon product: one row per model, basin, issue month and lead time, with the target and the prediction in standardized units |
| `results/*_placebo_*.csv` | Null draws for the neighbor and surrogate tests |
| `results/kalman_mission_*.csv` | Per-basin, per-fold parameters and predictions of the two-variance filter |
| `results/SHA256_MANIFEST_LIVE.csv` | SHA-256 checksums of every file above |
| `data/processed/basin_month_twsa_global.csv` | Monthly basin-average CSR TWSA (cm of equivalent water height) |
| `data/processed/basin_meta.csv` | Basin names, areas, centroids and the sample flags of Sect. 2.2 |
| `data/processed/era5_basin_month.csv`, `era5_basin_coverage.csv` | Monthly basin-average ERA5-Land variables and their cell coverage |
| `data/processed/indices.csv` | Monthly climate indices |
| `data/processed/li2026_csr_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The GRACE-FCast CSR member aggregated to our basins, and per-basin coverage |

Model names in the manuscript map to the `model` identifiers in these files
through Appendix A (Table A1) of the manuscript.

## Reproducing the figures and tables

Unpack both archives into the same directory, install the pinned environment
(`pip install -r requirements-lock.txt`, Python 3.11), verify the large files
with `python scripts/make_manifest.py --check`, and run
`python scripts/make_figures.py`. The figure script asserts every plotted
headline value against its source table. Rerunning the experiments from raw
data follows the steps in `README.md` and needs the third-party inputs below.

## Third-party data

The processed inputs are derived from: CSR RL06 mascons (Save et al., 2016;
Save, 2020); JPL RL06.3Mv04 mascons (Wiese et al., 2023); ERA5-Land
(Muñoz-Sabater et al., 2021), which contains modified Copernicus Climate
Change Service information; NOAA PSL climate indices; the GRACE-FCast hindcast
of Li and Kusche (PANGAEA, CC BY 4.0); and the HydroSHEDS basin partition
(Lehner and Grill, 2013). Cite the original products when reusing these files.

## Licence

Code: MIT. Data and result files: CC BY 4.0.
