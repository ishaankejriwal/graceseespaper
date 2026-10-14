# Data and code for "A Kalman filter benchmark for data-driven forecasts of GRACE terrestrial water storage"

Ishaan Kejriwal and Ian Wang. Deposit accompanying the manuscript submitted to
Hydrology and Earth System Sciences.

## Contents

Both archives unpack into the same folder, `graceseespaper-<commit>/`.

`graceseespaper-<commit>-code.zip` is the analysis repository at the commit cited
in the manuscript: the `gracefc` package (`src/`), the experiment and figure
scripts (`scripts/`), every result table behind the manuscript (`results/*.csv`,
`results/jpl/*.csv`), the figures, and the LaTeX source. `README.md` in that
archive describes the pipeline, and `docs/ARCHIVE_MANIFEST.md` maps every
manuscript table and figure to the result file and column it is drawn from.

`graceseespaper-<commit>-data.zip` holds the files the repository excludes for size:

| Path | Content |
|---|---|
| `results/*_predictions.csv` | Row-level forecasts on the CSR mascon product: one row per model, basin, issue month and lead time. Predictions and targets are in standardized units; `phase2_baseline_predictions.csv` also gives them in cm (`target`, `pred`) next to the standardized columns |
| `results/*_placebo_basin.csv`, `results/phase3b_placebo_monthly.csv` | Null draws for the neighbor and surrogate tests (the other placebo tables are small and are in the code archive) |
| `results/SHA256_MANIFEST_LIVE.csv` | SHA-256 checksums of the 25 large result files above |
| `results/jpl/` files listed in `results/jpl/SHA256_MANIFEST_LIVE.csv` | JPL result files, including row-level predictions, placebo null draws and saved model state, with their checksum manifest and `RUN_PROVENANCE.md` |
| `HydroShed+Mascon_Basins_L3.nc` | The 0.25° basin partition combining HydroSHEDS Level-3 basins with mascon boundaries (see the manuscript acknowledgements) |
| `data/processed/basin_month_twsa_global.csv` | Monthly basin-average CSR TWSA (cm of equivalent water height) |
| `data/processed/basin_meta.csv` | Basin names, areas, centroids and the exclusion flags of Sect. 2.2 |
| `data/processed/era5_basin_month.csv`, `era5_basin_coverage.csv` | Monthly basin-average ERA5-Land variables and their land-cell coverage |
| `data/processed/indices.csv` | Monthly climate indices |
| `data/processed/li2026_csr_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The GRACE-FCast CSR member aggregated by the authors to the basins above, and per-basin coverage |
| `data/processed/jpl/basin_month_twsa_global.csv`, `basin_meta.csv` | Monthly basin-average JPL TWSA and JPL basin metadata |
| `data/processed/jpl/li2026_jpl_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The GRACE-FCast JPL member aggregated to basins, and per-basin coverage |

Model names in the manuscript map to the `model` identifiers in these files
through Appendix A (Table A1) of the manuscript. Row-level predictions for the
JPL product are included in the data archive; compact JPL result tables are
also included in `results/jpl/*.csv` in the code archive.

`data/processed/indices.csv` was downloaded again from NOAA PSL on 16 September
2026, after the runs that used it. NOAA revises some indices, so its values may
differ slightly from the run inputs. The climate indices enter one benchmark
model only.

## Reproducing the figures and tables

Install the pinned environment (`pip install -r requirements-lock.txt`,
Python 3.11), verify the large files with `python scripts/make_manifest.py --check`,
and run `python scripts/make_figures.py`. The figure script asserts every
plotted headline value against its source table. It needs the Arial font, and
Cartopy downloads the Natural Earth base map on first use. Rerunning the
experiments from raw data follows the steps in `README.md` and needs the
third-party inputs below.

## Third-party data

The processed inputs are derived from: CSR RL06 mascons (Save et al., 2016;
Save, 2020); JPL RL06.3Mv04 mascons (Wiese et al., 2023); ERA5-Land
(Muñoz-Sabater et al., 2021), which contains modified Copernicus Climate
Change Service information [2026]; NOAA PSL climate indices; the GRACE-FCast
hindcast of Li and Kusche (PANGAEA, CC BY 4.0), aggregated to basins by the
authors; and HydroSHEDS (Lehner and Grill, 2013). Cite the original products
when reusing these files.

## Licence

Code: MIT. Data and result files: CC BY 4.0.
