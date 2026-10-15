# Data and code for "A Kalman filter benchmark for data-driven forecasts of GRACE terrestrial water storage"

Ishaan Kejriwal and Ian Wang. Deposit accompanying the manuscript submitted to
Hydrology and Earth System Sciences.

## Contents

The code and data archives unpack into the same folder, `graceseespaper-<commit>/`.

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
| `results/jpl/*_predictions.csv`, `results/jpl/*_placebo_basin.csv`, `results/jpl/*_placebo_monthly.csv`, `results/jpl/kalman_fold_params.pkl` | Row-level forecasts, null draws and fitted Kalman parameters (a Python pickle; load it only in the pinned environment) on the JPL mascon product, in the same layout as the CSR files above |
| `results/jpl/SHA256_MANIFEST_LIVE.csv`, `results/jpl/RUN_PROVENANCE.md` | SHA-256 checksums of the 33 large JPL files above, and the JPL run record (product release, run dates, seeds, input checksums) |
| `HydroShed+Mascon_Basins_L3.nc` | The 0.25° basin partition combining HydroSHEDS Level-3 basins with mascon boundaries (see the manuscript acknowledgements) |
| `data/processed/basin_month_twsa_global.csv` | Monthly basin-average CSR TWSA (cm of equivalent water height) |
| `data/processed/basin_meta.csv` | Basin names, areas, centroids and the exclusion flags of Sect. 2.2 |
| `data/processed/era5_basin_month.csv`, `era5_basin_coverage.csv` | Monthly basin-average ERA5-Land variables and their land-cell coverage |
| `data/processed/indices.csv` | Monthly climate indices |
| `data/processed/li2026_csr_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The GRACE-FCast CSR member aggregated by the authors to the basins above, and per-basin coverage |
| `data/processed/jpl/basin_month_twsa_global.csv`, `basin_meta.csv` | As above for the JPL RL06.3Mv04 CRI product: monthly basin-average TWSA (cm) and basin metadata with the JPL exclusion flags |
| `data/processed/jpl/li2026_jpl_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The GRACE-FCast JPL member aggregated by the authors to the basins above, and per-basin coverage |

`graceseespaper-71be5ec-jpl-run-code.zip` is the repository at commit `71be5ec`
of the JPL pipeline branch, the code that produced the JPL tables and row-level
files in `results/jpl/` (run record in `results/jpl/RUN_PROVENANCE.md`). The CSR
results and the figures come from the main code archive. Its JPL steps have changed
since this revision (for example, the archived JPL tables have no flat12 models), so
regenerate the archived JPL tables with this revision rather than the main archive.

Model names in the manuscript map to the `model` identifiers in these files
through Appendix A (Table A1) of the manuscript. Row-level JPL predictions are in
the data archive; every JPL table and figure value is also in `results/jpl/*.csv`
in the code archive.

`data/processed/jpl/li2026_basin_coverage.csv` is the file the JPL runs used and
predates the `n_full_native_mascons` column that the current comparison code
reads. To rerun the JPL comparison with this file, add that column from
`n_full_jpl_mascons` in `data/processed/jpl/basin_meta.csv` (matched on `name`),
which is what `scripts/build_li_basin_series.py --source jpl` now writes.

`data/processed/indices.csv` was downloaded again from NOAA PSL on 16 September
2026, after the runs that used it. NOAA revises some indices, so its values may
differ slightly from the run inputs. The climate indices enter one benchmark
model only.

## Reproducing the figures and tables

Install the pinned environment (`pip install -r requirements-lock.txt`,
Python 3.11), verify the large files with `python scripts/make_manifest.py --check` (and,
for the JPL files, the same command with the environment variable
`GRACEFC_SOURCE=jpl`),
and run `python scripts/make_figures.py`. The figure script asserts every
plotted headline value against its source table. It needs the Arial font, and
Cartopy downloads the Natural Earth base map on first use. Rerunning the
experiments from raw data follows the steps in `README.md` and needs the
third-party inputs below.

## Third-party data

The processed inputs are derived from: CSR RL06 mascons (Save et al., 2016;
Save, 2020); JPL RL06.3Mv04 mascons (Wiese et al., 2023); ERA5-Land
(Muñoz-Sabater et al., 2021), which contains modified Copernicus Climate
Change Service information 2026; NOAA PSL climate indices; the GRACE-FCast
hindcast of Li and Kusche (PANGAEA, CC BY 4.0), aggregated to basins by the
authors; and HydroSHEDS (Lehner and Grill, 2013). Cite the original products
when reusing these files.

## Licence

Code: MIT (see `LICENSE`), except the Copernicus LaTeX class, style and template
files under `paper/`, which keep Copernicus's terms. Data and result files: CC BY
4.0. Files derived from the third-party products above remain subject to those
products' attribution terms.
