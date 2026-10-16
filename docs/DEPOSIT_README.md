# Data and code for "A Kalman filter benchmark for data-driven forecasts of GRACE terrestrial water storage"

Ishaan Kejriwal and Ian Wang. This deposit accompanies the manuscript submitted to
Hydrology and Earth System Sciences.

## Contents

The deposit has three archives. The code and data archives unpack into the same folder,
`graceseespaper-<commit>/`.

`graceseespaper-<commit>-code.zip` is the repository at the commit cited in the
manuscript, without our working notes and submission tooling. It holds the `gracefc`
package (`src/`), the experiment and figure scripts (`scripts/`), every result table
behind the manuscript (`results/*.csv`, `results/jpl/*.csv`), the figures and the LaTeX
source. `README.md` describes the pipeline, and `docs/ARCHIVE_MANIFEST.md` maps each
manuscript table and figure to the result file and column it comes from.

`graceseespaper-<commit>-data.zip` holds the files that git leaves out for size, plus a
copy of this readme:

| Path | Content |
|---|---|
| `results/*_predictions.csv` | Row-level forecasts on the CSR mascon product, one row per model, basin, issue month and lead time. Predictions and targets are in standardized units; `phase2_baseline_predictions.csv` also gives them in cm (`target`, `pred`) next to the standardized columns |
| `results/*_placebo_basin.csv`, `results/phase3b_placebo_monthly.csv` | Null draws for the neighbour and surrogate tests (the smaller placebo tables are in the code archive) |
| `results/SHA256_MANIFEST_LIVE.csv` | SHA-256 checksums of the 25 CSR files above |
| `results/jpl/*_predictions.csv`, `results/jpl/*_placebo_basin.csv`, `results/jpl/*_placebo_monthly.csv`, `results/jpl/kalman_fold_params.pkl` | The same row-level files for the JPL mascon product (33 files), including the fitted Kalman parameters as a Python pickle; load it only in the pinned environment |
| `results/jpl/SHA256_MANIFEST_LIVE.csv`, `results/jpl/RUN_PROVENANCE.md` | SHA-256 checksums of all 92 generated JPL result files (the 33 above and the tables in the code archive), and the JPL run record: product release, run dates, seeds, input checksums |
| `data/processed/basin_month_twsa_global.csv` | Monthly basin-average CSR TWSA (cm of equivalent water height) |
| `data/processed/basin_meta.csv` | Basin names, areas, centroids and the exclusion flags of Sect. 2.2 |
| `data/processed/era5_basin_month.csv`, `era5_basin_coverage.csv` | Monthly basin-average ERA5-Land variables and their land-cell coverage |
| `data/processed/indices.csv` | Monthly climate indices |
| `data/processed/li2026_csr_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The CSR member of GRACE-FCast, aggregated by us to the basins above, and per-basin coverage |
| `data/processed/jpl/basin_month_twsa_global.csv`, `basin_meta.csv` | Monthly basin-average TWSA (cm) and basin metadata with the JPL exclusion flags, for the JPL RL06.3Mv04 CRI product |
| `data/processed/jpl/li2026_jpl_basin_forecasts.csv`, `li2026_basin_coverage.csv` | The JPL member of GRACE-FCast, aggregated by us to the basins above, and per-basin coverage |

`graceseespaper-71be5ec-jpl-run-code.zip` unpacks into `graceseespaper-71be5ec/`. It holds
`src/`, `scripts/`, `tests/` and `requirements-lock.txt` at commit `71be5ec` of the JPL
pipeline branch, the code that produced the JPL tables and row-level files in
`results/jpl/` (run record in `results/jpl/RUN_PROVENANCE.md`). The CSR results and the
figures come from the main code archive. The JPL steps in the main archive have changed
since this revision: it adds the `flat12_ridge` and `kalman_mission` steps, which the JPL
run did not have. Regenerate the archived JPL tables with this revision.

Appendix A (Table A1) of the manuscript maps model names to the `model` identifiers in
these files. Every JPL table and figure value is in `results/jpl/*.csv` in the code
archive.

`data/processed/jpl/li2026_basin_coverage.csv` is the file the JPL runs used. It predates
the `n_full_native_mascons` column that the current comparison code reads. To rerun the
JPL comparison with this file, add that column from `n_full_jpl_mascons` in
`data/processed/jpl/basin_meta.csv`, matched on `name`. This is what
`scripts/build_li_basin_series.py --source jpl` now writes.

`data/processed/indices.csv` was downloaded again from NOAA PSL on 16 September 2026,
after the runs that used it. NOAA revises some indices, so its values may differ slightly
from the run inputs. The climate indices enter one benchmark model only.

## Reproducing the figures and tables

Install the pinned environment with Python 3.11 and check the large files against their
manifests:

```
pip install -r requirements-lock.txt
python scripts/make_manifest.py --check
GRACEFC_SOURCE=jpl python scripts/make_manifest.py --check
python scripts/make_figures.py
```

The figure script checks each plotted headline value against its source table. It needs
the Arial font, and Cartopy downloads the Natural Earth base map on first use. Map figures
also require the basin mask described below. Rerunning
the experiments from raw data follows `README.md` and needs the third-party inputs below,
including a compatible basin mask.

## Basin mask

The basin mask used in this study (`HydroShed+Mascon_Basins_L3.nc`) combines
HydroSHEDS basins with the CSR mascon land/ocean mask. It is not available for
public redistribution and is not included in this deposit.

The processed basin series and result tables are included. Reproducing the
original basin aggregation and map figures also requires the mask, so those
steps cannot be reproduced from this deposit alone.

## Third-party data

The processed inputs are derived from CSR RL06 mascons (Save et al., 2016; Save, 2020),
JPL RL06.3Mv04 mascons (Wiese et al., 2023), ERA5-Land (Muñoz-Sabater et al., 2021), NOAA
PSL climate indices, the GRACE-FCast hindcast of Li and Kusche (PANGAEA, CC BY 4.0) and
HydroSHEDS (Lehner and Grill, 2013). The files derived from ERA5-Land carry the
attribution Copernicus requires:
"Contains modified Copernicus Climate Change Service information 2026."
Cite the original products when reusing these files.

## Licence

Code: MIT (see `LICENSE`), except the Copernicus LaTeX class, style and template files
under `paper/`, which keep Copernicus's terms. The authors' original data and result
files are licensed under CC BY 4.0. Third-party materials retain their original
licences and attribution requirements. The basin mask is not included and is not
covered by these licences.
