# A Kalman filter benchmark for GRACE water storage forecasts

Code and result tables for Kejriwal and Wang, "A Kalman filter benchmark for data-driven
forecasts of GRACE terrestrial water storage", submitted to Hydrology and Earth System Sciences
(HESS).

The study forecasts deseasonalized GRACE and GRACE-FO terrestrial water storage anomalies in
234 river basins at lead times of 1 to 6 months. A per-basin Kalman filter (KF) with an AR(1)
state and model noise with variance r is proposed as the reference forecast in place of damped
persistence. Ridge corrections on top of KF (KF-R1, KF-R12, KF-R12E), nonlinear models and Li
and Kusche's GRACE-FCast are scored against it on the CSR RL0603M mascons, with the JPL
RL06.3Mv04 CRI mascons as a second product.

## Requirements

Python 3.11 (the lock file was built with 3.11.4). Every dependency is pinned in
`requirements-lock.txt`, including torch, which only the extended steps use. Nothing needs a
GPU.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt
.venv/Scripts/python -m pytest tests/ -q
```

The tests take about 30 s and report 28 passed. Tests whose input files are absent are skipped.

`scripts/run_chain.py` starts each step with `.venv/Scripts/python.exe`, so the environment must
be a virtual environment named `.venv` in the repository root, in the Windows layout.
`scripts/make_figures.py` needs the Arial font, and Cartopy downloads the Natural Earth base map
the first time it runs.

## Input data

Raw inputs are not in git. The processed basin tables and row-level result files are
listed for the Zenodo data archive (`docs/DEPOSIT_README.md`). The supplied basin mask
cannot be redistributed publicly. A compatible mask is needed to rebuild the map figures
or rerun the pipeline from raw data. Release of the basin names in the public tables
also remains under review; see the deposit README.

| Input | Path | Source |
|---|---|---|
| CSR RL0603M mascons | `CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc` (repository root) | [CSR mascon page](http://www2.csr.utexas.edu/grace/RL0603_mascons.html) |
| CSR mapping file and land mask | `data/raw/csr_ancillary/CSR_GRACE_GRACE-FO_RL0603_mascons_mapping_file.nc`, `data/raw/csr_ancillary/CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc` | CSR mascon page, ancillary files |
| JPL RL06.3Mv04 CRI mascons (JPL runs only) | passed with `--mascon-file` | PO.DAAC, doi:10.5067/TEMSC-3JC634 |
| Basin mask | `HydroShed+Mascon_Basins_L3.nc` (repository root, local use only) | Supplied study input; excluded from the public archive. A public replacement must be independently generated from HydroSHEDS basins intersected with the CSR land/ocean mask and validated |
| ERA5-Land monthly means | `data/raw/era5/` | `python scripts/download_era5.py` |
| Climate indices | `data/processed/indices.csv` | `python scripts/download_indices.py` |
| GRACE-FCast hindcast (Li and Kusche) | `data/raw/li2026/CSR-FCast/global_gridded/`, `data/raw/li2026/JPL-FCast/global_gridded/` | [PANGAEA, doi:10.1594/PANGAEA.973113](https://doi.pangaea.de/10.1594/PANGAEA.973113) |

`download_era5.py` requests 11 ERA5-Land variables at 0.5° for 2001 through May 2026 from the
Copernicus Climate Data Store, one file per year (`era5_land_monthly_<year>.nc`). It needs a CDS
account with an API key configured for `cdsapi`, and skips years already on disk, so a rerun
resumes an interrupted download. `download_indices.py` needs no account; it fetches the indices
from NOAA PSL, CPC and NCEI. `scripts/inspect_inputs.py` prints the structure of the NetCDF
inputs without changing them.

## Reproducing the figures and tables from the archive

Unpack the code and data archives into the same folder, check the large result files against
their checksums, and, if a compatible basin mask is available locally, build the figures:

```bash
python scripts/make_manifest.py --check
GRACEFC_SOURCE=jpl python scripts/make_manifest.py --check
python scripts/make_figures.py
```

In PowerShell, set the variable with `$env:GRACEFC_SOURCE = "jpl"` before the second command.
`make_manifest.py --check` compares the result files git does not store with
`results/SHA256_MANIFEST_LIVE.csv`, or `results/jpl/SHA256_MANIFEST_LIVE.csv` when
`GRACEFC_SOURCE=jpl`. `make_figures.py` writes the seven manuscript figures to `figures/` as PDF
and 300 dpi PNG. It checks the headline values it reads against the numbers printed in the
paper, to the printed precision, and stops instead of drawing a figure when one differs.

`docs/ARCHIVE_MANIFEST.md` lists the result file and column behind each table, figure and quoted
number in the paper. Appendix A (Table A1) of the paper maps model names to the identifiers in
the files: KF is `kalman_ar1`, KF-R1 `kalman_own_ridge`, KF-R12 `ridge_own_flat12` and KF-R12E
`ridge_own_era5_flat12`. `docs/CODE_MAP.md` describes each module and script, and
`results/README.md` the result files.

Skill is 1 - MSE(model)/MSE(vs), so a positive value means the first-named model has the lower
error. Read the `model` and `vs` columns before quoting a value: the JPL GRACE-FCast comparison
lists GRACE-FCast first. Errors are in standardized units (each basin's deseasonalized residual
divided by its standard deviation over the fold's training months) unless the column name says
cm. In the GRACE-FCast
comparisons, `joint_full_cells` is the set of fully covered basins: a basin must contain every
0.25° subcell of at least one native mascon and all 16 subcells of at least one finite 1°
GRACE-FCast cell. `all_matched` keeps every basin and month available to all compared models.

## Rerunning the experiments

`scripts/run_chain.py` runs the pipeline steps in dependency order. Before a step it checks that
the declared inputs exist, afterwards that every declared output was rewritten, and it stops at
the first failure. Each step logs to `results/chain_<step>.log`.

```bash
python scripts/run_chain.py --list                # steps with their inputs and outputs
python scripts/run_chain.py                       # default list
python scripts/run_chain.py --extended            # extended list only
python scripts/run_chain.py --steps phase7_lstm flat12_train85 figures
```

The default list has 12 steps and does not need torch:

```
build_basin build_era5 build_li phase2 kalman flat12_ridge kalman_mission
r0_ablation ladder li_comparison conventional_metrics manifest
```

It writes the result files behind Tables 2, 3 and 5, Appendices B and C, and Fig. 4. Three
scripts that are not chain steps feed Sects. 3.1, 4.1 and 4.5; run them after the default list:
`scripts/climatology_offset.py`, `scripts/kalman_gain_correlates.py` and
`scripts/centimeter_contrasts.py`.

Every other step is on the extended list: the neighbour experiments of Sect. 4.6, the nonlinear
models of Table 4 and Fig. 6, the KF-R12E refit on the LSTM's training months
(`flat12_train85`), several experiments not reported in the paper, and `figures`. `--extended`
runs only that list, so run the default list first. `--steps` runs the named steps in the order
given and cannot be combined with `--extended`. The `figures` step reads outputs of
`phase7_lstm` and `flat12_train85`, which the data archive already contains.

On the CSR rerun of 10 September 2026 the timed default steps took 5.9 min (`build_li`), 2.0 min
(`flat12_ridge`), 29 min (`kalman_mission`), 1.0 min (`ladder`), 6.0 min (`li_comparison`),
0.7 min (`conventional_metrics`) and 0.2 min (`manifest`); `r0_ablation` took 4.4 min in
August 2026. The neural extended steps take hours each. In the August 2026 run `phase7_lstm`
took 9.0 h, `phase7_resmlp` 8.0 h, `phase8_h13` 5.5 h, `phase8_h46` 5.0 h, `phase6_era5` 3.0 h
and `phase5_nonlinear` 2.2 h.

## JPL mascons

```bash
python scripts/run_chain.py --source jpl --mascon-file data/raw/GRCTellus.JPL.200204_202606.GLO.RL06.3M.MSCNv04CRI.nc
```

Without `--mascon-file` the chain looks for
`data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`. JPL processed tables go to
`data/processed/jpl/` and results to `results/jpl/`; the ERA5-Land and index tables are shared
with CSR, and the CSR results are not overwritten. If the file has a `scale_factor` field, as the
CRI product does, it is applied and recorded in `basin_meta.csv`; `--no-scale-factors` turns
this off. Six small-island basins have no valid cells in the CRI product, which leaves 228
basins. The GRACE-FCast JPL member must be under `data/raw/li2026/JPL-FCast/global_gridded/`. The
`figures` step is not available for JPL because it asserts the CSR values in the paper.

The JPL tables in `results/jpl/` were produced at commit `71be5ec` of the JPL pipeline branch,
archived separately as `graceseespaper-71be5ec-jpl-run-code.zip`. The JPL steps have changed
since that commit (the archived JPL tables have no flat-12 models), so regenerate those tables
with the archived revision. `results/jpl/RUN_PROVENANCE.md` records the product file and its
checksum, the run dates, seeds and comparison populations, and `docs/DEPOSIT_README.md`
describes the three archives.

## Repository layout

| Path | Contents |
|---|---|
| `src/gracefc/` | Python package: basin aggregation, deseasonalization, KF, models, statistics |
| `scripts/` | Downloads, pipeline steps, figures, checksums |
| `tests/` | Regression and data tests (pytest) |
| `results/`, `results/jpl/` | Result tables for CSR and JPL |
| `figures/` | Manuscript figures |
| `paper/` | Manuscript source (`main.tex`, `references.bib`) and the Copernicus LaTeX class |
| `docs/` | `CODE_MAP.md`, `ARCHIVE_MANIFEST.md`, `DEPOSIT_README.md` |
| `notebooks/` | Two exploratory notebooks, saved without outputs |

## Licence

Code is released under the MIT licence (`LICENSE`), except the Copernicus LaTeX class, style and
template files under `paper/`, which keep Copernicus's terms. Data and result files are released
under CC BY 4.0. Files derived from third-party products (CSR and JPL mascons, ERA5-Land, NOAA
indices, GRACE-FCast, HydroSHEDS) remain subject to those products' attribution terms.

## How to cite

Kejriwal, I. and Wang, I.: A Kalman filter benchmark for data-driven forecasts of GRACE
terrestrial water storage, submitted to Hydrology and Earth System Sciences, 2026.

Kejriwal, I. and Wang, I.: Data and code for "A Kalman filter benchmark for data-driven forecasts
of GRACE terrestrial water storage", Zenodo, 2026.

DOI: to be added
