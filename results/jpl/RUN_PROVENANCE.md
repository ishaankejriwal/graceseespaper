# JPL run record

This file records how the JPL results in this directory were produced. Git versions the
summary and diagnostic tables. It ignores the row-level predictions, placebo draws, model
state and chain logs, which can be regenerated and make up nearly all of the output size.
The predictions, placebo draws and model state are in the data archive of the Zenodo
deposit. `SHA256_MANIFEST_LIVE.csv` identifies the generated files behind the shared
tables.

## Run

- Pipeline: the JPL mascon experiment chain (`--source jpl`), kept separate from the CSR
  outputs.
- Code revision: `71be5ec6af30e9d8f3c72549ca114c3b47071380` ("compare JPL results with
  Li"). The strict comparison was run from the work that became this commit.
- Main suite: 2026-09-05 15:36 to 2026-09-06 01:21, America/Chicago workstation time.
- JPL basin table and strict Li comparison rerun: 2026-09-06 12:30 to 12:34,
  America/Chicago workstation time.
- Evaluation: five frozen-origin folds beginning 2019-06, 2020-11, 2022-04, 2023-09 and
  2025-02. The matched Li comparison has 59 issue months and leads 1 to 6.

## GRACE/GRACE-FO target

- File: `data/GRCTellus.JPL.200204_202606.GLO.RL06.3M.MSCNv04CRI.nc`
- Product: JPL GRACE and GRACE-FO Mascon RL06.3Mv04 CRI (source `GRACE and GRACE-FO JPL
  RL06.3Mv04-CRI`, product version `v3.0`), DOI `10.5067/TEMSC-3JC634`.
- File created 2026-08-19 10:08:27 UTC; time coverage 2002-04-16 to 2026-06-16.
- Grid: sampled at 0.5 degrees. The native resolution is about 3-degree equal-area
  mascons.
- SHA-256: `d5a6737bd0406ef4e9ea72c825ea95c6693a15faf2027e52be491efa8cf89bbd`

JPL has already applied the Coastline Resolution Improvement filter in the CRI product.
The pipeline applied the NetCDF `scale_factor` field (the run did not use
`--no-scale-factors`), and every row of `data/processed/jpl/basin_meta.csv` records
`scale_factors_applied=True`. The target reader uses JPL's official missing-month
metadata.

## Basins

The basin mask is `HydroShed+Mascon_Basins_L3.nc`, SHA-256
`b0434196b7615366626b4bd88ce1f450c63c79e1ebe0b866fea3a05aebaf4a24`. The JPL table has 284
mask basins: 228 kept for modelling, 46 ice sheets, 4 water bodies and 6 without valid
JPL observations.

## Processed inputs

SHA-256 of the processed JPL inputs in `data/processed/jpl/`, recorded on 2026-10-14 from
the files used for these runs:

| File | SHA-256 |
|---|---|
| `basin_meta.csv` | `e053deba3c3d928780d649b82709b0cbaf21672ebe80729c838fa92d892fc873` |
| `basin_month_twsa_global.csv` | `e243dcfd114beca90892f11771881947dfa923d9ccf7bf7bd3398cfde5e881af` |
| `li2026_basin_coverage.csv` | `c8eea0b3eb82ef5633a721acd2b5c1eda02b5b57e01ccc5f599519f98c753e04` |
| `li2026_jpl_basin_forecasts.csv` | `36645fea53b15f84ae36be93d74b92f0e11bf5a826bb2633e9d9105d97ea1e41` |

Rebuilt from these files, the JPL baseline ladder and the matched Li comparison
populations reproduce `phase2_baseline_summary.csv` and `phase6_li_comparison_summary.csv`
to rounding error.

## GRACE-FCast comparison

The forecasts are Li and Kusche's JPL-FCast, the JPL member of GRACE-FCast, PANGAEA dataset
`10.1594/PANGAEA.973113`: 173 monthly NetCDF initialization files in
`data/raw/li2026/JPL-FCast/global_gridded/`, from 2009-12 to 2024-04, on 1-degree global
land cells. Li and Kusche trained it against JPL RL06.1_v03 (April 2002 to April 2024),
while our target is JPL RL06.3Mv04 CRI, so the comparison crosses product releases.

`li_lstm_nonseas` is their interannual plus subseasonal forecast, transformed into the
pipeline's fold-specific standardized, deseasonalized target space. `li_lstm_full` adds
their published seasonal and linear terms before the same transformation.

Each comparison table reports two populations, and within each population every model is
scored on the same `(basin, issue_date, target_date, horizon)` rows.

- `all_matched`: every basin and date row available to all compared models; 227 basins,
  13,393 rows per horizon and 59 issue months.
- `joint_full_cells`: a basin must contain every 0.25-degree mask subcell of at least one
  native JPL `mascon_ID`, and all sixteen 0.25-degree subcells of at least one finite
  1-degree GRACE-FCast cell. This is full containment, not an area threshold or a
  coverage fraction. It keeps 67 basins, 3,953 rows per horizon and 59 issue months.

The strict population is the main resolution-controlled sensitivity test for the
JPL-versus-GRACE-FCast results. The headline and summary files keep both populations so
the effect of the restriction is visible.

## Seeds and resampling

- Moving-block bootstrap confidence intervals: 2,000 resamples, seed 0. The block length
  comes from the autocorrelation of the loss series unless supplied.
- LSTM models: seeds 0 and 1.
- Stacked LSTM and residual models: seeds 0 and 1.
- Residual MLP experiments: seeds 0, 1 and 2.
- Random-graph placebo experiments: 20 draws per fold and horizon by default, with seeds
  built deterministically from the fold and horizon in the experiment modules.
- IAAFT surrogate control: draws 0 to 98 (99 in total).
- Basin random-forest analysis and permutation importance: seed 0.

## Checks and conventions

`SHA256_MANIFEST_LIVE.csv` lists the byte count and SHA-256 digest of every generated CSV
and serialized model-state file in this directory.
`GRACEFC_SOURCE=jpl python scripts/make_manifest.py --check` verifies the files git does
not store; git versions the rest.

Positive `skill` means the first named model has lower MSE than its `vs` model:
`skill = 1 - MSE(model) / MSE(vs)`. Check the `model` and `vs` columns before reading a
value as a percentage improvement.

Raw NetCDF inputs and processed row-level data are not stored in git. The product
identifiers and input checksums above let another researcher obtain and verify the same
sources.
