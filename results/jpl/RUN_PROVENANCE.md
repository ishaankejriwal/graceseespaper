# JPL experiment result provenance

This record describes the JPL results in this directory. Compact result and
diagnostic tables are versioned; row-level predictions, placebo draws, model
state, and transient chain logs remain ignored because they are regenerable and
account for nearly all of the output size. `SHA256_MANIFEST_LIVE.csv` identifies
the exact local generated artifacts used to produce the shared tables.

## Run identity

- Pipeline: source-isolated JPL mascon experiment chain (`--source jpl`).
- Experimental code revision: `71be5ec6af30e9d8f3c72549ca114c3b47071380`
  (`compare JPL results with Li`). The strict comparison was run from the work
  that became this commit.
- Main suite run window: 2026-09-05 15:36 through 2026-09-06 01:21,
  America/Chicago workstation time.
- JPL basin table and strict Li comparison rerun: 2026-09-06 12:30 through
  12:34, America/Chicago workstation time.
- Forecast evaluation folds: five frozen-origin folds beginning 2019-06,
  2020-11, 2022-04, 2023-09, and 2025-02. The matched Li comparison contains
  59 issue months and evaluates leads 1 through 6.

## GRACE/GRACE-FO target

- File: `data/GRCTellus.JPL.200204_202606.GLO.RL06.3M.MSCNv04CRI.nc`
- Product title: JPL GRACE and GRACE-FO Mascon RL06.3Mv04 CRI.
- Product source/version: `GRACE and GRACE-FO JPL RL06.3Mv04-CRI`, product
  version `v3.0`.
- Dataset DOI: `10.5067/TEMSC-3JC634`.
- File creation time: 2026-08-19 10:08:27 UTC.
- Time coverage: 2002-04-16 through 2026-06-16.
- Grid: sampled at 0.5 degrees; native product resolution is approximately
  3-degree equal-area mascons.
- SHA-256:
  `d5a6737bd0406ef4e9ea72c825ea95c6693a15faf2027e52be491efa8cf89bbd`.

### Corrections and scale factors

- The input is the CRI product: its Coastline Resolution Improvement filter is
  already applied by JPL.
- The pipeline applied the NetCDF `scale_factor` field. This is recorded as
  `scale_factors_applied=True` for every row in
  `data/processed/jpl/basin_meta.csv`; the run did not use
  `--no-scale-factors`.
- The target reader also uses JPL's official missing-month metadata.

## Basin geometry

- Basin mask: `HydroShed+Mascon_Basins_L3.nc`.
- Basin-mask SHA-256:
  `b0434196b7615366626b4bd88ce1f450c63c79e1ebe0b866fea3a05aebaF4a24`
  (hexadecimal is case-insensitive).
- The JPL table contains 284 mask basins: 228 retained for modeling, 46 ice
  sheets, 4 water bodies, and 6 without valid JPL observations.

## Li and Kusche comparison source

- Forecast product: Li and Kusche JPL-FCast from PANGAEA dataset
  `10.1594/PANGAEA.973113`.
- Local source: 173 monthly NetCDF initialization files under
  `data/raw/li2026/JPL-FCast/global_gridded/`, from 2009-12 through 2024-04.
- Resolution: 1-degree global land forecast cells.
- Li's JPL-FCast was trained against JPL RL06.1_v03 (April 2002 through April
  2024), whereas this pipeline's verification target is JPL RL06.3Mv04 CRI.
  Results must therefore be described as a cross-release comparison, not a
  perfectly release-matched model comparison.
- `li_lstm_nonseas` is Li's interannual plus subseasonal forecast transformed
  into the pipeline's fold-specific standardized deseasonalized target space.
  `li_lstm_full` adds Li's published seasonal and linear terms before the same
  comparison transformation.

## Spatial comparison rules

The comparison tables report both populations and match every model on the
same `(basin, issue_date, target_date, horizon)` rows within each population.

- `all_matched`: every basin/date row jointly available to all compared models;
  227 basins, 13,393 rows per horizon, and 59 issue months in the current run.
- `joint_full_cells`: a basin must contain every 0.25-degree mask subcell of at
  least one native JPL `mascon_ID` and all sixteen 0.25-degree subcells of at
  least one finite 1-degree Li forecast cell. This literal full-containment rule
  retains 67 basins, 3,953 rows per horizon, and 59 issue months. It is not an
  area-threshold or fractional-coverage proxy.

The strict population is the primary resolution-controlled sensitivity for the
JPL-versus-Li claims. Both populations remain in the headline and summary files
so readers can see how the spatial restriction changes the result.

## Randomness and uncertainty settings

- Moving-block bootstrap confidence intervals: 2,000 resamples, seed 0; block
  length is chosen from the loss-series autocorrelation unless explicitly
  supplied.
- LSTM models: seeds 0 and 1.
- Stacked LSTM/residual models: seeds 0 and 1.
- Residual MLP experiments: seeds 0, 1, and 2.
- Random-graph placebo experiments: 20 draws per fold/horizon by default, with
  deterministic fold/horizon-specific seed construction in the experiment
  modules.
- IAAFT surrogate control: draws 0 through 98 (99 total).
- Basin random-forest analysis and permutation importance: seed 0.

## Integrity and interpretation

- `SHA256_MANIFEST_LIVE.csv` contains byte counts and SHA-256 digests for every
  generated CSV and serialized model-state file in this JPL result directory.
- Positive `skill` means the table's first named model has lower MSE than its
  `vs` model: `skill = 1 - MSE(model) / MSE(vs)`. Always check the `model` and
  `vs` columns before translating a value into percent improvement.
- Raw NetCDF inputs and processed row-level data are intentionally not stored in
  Git. Their product identifiers and critical input checksums are recorded here
  so another researcher can obtain and verify the same sources.
