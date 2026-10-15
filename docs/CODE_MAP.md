# Code map

`src/gracefc/` is the library: data aggregation, the evaluation folds, the Kalman filter and
the forecast models. Each script in `scripts/` runs one step and writes its outputs to
`results/`, or to `results/jpl/` when `GRACEFC_SOURCE=jpl`. `scripts/run_chain.py` runs the
steps in dependency order. Inputs, outputs and options for each file are in its docstring.
[ARCHIVE_MANIFEST.md](ARCHIVE_MANIFEST.md) lists the result file behind each table and
figure.

## Library (`src/gracefc/`)

Data, evaluation and the filter:

| Module | Contents |
|---|---|
| `basins.py` | Aggregates a CSR or JPL mascon file to monthly basin-mean TWSA in cm and builds the basin metadata table, with the reason for each excluded basin |
| `era5.py` | Aggregates ERA5-Land monthly fields to basin means and builds the per-fold anomaly features |
| `decompose.py` | Linear trend plus annual and semiannual harmonics, fitted on the training window only |
| `features.py` | Own-basin lag features indexed by issue date, and the short-gap fill behind them |
| `evaluate.py` | Evaluation folds, the issue-date train/test split, the benchmark forecasts and scoring |
| `models.py` | Pooled and per-basin lag ridges for the benchmarks, and RMSE |
| `kalman.py` | Scalar Kalman filter for the AR(1) state with model noise (KF): maximum-likelihood fit of (rho, q, r) per basin and fold, filtering and forecasts |
| `kalman_mission.py` | Two-variance variant with separate model-noise variances for GRACE and GRACE-FO (Fig. 4b) |
| `cache.py` | Cache of per-fold KF parameters, keyed by a fingerprint of the basin tables, `kalman.py`, the numpy and scipy versions and the fold specification |
| `stats.py` | DM test with the Harvey et al. (1997) correction, moving-block bootstrap intervals for skill, and per-basin DM tests with Benjamini-Hochberg control |
| `comparison.py` | Coverage rules that define the fully covered basins for the GRACE-FCast comparison |
| `runtime.py` | Data and result paths for the product selected by `GRACEFC_SOURCE` (`csr`, the default, or `jpl`) |

Correction models. Each one predicts the KF error (target minus KF forecast), and the
forecast is KF plus the correction.

| Module | Contents |
|---|---|
| `experiment_flat12.py` | KF, KF-R1, KF-R12 and KF-R12E, and the 12-month window features shared with the LSTM. No torch (Tables 2 and 3, Fig. 5) |
| `phase7.py` | Shared setup for the correction experiments: folds, design frames and the summary writer |
| `experiment_lstm.py` | Shared-encoder LSTM over 12-month filtered-state windows, and the MLP on the same window. Needs torch (Table 4, Fig. 6) |
| `experiment_resmlp.py` | Ridge on the own filtered state, then an MLP on the correction features (residual MLP rows of Table 4) |
| `experiment_nonlinear.py` | GBM and MLP heads on own and neighbor states, with a 2-hop neighbor arm; its `_fit_head` is used by the other modules |
| `experiment_kalman.py` | Neighbor ridge corrections with random-graph placebos (Sect. 4.6) |
| `graphs.py` | Neighbor graphs by correlation, lagged correlation, distance or correlation beyond a minimum distance, and degree-matched random placebo graphs |
| `surrogates.py` | IAAFT surrogates for the surrogate test in Sect. 4.6 |
| `experiment_era5.py` | Ridge, GBM and MLP corrections with ERA5-Land and neighbor features (not in the paper) |
| `fusion.py` | KF with the neighbor's series as a second observation of the state (not in the paper) |
| `coupled.py` | Bivariate KF for a basin and its top-1 neighbor with correlated process noise (not in the paper) |
| `experiment_gnn.py` | One-layer graph attention network. Needs torch (not in the paper) |
| `experiment_lstm_combined.py` | LSTM correction followed by a neighbor MLP, the stacked model (not in the paper) |

## Scripts (`scripts/`)

Input data. Downloads need network access and are not chain steps.

| Script | Purpose |
|---|---|
| `download_era5.py` | ERA5-Land monthly means at 0.5 degrees from the Copernicus Climate Data Store; needs a CDS API key |
| `download_indices.py` | Monthly climate indices from NOAA PSL, CPC and NCEI, combined into `data/processed/indices.csv` |
| `build_basin_series.py` | Basin TWSA table and `basin_meta.csv` (Sects. 2.1 and 2.2); options `--source jpl`, `--mascon-file`, `--no-scale-factors` |
| `build_era5_basin_table.py` | Basin means of the 11 ERA5-Land variables (Sect. 2.3) |
| `build_li_basin_series.py` | GRACE-FCast basin means and the coverage table `li2026_basin_coverage.csv` (Sect. 3.6, Fig. 1) |
| `inspect_inputs.py` | Prints the structure of the CSR mascon file and the basin mask; writes nothing |

Default chain steps, in run order. None needs torch.

| Script | Chain step | Manuscript items |
|---|---|---|
| `run_phase2_baselines.py` | `phase2` | Benchmark forecasts of Sect. 3.3 |
| `run_kalman_baseline.py` | `kalman` | KF forecasts |
| `run_flat12_ridge.py` | `flat12_ridge` | KF-R1, KF-R12 and KF-R12E; Table 3, Fig. 5 |
| `run_kalman_mission_sensitivity.py` | `kalman_mission` | Fig. 4b, Sect. 4.2, boundary fits in Sect. 3.4 |
| `run_r0_ablation.py` | `r0_ablation` | Zero-r variant; Fig. 4a, Sect. 4.2 |
| `build_paper_ladder.py` | `ladder` | Tables 2, 3 and B1, Fig. 2a; Table B2 with `GRACEFC_SOURCE=jpl` |
| `run_phase6_li_comparison.py` | `li_comparison` | Table 5, Fig. 7, Sects. 3.6 and 4.4 |
| `compute_conventional_metrics.py` | `conventional_metrics` | Figs. 2b and 4c, panel RMSEs of Fig. 3, Sects. 4.1 and 4.5, Table C1 |
| `make_manifest.py` | `manifest` | SHA-256 checksums of the result files git excludes; `--check` verifies them |

Run these three after the default chain. They are not chain steps.

| Script | Manuscript items |
|---|---|
| `climatology_offset.py` | Climatology drift in Sect. 3.1 (CSR only) |
| `kalman_gain_correlates.py` | Per-basin gain correlates in Sect. 4.1 (CSR only) |
| `centimeter_contrasts.py` | DM tests and bootstrap intervals in cm, Sect. 4.5 |

Extended chain steps (`run_chain.py --extended`, or by name with `--steps`):

| Script | Chain step | Manuscript items |
|---|---|---|
| `run_phase7.py --arch lstm` | `phase7_lstm` | Table 4, Fig. 6a. Needs torch |
| `run_phase7.py --arch resmlp` | `phase7_resmlp` | Residual MLP rows of Table 4 |
| `run_flat12_train85_sensitivity.py` | `flat12_train85` | KF-R12E refit on the LSTM's training rows; Fig. 6b, Sect. 4.3 |
| `run_phase3b_kalman_neighbors.py` | `phase3b`, `predlag`, `conditioned` | Neighbor corrections against random-graph placebos; Sect. 4.6 (`phase3b`) |
| `run_phase4_surrogates.py` | `surrogates` | Neighbor against IAAFT surrogates; Sect. 4.6 |
| `make_figures.py` | `figures` | Figs. 1 to 7 as PDF and PNG in `figures/`, with every headline value asserted against its source file. CSR only |
| `run_jump_screen.py` | `jump_screen` | Not in the paper |
| `run_phase5_infilter.py --model fusion\|coupled` | `phase5_fusion`, `phase5_coupled` | Not in the paper |
| `run_phase5_nonlinear.py` | `phase5_nonlinear` | Not in the paper |
| `run_phase5_stats.py` | `phase5_stats` | Not in the paper |
| `run_phase6_era5.py` | `phase6_era5` | Not in the paper |
| `run_phase6_era5_attribution.py` | `era5_attribution` | Not in the paper |
| `run_phase6_hybrid.py` | `hybrid` | Not in the paper |
| `run_phase6_basin_analysis.py` | `basin_analysis` | Not in the paper |
| `run_resolution_sensitivity.py` | `resolution` | Not in the paper; `build_li_basin_series.py` reuses its native-mascon geometry |
| `run_phase7.py --arch gnn` | `phase7_gnn` | Not in the paper. Needs torch |
| `run_phase8_lstm_combined.py` | `phase8_h13`, `phase8_h46` | Not in the paper; adds the `stacked_ens` rows to the conventional metrics when present. Needs torch |
| `run_phase8b_merge.py` | `phase8b_merge` | Not in the paper |
| `run_phase8_stratification.py` | `phase8_strat` | Not in the paper |

Other scripts:

| Script | Purpose |
|---|---|
| `run_chain.py` | Runs the chain; options `--extended`, `--steps`, `--list`, `--source jpl`, `--mascon-file`, `--no-scale-factors` |
| `audit_paper_statistics.py` | Recomputes the Table B1 statistics from the prediction files and checks them against `paper/main.tex`; writes nothing |
| `build_deposit.py` | Stages the Zenodo code and data archives in `deposit/`; uploads nothing |
