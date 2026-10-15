# Archive manifest

This file lists the result file and columns behind each table, figure and in-text number in
`paper/main.tex`. Tables and figures are numbered as in the manuscript. Table A1 of the
paper maps each model name (KF, KF-R1, KF-R12, KF-R12E and so on) to the identifier used in
the result files.

## Conventions

Skill is 1 - MSE(first)/MSE(second), so a positive value means the first-named model (the
`model` or `challenger` column) has the lower error. Columns named `skill` or `skill_vs_*`
hold fractions; columns ending in `_pct` hold percentages. The paper prints percentages and
states every skill with our model first. `results/phase6_li_comparison_headline.csv` holds
each pair in both orders. `results/jpl/phase6_li_comparison_headline.csv` stores GRACE-FCast
first, so a stored skill s becomes 1 - 1/(1 - s) with our model first; the DM p value is
unchanged.

`rmse_std` and every skill computed from it are in standardized units: each basin and fold
is divided by its training-window residual standard deviation before pooling. `rmse_cm`,
`pooled_rmse_cm` and the other `_cm` columns are in cm of equivalent water height.
`horizon` is the lead time in months.

The reference in Table 2, Table B1, Table B2 and Fig. 2a is the stronger damped-persistence
variant at each lead time, named in the `damped_ref` column of `paper_baseline_ladder.csv`:

| Product | ρ^h variant (`damped_persistence_rho`) | Regression variant (`damped_persistence_reg`) |
|---|---|---|
| CSR | h = 1 | h = 2 to 6 |
| JPL | h = 1 and 2 | h = 3 to 6 |

Table 5 and Sect. 4.4 use the ρ^h variant at every lead time, and Fig. 4a uses the
regression variant at every lead time. The `damped_persistence` rows of the
conventional-metric files follow the CSR mapping on both products.

Files ending in `_predictions.csv` and everything under `data/processed/` are in the data
archive because git excludes them for size; `results/SHA256_MANIFEST_LIVE.csv` and its
copy in `results/jpl/` hold the checksums of the result files among them. All other files
listed here are in the code archive. The tables in
`results/jpl/` were produced at commit 71be5ec (`results/jpl/RUN_PROVENANCE.md`).

## Tables

| Item | Source file and columns |
|---|---|
| Table 1 (forecast names) | None; the forecasts are defined in Sect. 3 |
| Table 2 (benchmark skill, CSR) | `results/paper_baseline_ladder.csv`: `skill_vs_damped`, reference in `damped_ref` |
| Table 3 (own-basin models against KF) | KF-R1: `results/flat12_ridge_summary.csv`, `kalman_own_ridge` rows, `skill_vs_kalman` and `dm_p_vs_kalman`. KF-R12 and KF-R12E: `results/paper_baseline_contrasts.csv`, challengers `ridge_own_flat12` and `ridge_own_era5_flat12` against `kalman_ar1`, `skill` and `dm_p` |
| Table 4 (nonlinear models, h = 1 to 3) | `rmse_std` in `results/phase7_lstm_summary.csv` (`kalman_ar1`, `ridge_own_era5`, `ridge_own_flat12`, `ridge_own_era5_flat12`, `lstm_own_era5_s0`/`_s1`, `mlp_own_era5_flat12_s0`/`_s1`) and `results/phase7_resmlp_summary.csv` (`resmlp_own_era5_s0`/`_s1`/`_s2`) |
| Table 5, CSR rows | `results/phase6_li_comparison_headline.csv`, subset `joint_full_cells`: `kalman_ar1`, `ridge_own_era5_flat12` and `damped_persistence_rho` against `li_lstm_full` and `li_lstm_nonseas`, `skill` and `dm_p`. Basin counts: `results/phase6_li_comparison_perbasin.csv`, `li_lstm_full` against `kalman_ar1` and `ridge_own_era5_flat12`, number of basins with `a_better` true (GRACE-FCast better) at h = 1 and 6 |
| Table 5, JPL rows | `results/jpl/phase6_li_comparison_headline.csv`, subset `joint_full_cells`: `li_lstm_full` and `li_lstm_nonseas` against `kalman_ar1` and `damped_persistence_rho`, `skill` converted as above and `dm_p` |
| Table A1 (identifiers) | None |
| Table B1 (CSR, full) | `results/paper_baseline_ladder.csv`: `rmse_std`, `dm_p_vs_damped` |
| Table B2 (JPL benchmarks) | `results/jpl/paper_baseline_ladder.csv`: `skill_vs_damped`, `dm_p_vs_damped`, reference in `damped_ref` |
| Table C1 (conventional metrics) | `results/conventional_metrics_summary.csv`: `pooled_rmse_cm`, `rmse_cm_med`, `cc_anom_med`, `cc_full_med`, `nse_anom_med`, `nse_full_med` for `damped_persistence`, `kalman_ar1`, `kalman_own_ridge`, `ridge_own_flat12`, `ridge_own_era5_flat12`. The `stacked_ens` rows, present after an extended run, are not used |

## Figures

`scripts/make_figures.py` writes each figure to `figures/fig0N_*.pdf` and `.png`, where N is
the figure number, and stops if a headline value differs from its source file.

| Item | Source file and columns |
|---|---|
| Fig. 1 (basin sample and comparison sets) | `data/processed/basin_meta.csv`; `data/processed/li2026_basin_coverage.csv` (`li_coverage`, `n_full_li_cells`, `n_full_native_mascons`); basin lists and counts from `results/phase6_li_comparison_perbasin.csv`, `results/phase6_li_comparison_summary.csv` and their copies in `results/jpl/` |
| Fig. 2a (benchmark skill) | `results/paper_baseline_ladder.csv` (`skill_vs_damped`, `damped_ref`); intervals from `results/paper_baseline_contrasts.csv` (`ci_lo`, `ci_hi` against the stronger variant), which has rows for `ridge_own_era5_flat12`, `kalman_ar1`, `ridge_own_flat12`, `kalman_own_ridge` and `ridge_own_perbasin` only |
| Fig. 2b (pooled RMSE in cm) | `results/conventional_metrics_summary.csv`: `pooled_rmse_cm` for the five Table C1 forecasts |
| Fig. 3 (example series, h = 1, cm) | `results/phase2_baseline_predictions.csv` (`damped_persistence_rho`, `target` and `pred` in cm, `target_std_units`); `results/kalman_predictions.csv` (`kalman_ar1`, standardized, returned to cm with the fold's training std, `target / target_std_units`); panel RMSEs checked against `rmse_cm` in `results/conventional_metrics_perbasin.csv` |
| Fig. 4a (zero-r variant) | `results/r0_ablation_summary.csv`: `skill_pct` of components `full_margin` (KF) and `rho_estimation_term` (zero-r variant); annotation from `noise_filtering_term` |
| Fig. 4b (two-variance variant) | `results/kalman_mission_summary.csv`: component `mission_split_term`, `skill_pct`, `ci_lo_pct`, `ci_hi_pct` |
| Fig. 4c (per-basin KF skill map) | `results/conventional_metrics_perbasin.csv`: `rmse_cm` of `kalman_ar1` and `damped_persistence` at h = 1 |
| Fig. 5 (where the ERA5-Land window helps) | `results/flat12_ridge_predictions.csv`: per-basin 1-month MSE of `ridge_own_era5_flat12` and `kalman_ar1`, per-basin DM tests with Benjamini-Hochberg q = 0.10; per-basin training-window residual std (median over folds) recomputed from `data/processed/basin_month_twsa_global.csv` |
| Fig. 6a (nonlinear models against KF) | `results/phase7_lstm_summary.csv` (`rmse_std`); seed ensemble and bootstrap intervals from `results/phase7_lstm_predictions.csv`; cross-checked against `results/paper_baseline_contrasts.csv` |
| Fig. 6b (KF-R12E refit on LSTM rows) | `results/flat12_train85_sensitivity.csv`: `ridge_own_era5_flat12_train85` against `lstm_own_era5_ens`, `skill_pct` and `dm_p` |
| Fig. 7a (CSR, against GRACE-FCast) | `results/phase6_li_comparison_headline.csv`, subset `joint_full_cells`, ours first: `skill`, `ci_lo`, `ci_hi`, `dm_p` |
| Fig. 7b (JPL, against GRACE-FCast) | `results/jpl/phase6_li_comparison_headline.csv`, subset `joint_full_cells`, converted as above; `results/jpl/phase6_li_comparison_summary.csv` (`rmse_std`) checks the conversion |

## Numbers in the text

| Item | Source file and columns |
|---|---|
| Abstract | `results/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_ladder.csv`, `results/r0_ablation_summary.csv`, `results/paper_baseline_contrasts.csv`, `results/centimeter_contrasts.csv` and the two `phase6_li_comparison_headline.csv` files, as in the rows below |
| Sect. 3.1, climatology drift | `results/climatology_offset.csv` (`rmse`, `offset_share`), from `scripts/climatology_offset.py` |
| Sect. 3.2, sample sizes | `n` in `results/paper_baseline_ladder.csv` and `results/jpl/paper_baseline_ladder.csv` |
| Sect. 3.4, boundary fits | `results/kalman_mission_summary.csv`, `diagnostic` rows (`stat_name`, `stat_value`) |
| Sect. 3.6, comparison basin sets | `data/processed/li2026_basin_coverage.csv`; `n_basins` and `rmse_std` by subset in `results/phase6_li_comparison_summary.csv` and `results/jpl/phase6_li_comparison_summary.csv`; `n_months` in the two headline files |
| Sect. 4.1, KF against the benchmarks | `results/paper_baseline_ladder.csv` (`skill_vs_damped`, `dm_p_vs_damped`); `results/paper_baseline_contrasts.csv` (`kalman_ar1` against `ridge_own_perbasin` and `ridge_own_lags`) |
| Sect. 4.1, JPL | `results/jpl/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_contrasts.csv` |
| Sect. 4.1, per-basin gain correlates | `results/kalman_gain_correlates.csv` (`spearman_rho`, `p`; median skill in the driest quarter and the other basins), from `scripts/kalman_gain_correlates.py` |
| Sect. 4.2, 1-month decomposition | `results/r0_ablation_summary.csv` (`rmse_a`, `skill_pct`, `dm_p`); `rmse_std` of `damped_persistence_rho` at h = 1 in `results/paper_baseline_ladder.csv` |
| Sect. 4.2, two-variance variant | `results/kalman_mission_summary.csv`: `mission_split_term` rows and the per-basin `diagnostic` rows |
| Sect. 4.3, own-basin corrections | `results/flat12_ridge_summary.csv` (`skill_vs_kalman`, `skill_vs_ridge_own`, `dm_p_vs_kalman`); `results/paper_baseline_contrasts.csv`; `results/paper_baseline_ladder.csv` |
| Sect. 4.3, nonlinear models | `results/phase7_lstm_summary.csv` (`dm_vs_ridge_twin`, `dm_p`); `results/flat12_train85_sensitivity.csv` |
| Sect. 4.4, comparison with GRACE-FCast | As Table 5; the damped-persistence check uses the `damped_persistence_rho` rows of both headline files and `kalman_ar1` against `damped_persistence_rho` at h = 1 in `results/jpl/paper_baseline_contrasts.csv` |
| Sect. 4.5, basin weighting | `results/conventional_metrics_summary.csv`; tests in cm from `results/centimeter_contrasts.csv` (`skill`, `dm_p`), built by `scripts/centimeter_contrasts.py` |
| Sect. 4.6, neighboring basins | `results/phase3b_summary.csv` (`skill_vs_own_ridge`, `placebo_beaten`) and `results/phase4_surrogate_summary.csv` (`beats_n_of`), with the same files in `results/jpl/` |
| Sect. 5, Discussion | `results/conventional_metrics_summary.csv` (`pooled_rmse_cm`, `rmse_cm_med`); `results/phase6_li_comparison_headline.csv` |
