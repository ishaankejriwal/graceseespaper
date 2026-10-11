# Archive manifest: manuscript tables and figures to source files

This mapping lists, for each table and figure in `paper/main.tex` (rewritten 2026-09-11 to the
Kalman-benchmark structure, then rewritten for readability the same day with model abbreviations
KF, KF-R1, KF-R12, KF-R12E and KF-R1E; figures numbered by first citation), the archived result
file(s) it is drawn from. Include this file in the Zenodo deposit (`scripts/build_deposit.py` stages it; see `docs/DEPOSIT_README.md`). Appendix A (Table A1) of the
paper maps each model name to its identifier in the result files. Every skill in the paper is
stated with our model first; the JPL comparison file stores GRACE-FCast first and is converted as
`1 - 1/(1 - s)`.

| Manuscript item | Archived source file(s) |
|---|---|
| Table 1 (forecast names) | none; definitions in Sect. 3. Appendix A (Table A1) maps names to identifiers |
| Table 2 (benchmark forecasts, skill vs stronger damped persistence), Fig. 2a | `results/paper_baseline_ladder.csv` (`skill_vs_damped`, `damped_ref`); confidence intervals in `results/paper_baseline_contrasts.csv` (`ci_lo`, `ci_hi` against `damped_persistence_rho` at h1 and `damped_persistence_reg` at h2-h6; rows exist for `ridge_own_era5_flat12`, `kalman_ar1`, `ridge_own_flat12`, `kalman_own_ridge`, `ridge_own_perbasin`) |
| Fig. 2b (pooled RMSE in cm, the five forecasts reported in cm) | `results/conventional_metrics_summary.csv` (`pooled_rmse_cm`; the `stacked_ens` rows are not plotted) |
| Fig. 3 (example time series: observed residual, damped persistence and KF at a 1-month lead, four basins, cm) | `results/phase2_baseline_predictions.csv` (`damped_persistence_rho`, horizon 1, `target` and `pred` in cm, `target_std_units`); `results/kalman_predictions.csv` (`kalman_ar1`, horizon 1, standardized; returned to cm with the per-fold training std implied by `target / target_std_units`); panel RMSEs asserted against `results/conventional_metrics_perbasin.csv` (`rmse_cm`, horizon 1) |
| Sect. 4.1 per-basin gain correlates (Spearman -0.34 with residual lag-1 autocorrelation, -0.27 with area, -0.30 with precipitation; median gain 5.0 % in the driest quarter vs 0.9 %) | `results/kalman_gain_correlates.csv` (built by `scripts/kalman_gain_correlates.py` from `results/conventional_metrics_perbasin.csv`, `results/phase2_baseline_predictions.csv` (residual AC1 over the test window), `data/processed/basin_meta.csv` and `data/processed/era5_basin_month.csv` (full years 2001-2025)); driest-quarter median 5.0 % |
| Sect. 3.1 climatology drift (pooled RMSE 1.62 at h1; 57 % of its MSE from the basin-fold mean residual) | `results/climatology_offset.csv` (built by `scripts/climatology_offset.py` from `results/phase2_baseline_predictions.csv`) |
| Sect. 4.5 centimeter tests (KF vs damped persistence +7.1 % at h1, p < 0.001 at every lead; KF-R12E vs KF p >= 0.25) | `results/centimeter_contrasts.csv` (built by `scripts/centimeter_contrasts.py`, which reuses the row construction of `scripts/compute_conventional_metrics.py`) |
| Table 3 (own-basin models vs KF) | `results/flat12_ridge_summary.csv` (KF-R1: `skill_vs_kalman`, `dm_p_vs_kalman`); `results/paper_baseline_contrasts.csv` (KF-R12 and KF-R12E vs `kalman_ar1`) |
| Table 4 (nonlinear models, standardized RMSE at leads 1-3) | `results/phase7_lstm_summary.csv`, `results/phase7_resmlp_summary.csv` (`rmse_std`) |
| Table 5, CSR block (KF, KF-R12E and damped persistence vs GRACE-FCast full and non-seasonal, fully covered basins), Fig. 7a | `results/phase6_li_comparison_headline.csv` (subset `joint_full_cells`, ours-first rows incl. `damped_persistence_rho` vs `li_lstm_full` and vs `li_lstm_nonseas`); per-basin wins from `results/phase6_li_comparison_perbasin.csv` (`a_better`) |
| Table 5, JPL block (KF and damped persistence vs GRACE-FCast full and non-seasonal, all leads), Fig. 7b | `results/jpl/phase6_li_comparison_headline.csv` (subset `joint_full_cells`, stored product-first, converted as `1 - 1/(1 - s)`; `dm_p` unchanged); `results/jpl/phase6_li_comparison_summary.csv` (`rmse_std`, used to verify the conversion) |
| Fig. 1 (basin maps, fully covered comparison set) | `data/processed/li2026_basin_coverage.csv` (`li_coverage`, `n_full_li_cells`, `n_full_native_mascons`), `data/processed/basin_meta.csv`; JPL membership from `results/jpl/phase6_li_comparison_perbasin.csv` and `results/jpl/phase6_li_comparison_summary.csv` |
| Fig. 4a (KF and the zero-r variant vs the regression damped variant; no CI) | `results/r0_ablation_summary.csv` (components `full_margin`, `rho_estimation_term`, `noise_filtering_term`, `skill_pct`) |
| Fig. 4b (two-variance variant vs the single-variance KF, with CI) | `results/kalman_mission_summary.csv` (component `mission_split_term`, `skill_pct`, `ci_lo_pct`, `ci_hi_pct`) |
| Fig. 4c (per-basin 1-month skill of KF over the AR(1) damped variant, from RMSE in cm) | `results/conventional_metrics_perbasin.csv` (`rmse_cm` of `kalman_ar1` and `damped_persistence` at horizon 1; built by `scripts/compute_conventional_metrics.py`, where `damped_persistence` is `damped_persistence_rho` at h1) |
| Fig. 5 (where the ERA5-Land window helps) | `results/flat12_ridge_predictions.csv` (per-basin 1-month MSE of `ridge_own_era5_flat12` and `kalman_ar1`; per-basin DM with BH q = 0.10); training-window residual standard deviation per basin from the fold deseasonalization (`src/gracefc/evaluate.py`) on `data/processed/basin_month_twsa_global.csv` |
| Fig. 6 (nonlinear models vs KF; MLP seeds `mlp_own_era5_flat12_s0/s1`, LSTM ensemble) | `results/phase7_lstm_summary.csv` (`rmse_std`, asserted), `results/phase7_lstm_predictions.csv` (seed ensemble and bootstrap CIs), `results/flat12_train85_sensitivity.csv` (panel b), `results/paper_baseline_contrasts.csv` (cross-check) |
| Sect. 4.1 JPL benchmark paragraph | `results/jpl/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_contrasts.csv` |
| Sect. 4.2 1-month decomposition (model-noise term +5.35 %, zero-r variant -0.40 % vs the AR(1) variant) | `results/r0_ablation_summary.csv` (`rmse_a`, `skill_pct`), `results/paper_baseline_ladder.csv` (`rmse_std`, `damped_persistence_rho`, h1) |
| Sect. 4.4 persistence-class check (damped persistence vs GRACE-FCast at 1 month, CSR and JPL) | `results/phase6_li_comparison_headline.csv`, `results/jpl/phase6_li_comparison_headline.csv` (`damped_persistence_rho` rows); `results/jpl/paper_baseline_contrasts.csv` (`kalman_ar1` vs `damped_persistence_rho`, h1) |
| Sect. 4.5 (weighting sensitivity), Appendix C (Table C1, conventional metrics) | `results/conventional_metrics_summary.csv` (five forecasts; the `stacked_ens` rows are quoted once in Appendix C as an extended-run entry supporting no claim); per-basin values in `results/conventional_metrics_perbasin.csv` (built by `scripts/compute_conventional_metrics.py`) |
| Sect. 4.6 (neighboring basins, set aside) | `results/phase3b_summary.csv`, `results/phase4_surrogate_summary.csv` (CSR); `results/jpl/phase3b_summary.csv`, `results/jpl/phase4_surrogate_summary.csv` (JPL) |
| Appendix B (Table B1, full benchmark table with RMSE and p) | `results/paper_baseline_ladder.csv` (`rmse_std`, `dm_p_vs_damped`) |
| Methods, Sect. 3.2 (sample sizes; 228 x 84 on JPL, 234 x 83 on CSR) | `results/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_ladder.csv` (`n`) |
| Methods, Sect. 3.6 (matched and fully covered basin sets; fold-4 truncation; JPL matched-set inflation) | `results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv` (`n_basins`, `rmse_std`); the two headline files (`n_months`) |
| Methods, Sect. 3.4 (boundary fits) | `results/kalman_mission_summary.csv` (diagnostic rows) |

Figures are built by `scripts/make_figures.py` into `figures/fig01_basins`, `fig02_benchmark_ladder`,
`fig03_example_series`, `fig04_filter_mechanism`, `fig05_era5_where`, `fig06_sequence_models`,
`fig07_crossing` (PDF and PNG); the `fig0N_` prefix is the manuscript figure number. Every headline
value drawn is asserted against its source file at build time (`figures/BUILD_NOTES.md`).
