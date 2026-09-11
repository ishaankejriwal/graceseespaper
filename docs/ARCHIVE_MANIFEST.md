# Archive manifest: manuscript tables and figures to source files

This mapping lists, for each table and figure in `paper/main.tex` (rewritten 2026-09-11 to the
Kalman-reference structure; figures renumbered by first citation on the same day), the archived
result file(s) it is drawn from. Include this file in the Zenodo deposit. The number-by-number
ledger is `paper/notes/REWRITE_LEDGER.md`. Every skill in the paper is stated with our model
first; the JPL comparison file stores the product first and is converted as `1 - 1/(1 - s)`.

| Manuscript item | Archived source file(s) |
|---|---|
| Table 1 (model names) | none; definitions in Sect. 3. Appendix A (Table A1) maps names to identifiers |
| Table 2 (reference ladder, skill vs stronger damped persistence), Fig. 2a | `results/paper_baseline_ladder.csv` (`skill_vs_damped`, `damped_ref`); confidence intervals in `results/paper_baseline_contrasts.csv` (`ci_lo`, `ci_hi` against `damped_persistence_rho` at h1 and `damped_persistence_reg` at h2-h6; rows exist for `ridge_own_era5_flat12`, `kalman_ar1`, `ridge_own_flat12`, `kalman_own_ridge`, `ridge_own_perbasin`) |
| Fig. 2b (pooled RMSE in cm, five retained models) | `results/conventional_metrics_summary.csv` (`pooled_rmse_cm`; the `stacked_ens` rows are not plotted) |
| Table 3 (corrections vs the Kalman reference) | `results/flat12_ridge_summary.csv` (state ridge: `skill_vs_kalman`, `dm_p_vs_kalman`); `results/paper_baseline_contrasts.csv` (flat-12 and ERA5 flat-12 ridges vs `kalman_ar1`) |
| Table 4 (sequence models, standardized RMSE at leads 1-3) | `results/phase7_lstm_summary.csv`, `results/phase7_resmlp_summary.csv` (`rmse_std`) |
| Table 5 (comparison with GRACE-FCast, CSR strict subset), Fig. 6a | `results/phase6_li_comparison_headline.csv` (subset `joint_full_cells`, ours-first rows); per-basin wins from `results/phase6_li_comparison_perbasin.csv` (`a_better`) |
| Fig. 1 (basin maps, strict-subset membership) | `data/processed/li2026_basin_coverage.csv` (`li_coverage`, `n_full_li_cells`, `n_full_native_mascons`), `data/processed/basin_meta.csv`; JPL membership from `results/jpl/phase6_li_comparison_perbasin.csv` and `results/jpl/phase6_li_comparison_summary.csv` |
| Fig. 3a (Kalman reference and noise-free filter vs the regression damped variant; no CI) | `results/r0_ablation_summary.csv` (components `full_margin`, `rho_estimation_term`, `noise_filtering_term`, `skill_pct`) |
| Fig. 3b (mission-split filter vs the one-variance filter, with CI) | `results/kalman_mission_summary.csv` (component `mission_split_term`, `skill_pct`, `ci_lo_pct`, `ci_hi_pct`) |
| Fig. 3c (per-basin lead-1 skill of the filter over the AR(1) damped variant, from RMSE in cm) | `results/conventional_metrics_perbasin.csv` (`rmse_cm` of `kalman_ar1` and `damped_persistence` at horizon 1; built by `scripts/compute_conventional_metrics.py`, where `damped_persistence` is `damped_persistence_rho` at h1) |
| Fig. 4 (where the ERA5 window helps) | `results/flat12_ridge_predictions.csv` (per-basin lead-1 MSE of `ridge_own_era5_flat12` and `kalman_ar1`; per-basin DM with BH q = 0.10); training-window residual standard deviation per basin from the fold deseasonalization (`src/gracefc/evaluate.py`) on `data/processed/basin_month_twsa_global.csv` |
| Fig. 5 (sequence models vs the Kalman reference; flat-history MLP seeds `mlp_own_era5_flat12_s0/s1`, LSTM ensemble) | `results/phase7_lstm_summary.csv` (`rmse_std`, asserted), `results/phase7_lstm_predictions.csv` (seed ensemble and bootstrap CIs), `results/flat12_train85_sensitivity.csv` (panel b), `results/paper_baseline_contrasts.csv` (cross-check) |
| Fig. 6b (JPL, Kalman reference only) | `results/jpl/phase6_li_comparison_headline.csv` (subset `joint_full_cells`, stored product-first, converted); `results/jpl/phase6_li_comparison_summary.csv` (`rmse_std`, used to verify the conversion) |
| Sect. 4.1 JPL ladder paragraph | `results/jpl/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_contrasts.csv` |
| Sect. 4.2 lead-1 decomposition (noise term +5.35 %, noise-free filter -0.40 % vs the AR(1) variant) | `results/r0_ablation_summary.csv` (`rmse_a`, `skill_pct`), `results/paper_baseline_ladder.csv` (`rmse_std`, `damped_persistence_rho`, h1) |
| Sect. 4.4 persistence-class check (damped persistence vs the product at lead 1, CSR and JPL) | `results/phase6_li_comparison_headline.csv`, `results/jpl/phase6_li_comparison_headline.csv` (`damped_persistence_rho` rows); `results/jpl/paper_baseline_contrasts.csv` (`kalman_ar1` vs `damped_persistence_rho`, h1) |
| Sect. 4.5 (weighting sensitivity), Appendix C (Table C1, conventional metrics) | `results/conventional_metrics_summary.csv` (five retained models; `stacked_ens` rows quoted once in Sect. 4.5 as not part of the paper's claims); per-basin values in `results/conventional_metrics_perbasin.csv` (built by `scripts/compute_conventional_metrics.py`) |
| Sect. 4.6 (dropped neighbor experiments) | `results/phase3b_summary.csv`, `results/phase4_surrogate_summary.csv` (CSR); `results/jpl/phase3b_summary.csv`, `results/jpl/phase4_surrogate_summary.csv` (JPL) |
| Appendix B (Table B1, full ladder with RMSE and p) | `results/paper_baseline_ladder.csv` (`rmse_std`, `dm_p_vs_damped`) |
| Methods, Sect. 3.2 (sample sizes; 228 x 84 on JPL, 234 x 83 on CSR) | `results/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_ladder.csv` (`n`) |
| Methods, Sect. 3.6 (matched and strict populations; fold-4 truncation; JPL matched-population inflation) | `results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv` (`n_basins`, `rmse_std`); the two headline files (`n_months`) |
| Methods, Sect. 3.4 (boundary fits) | `results/kalman_mission_summary.csv` (diagnostic rows) |

Figures are built by `scripts/make_figures.py` into `figures/fig01_basins`, `fig02_benchmark_ladder`,
`fig03_filter_mechanism`, `fig04_era5_where`, `fig05_sequence_models`, `fig06_crossing` (PDF and PNG);
the `fig0N_` prefix is the manuscript figure number.
