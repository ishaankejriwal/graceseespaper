# Archive manifest: manuscript tables and figures to source files

This mapping lists, for each table and figure in `paper/main.tex` (rewritten 2026-09-11 to the
Kalman-reference structure), the archived result file(s) it is drawn from. Include this file
in the Zenodo deposit. The number-by-number ledger is `paper/notes/REWRITE_LEDGER.md`.

| Manuscript item | Archived source file(s) |
|---|---|
| Table 1 (model names) | none; definitions in Sect. 3. Appendix A (Table A1) maps names to identifiers |
| Table 2 (reference ladder, skill vs stronger damped persistence), Fig. 2a | `results/paper_baseline_ladder.csv` (`skill_vs_damped`); confidence intervals in `results/paper_baseline_contrasts.csv` (`ci_lo`, `ci_hi` against the stronger damped variant) |
| Fig. 2b (pooled RMSE in cm) | `results/conventional_metrics_summary.csv` (`pooled_rmse_cm`) |
| Table 3 (corrections vs the Kalman reference) | `results/flat12_ridge_summary.csv` (state ridge: `skill_vs_kalman`, `dm_p_vs_kalman`); `results/paper_baseline_contrasts.csv` (flat-12 and ERA5 flat-12 ridges vs `kalman_ar1`) |
| Table 4 (sequence models, standardized RMSE at leads 1-3), Fig. 6 | `results/phase7_lstm_summary.csv`, `results/phase7_resmlp_summary.csv` (`rmse_std`); `results/flat12_train85_sensitivity.csv` (equalized-training-window comparison) |
| Table 5 (comparison with GRACE-FCast, CSR strict subset), Fig. 3a | `results/phase6_li_comparison_headline.csv` (subset `joint_full_cells`); per-basin wins from `results/phase6_li_comparison_perbasin.csv` (`a_better`) |
| Fig. 3b (JPL, Kalman reference only) | `results/jpl/phase6_li_comparison_headline.csv` (subset `joint_full_cells`) |
| Fig. 1 (basin maps, strict-subset membership) | `data/processed/li2026_basin_coverage.csv` (`li_coverage`, `n_full_li_cells`, `n_full_native_mascons`), `data/processed/basin_meta.csv`; JPL membership from `results/jpl/phase6_li_comparison_summary.csv` |
| Fig. 4a (noise-free ablation) | `results/r0_ablation_summary.csv` (component `noise_filtering_term`) |
| Fig. 4b (mission-split filter) | `results/kalman_mission_summary.csv` (component `mission_split_term`) |
| Fig. 4c (per-basin lead-1 skill of the filter over damped persistence) | `results/kalman_predictions.csv` (per-basin MSE of `kalman_ar1` and the stronger damped variant at horizon 1) |
| Fig. 5 (where the ERA5 window helps) | `results/flat12_ridge_predictions.csv` (per-basin lead-1 MSE of `ridge_own_era5_flat12` and `kalman_ar1`; per-basin DM with BH q = 0.10); training-window residual standard deviation per basin from the fold deseasonalization (`src/gracefc/evaluate.py`) |
| Sect. 4.1 JPL ladder paragraph | `results/jpl/paper_baseline_ladder.csv`, `results/jpl/paper_baseline_contrasts.csv` |
| Sect. 4.5 (weighting sensitivity), Appendix C (Table C1, conventional metrics) | `results/conventional_metrics_summary.csv`; per-basin values in `results/conventional_metrics_perbasin.csv` (built by `scripts/compute_conventional_metrics.py`) |
| Sect. 4.6 (dropped neighbor experiments) | `results/phase3b_summary.csv`, `results/phase4_surrogate_summary.csv` (CSR); `results/jpl/phase3b_summary.csv`, `results/jpl/phase4_surrogate_summary.csv` (JPL) |
| Appendix B (Table B1, full ladder with RMSE and p) | `results/paper_baseline_ladder.csv` (`rmse_std`, `dm_p_vs_damped`) |
| Methods, Sect. 3.6 (matched and strict populations; JPL matched-population inflation) | `results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv` (`n_basins`, `rmse_std`) |
| Methods, Sect. 3.4 (boundary fits) | `results/kalman_mission_summary.csv` (diagnostic rows) |

Figures are built by `scripts/make_figures.py` into `figures/fig01_basins`, `fig02_benchmark_ladder`,
`fig03_crossing`, `fig04_filter_mechanism`, `fig05_era5_where`, `fig06_sequence_models` (PDF and PNG).
