# Archive manifest: manuscript tables and figures → source files

This mapping was formerly printed in the manuscript as Appendix Table A2
(`tab:sources`); it now lives with the archive, where its user needs it.
Include this file (or fold it into the README) in the Zenodo deposit.

| Manuscript item | Archived source file(s) |
|---|---|
| Table 1 (baseline ladder), Fig. 1 | `results/paper_baseline_ladder.csv`; pairwise tests in `results/paper_baseline_contrasts.csv` |
| Table 3 (crossing), Fig. 2 | `results/phase8b_li_comparison_headline.csv`, `results/phase8b_li_comparison_perbasin.csv` |
| Controls table, Fig. 4 | `results/phase3b_summary.csv`, `results/phase6_era5_headline.csv` |
| Sharing–area cross-stratification (Sect. on the linear neighbor result) | `results/resolution_cross_2x2.csv`, `results/resolution_sweep_area.csv`, `results/resolution_sweep_contamination.csv` |
| Stacked-system table, Fig. 5 | `results/phase8b_h16_ensemble_headline.csv`, `results/phase8b_h16_headline.csv`, `results/phase8b_h16_summary.csv`, `results/phase8b_lstm_h46_placebo_monthly.csv` |
| Stratification of the stacked correction | `results/phase8_stratification.csv` |
| Forcing and architecture comparisons, ERA5 attribution table | `results/phase6_era5_headline.csv`, `results/phase7_corrected_analysis.md`, `results/phase6_era5_attribution.csv` (and `_folds` / `_continent` / `_fdr` companions), `results/flat12_train85_sensitivity.csv` |
| Fig. 3 (neighbor map), Fig. 6 (complementarity) | `results/phase5_perbasin_fdr_h1.csv`, `results/phase6_basin_analysis_summary.csv`, `results/phase6_basin_analysis_q4.csv` |
| Conventional-metrics table (Sect. on comparability: per-basin RMSE cm / CC / NSE, anomaly and full signal) | `results/conventional_metrics_summary.csv`; per-basin values in `results/conventional_metrics_perbasin.csv` (built by `scripts/compute_conventional_metrics.py`) |
