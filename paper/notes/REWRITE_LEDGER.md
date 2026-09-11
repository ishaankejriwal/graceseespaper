# Manuscript number ledger (regenerated 2026-09-11 for the Kalman-reference manuscript)

This file lists every number that appears in `paper/main.tex` after the 2026-09-11 rewrite,
with the result file and column it was copied from. It replaces the pre-reframe ledger of
2026-08-15, whose numbers described the earlier three-finding manuscript. Every entry below
also has a `% source:` comment on the line before it in `main.tex`. Skills are stored as
fractions in the CSVs and printed as percent (x100) in the paper; RMSE is in standardized
units unless the column name ends in `_cm`.

Sign conventions. Every skill printed in the paper is ours-first (positive = our model has the
lower error). In `results/paper_baseline_contrasts.csv` and `results/flat12_ridge_summary.csv`
the challenger is listed first, so the stored value is used as is. In
`results/phase6_li_comparison_headline.csv` (CSR) the ours-first rows exist and are used as
stored. In `results/jpl/phase6_li_comparison_headline.csv` the rows are stored Li-first; the
paper converts them with the reciprocal `s' = 1 - 1/(1 - s)` (never `-s`) and Sect. 3.7 says so.
Updated 2026-09-11 (manuscript audit fixes): entries marked [audit] below were added or changed.

## Sample

| Number in paper | Source file | Column / row |
|---|---|---|
| 284 mask basins; 46 ice sheets + 4 water bodies excluded; 234 kept (CSR) | `data/processed/basin_meta.csv` | `exclude_reason` value counts; `docs/STUDY_CONTEXT.md` |
| 228 basins kept on JPL; 6 island masks without valid cells | `results/jpl/RUN_PROVENANCE.md` | "Basin geometry" |
| Fold issue windows 2019-06, 2020-11, 2022-04, 2023-09, 2025-02; targets to 2026-05 | `src/gracefc/evaluate.py` | `DEFAULT_FOLDS` |
| 83 lead-1 issue months per basin (17+17+17+17+15) | `src/gracefc/evaluate.py` | `DEFAULT_FOLDS` |
| n = 19,422 / 19,188 / 18,954 / 18,720 / 18,486 / 18,252 (CSR, leads 1-6) | `results/paper_baseline_ladder.csv` | `n` |
| n = 19,152 at lead 1 (JPL) = 228 basins x 84 issue months; 19,422 = 234 x 83 on CSR [audit] | `results/jpl/paper_baseline_ladder.csv`, `results/paper_baseline_ladder.csv` | `n`, horizon 1 |
| 11 ERA5-Land variables | `src/gracefc/era5.py` | variable list |
| 79 basins < 90 % ERA5 land coverage, 13 < 50 %, minimum 24.5 % | `data/processed/era5_basin_coverage.csv` | `era5_coverage` over the 234 keep basins |
| 12 NOAA indices; AMM and PDO dropped | `scripts/run_phase2_baselines.py` | index list |
| Lag set 0, 1, 2, 5, 11; ridge penalty 1 | `src/gracefc/features.py`, `src/gracefc/models.py` | `DEFAULT_LAGS`, `alpha` |
| 12-month window; 11 ERA5 channels; ridge penalty 1; MLP 64/32; LSTM 85/15 split, seeds 0-1; residual MLP seeds 0-2 | `src/gracefc/experiment_flat12.py`, `experiment_nonlinear.py`, `experiment_lstm.py`, `experiment_resmlp.py` | `LOOKBACK`, `_fit_head`, seeds |
| GRACE-FCast: 174 CSR initializations 2009-12 to 2024-05; 173 JPL to 2024-04 | results log Li ingestion entry; `results/jpl/RUN_PROVENANCE.md` | |
| CSR member training uncertainty 3.0 cm vs 2.8 cm | `docs/reference/li_kusche_2026_core.md` | sect. 4.1 |
| Li JPL member trained on RL06.1, scored against RL06.3Mv04 | `results/jpl/RUN_PROVENANCE.md` | "Li and Kusche comparison source" |
| 60 matched months on CSR = 17+17+17+9 (fold 4 truncated at the May 2024 final initialization); 59 on JPL = 17+17+17+8 [audit] | `results/phase6_li_comparison_headline.csv`, `results/jpl/phase6_li_comparison_headline.csv`; `src/gracefc/evaluate.py` DEFAULT_FOLDS | `n_months` |
| 227 matched basins on CSR; 227 of 228 on JPL | `results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv` | `n_basins`, subset `all_matched` |
| 229 basins with >= 1 full native CSR mascon; 211 with >= 1 full Li cell; 209 both | `data/processed/li2026_basin_coverage.csv` | `n_full_native_mascons`, `n_full_li_cells` |
| 209 strict basins on CSR; 67 on JPL | `results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv` | `n_basins`, subset `joint_full_cells` |
| 12,540 rows per lead (CSR strict); 3,953 (JPL strict) | same two headline files | `n_rows`, subset `joint_full_cells` |
| Fig. 1 counts: 209 in, 18 out, 7 unmatched (CSR); 67 in (JPL) | `data/processed/li2026_basin_coverage.csv`; JPL summary | 234 - 227 = 7; 227 - 209 = 18 |
| Li all_matched rmse_std 5.47 vs joint_full_cells 0.93 at lead 1 (JPL) | `results/jpl/phase6_li_comparison_summary.csv` | `rmse_std`, `li_lstm_full`, horizon 1 |
| Li: CC > 0.8 over 72 % (full) and 20 % (non-seasonal) of land at lead 1 | `docs/reference/li_kusche_2026_core.md` | sects. 5.1, 6.1 |

## Kalman reference and its variants

| Number in paper | Source file | Column / row |
|---|---|---|
| Kalman vs stronger damped: +4.98 / +8.79 / +5.62 / +3.07 / +2.55 / +3.63 % | `results/paper_baseline_ladder.csv` | `skill_vs_damped`, `kalman_ar1` |
| DM p: 3.5e-12 / 6.3e-19 / 5.7e-19 / 2.6e-11 / 7.9e-7 / 2.0e-9 | `results/paper_baseline_ladder.csv` | `dm_p_vs_damped`, `kalman_ar1` |
| Lead-1 CI +3.9 to +6.1 % | `results/paper_baseline_contrasts.csv` | `ci_lo`, `ci_hi`, `kalman_ar1` vs `damped_persistence_rho`, h1 |
| Weaker damped variant: -1.3 (h1, reg); -0.6 / -6.4 / -11.8 / -12.0 / -10.0 (h2-6, rho) | `results/paper_baseline_ladder.csv` | `skill_vs_damped` |
| JPL: AR(1) variant stronger at h1-h2, regression at h3-h6 [audit] | `results/jpl/paper_baseline_ladder.csv` | `damped_ref` |
| Kalman vs per-basin lag ridge: +1.99 / +3.04 / +1.43 / +1.04 / +2.17 / +3.92 %; p 6.6e-4 / 7.7e-7 / 0.013 / 0.078 / 0.0051 / 8.3e-6; h4 CI +0.02..+2.20 | `results/paper_baseline_contrasts.csv` | `kalman_ar1` vs `ridge_own_perbasin` |
| Kalman vs pooled lag ridge: +3.86 / +4.21 / +0.57 / -0.59 / -0.64 / +1.17 %; p 9.1e-6 / 1.2e-11 / 0.45 / 0.46 / 0.44 / 0.19 | `results/paper_baseline_contrasts.csv` | `kalman_ar1` vs `ridge_own_lags` |
| Persistence gap 16 to 23 % | `results/paper_baseline_ladder.csv` | `skill_vs_damped`, `persistence` (-0.158 to -0.227) |
| State ridge vs Kalman: -0.17 / -1.01 / +0.44 / +2.36 / +2.66 / +1.59 %; p 1.8e-6 / 1.9e-11 / 1.1e-14 / 1.6e-13 / 5.9e-14 / 2.3e-17 | `results/flat12_ridge_summary.csv` | `skill_vs_kalman`, `dm_p_vs_kalman`, `kalman_own_ridge` |
| JPL Kalman vs damped: +0.56 (p 0.60) / +6.35 / +4.64 / +1.55 / +2.36 / +4.74 %; p 2.7e-10 / 2.9e-9 / 0.017 / 9.8e-5 / 8.9e-11 | `results/jpl/paper_baseline_ladder.csv` | `skill_vs_damped`, `dm_p_vs_damped`, `kalman_ar1` |
| JPL Kalman vs per-basin lag ridge: -0.66 (p 0.46) / +3.21 / +2.44 / +2.11 / +3.92 / +6.67 %; p <= 0.026 | `results/jpl/paper_baseline_contrasts.csv` | `kalman_ar1` vs `ridge_own_perbasin` |
| Damped-persistence lead-1 RMSE 0.67 (JPL) vs 1.08 (CSR) | `results/jpl/paper_baseline_ladder.csv`, `results/paper_baseline_ladder.csv` | `rmse_std`, `damped_persistence_rho`, h1 |
| Noise-free ablation: +5.35 / +9.63 / +11.67 / +12.48 / +12.65 / +12.27 %; p <= 7.2e-14 | `results/r0_ablation_summary.csv` | `skill_pct`, `dm_p`, component `noise_filtering_term` |
| Noise-free filter vs regression-damped: +0.90 (p 1.6e-5) / -0.93 / -6.84 / -10.75 / -11.57 / -9.85 % | `results/r0_ablation_summary.csv` | `skill_pct`, component `rho_estimation_term` |
| Noise-free filter vs AR(1)-damped at h1: -0.40 % (RMSE 1.0783 vs 1.0761) [audit] | `results/r0_ablation_summary.csv` (`rmse_a`, `rho_estimation_term`, h1), `results/paper_baseline_ladder.csv` (`rmse_std`, `damped_persistence_rho`, h1) | 1 - (1.078260/1.076129)^2 |
| Lead-1 decomposition: (1 - 0.05354) x (1.078260/1.076129)^2 = 0.95021 = 1 - 0.0498 [audit] | same two files | `noise_filtering_term` h1 and the row above |
| Kalman vs regression-damped at h1 (Fig. 3a zero line): +6.21 % [audit] | `results/r0_ablation_summary.csv` | `skill_pct`, component `full_margin`, h1 |
| Mission split vs Kalman: -1.84 / -0.80 / -0.65 / -0.57 / -0.53 / -0.38 %; p 1.7e-7 / 1.1e-4 / 2.7e-4 / 5.3e-4 / 0.0020 / 0.0060 | `results/kalman_mission_summary.csv` | `skill_pct`, `dm_p`, component `mission_split_term` |
| Per-basin BH at lead 1: 0 helped, 0 hurt, 234 tested | `results/kalman_mission_summary.csv` | diagnostic rows `perbasin_h1_helped_bh`, `perbasin_h1_hurt_bh`, `perbasin_h1_n_tested` |
| 234 of 1170 basin-fold fits fall back to one variance; split 2018-01; minimum 12 FO months | `results/kalman_mission_summary.csv` | diagnostic rows `n_fallback_r_fo`, `n_fits`, `mission_split`, `min_fo_obs` |
| 259 of 1170 fits at the r = 0 boundary | `results/kalman_mission_summary.csv` | diagnostic row `n_r_one_at_boundary` |

## Corrections and sequence models

| Number in paper | Source file | Column / row |
|---|---|---|
| Flat-12 ridge vs Kalman: +0.81 / +1.48 / +1.76 / +1.81 / +1.49 / +0.63 %; p 0.0035 / 4.9e-6 / 2.9e-10 / 7.5e-10 / 3.4e-7 / 0.015 | `results/paper_baseline_contrasts.csv` | `ridge_own_flat12` vs `kalman_ar1` |
| ERA5 flat-12 ridge vs Kalman: +7.65 / +5.39 / +4.22 / +3.19 / +2.09 / +0.87 %; p 1.1e-8 / 1.1e-8 / 3.0e-8 / 5.2e-6 / 9.0e-4 / 0.12; h1 CI +4.7..+10.1 | `results/paper_baseline_contrasts.csv` | `ridge_own_era5_flat12` vs `kalman_ar1` |
| ERA5 flat-12 ridge vs damped: +12.24 / +13.71 / +9.61 / +6.16 / +4.58 / +4.47 % | `results/paper_baseline_ladder.csv` | `skill_vs_damped`, `ridge_own_era5_flat12` |
| ERA5 flat-12 ridge vs state ridge at h5, h6: -0.58 / -0.73 % | `results/flat12_ridge_summary.csv` | `skill_vs_ridge_own`, `ridge_own_era5_flat12` |
| RMSE 1.3804 vs 1.3764 (h5); 1.4421 vs 1.4369 (h6) | `results/paper_baseline_ladder.csv` | `rmse_std`, `ridge_own_era5_flat12`, `kalman_own_ridge` |
| Table 2 skills (all rows, one decimal) | `results/paper_baseline_ladder.csv` | `skill_vs_damped` x100 |
| Table 3 (corrections vs Kalman) | `results/flat12_ridge_summary.csv` (state ridge), `results/paper_baseline_contrasts.csv` (flat-12 ridges) | `skill_vs_kalman`/`dm_p_vs_kalman`; `skill`/`dm_p` |
| Table 4 RMSE at leads 1-3: Kalman 1.0490/1.1954/1.2734; ERA5 lag ridge 1.0254/1.1930/1.2685; flat-12 1.0448/1.1865/1.2622; ERA5 flat-12 1.0081/1.1627/1.2462; LSTM s0 1.0132/1.1758/1.2578, s1 1.0188/1.1851/1.2630; MLP s0 1.0235/1.1783/1.2602, s1 1.0291/1.1935/1.2799 | `results/phase7_lstm_summary.csv` | `rmse_std` |
| Residual MLP RMSE s0/s1/s2: 1.0148/1.0156/1.0145; 1.1760/1.1788/1.1773; 1.2635/1.2530/1.2540 | `results/phase7_resmlp_summary.csv` | `rmse_std`, `resmlp_own_era5_s{0,1,2}` |
| Flat-history MLP worse than flat ridge in every seed and lead, p <= 2.3e-6 (max over the six cells; was misprinted 1.1e-6) [audit] | `results/phase7_lstm_summary.csv` | `dm_vs_ridge_twin` (positive), `dm_p`, `mlp_own_era5_flat12_s{0,1}` (h1 1.9e-10, 1.1e-6; h2 2.3e-6, 4.1e-8; h3 5.5e-7, 8.3e-13) |
| Flat-12 ridge vs Kalman range quoted in Sect. 4.3: +0.63 to +1.81 % (lead 6 is the minimum) [audit] | `results/paper_baseline_contrasts.csv` | `ridge_own_flat12` vs `kalman_ar1` |
| Fig. 5 (sequence models) plots `mlp_own_era5_flat12_s0/s1` = the flat-history MLP, not the residual MLP [audit] | `results/phase7_lstm_predictions.csv`, `results/phase7_lstm_summary.csv` | |
| Row-matched flat ridge vs LSTM ensemble: +1.02 (p 0.094) / +2.76 (2.7e-10) / +2.27 (4.3e-9) % | `results/flat12_train85_sensitivity.csv` | `skill_pct`, `dm_p`, `ridge_own_era5_flat12_train85` vs `lstm_own_era5_ens` |
| Row-matched vs full-row flat ridge: -0.17 / +0.04 / +0.19 %, p 0.50 / 0.83 / 0.19 | `results/flat12_train85_sensitivity.csv` | vs `ridge_own_era5_flat12` |

## Comparison with GRACE-FCast (strict subset)

| Number in paper | Source file | Column / row |
|---|---|---|
| Kalman vs full (CSR): +16.4 (CI +4.9..+26.6, p 2.2e-3) / -6.0 (0.23) / -16.3 (1.4e-3) / -23.2 (3.9e-5) / -28.9 (1.7e-6) / -35.4 (1.7e-8) | `results/phase6_li_comparison_headline.csv` | subset `joint_full_cells`, `kalman_ar1` vs `li_lstm_full`, `skill`, `ci_lo`, `ci_hi`, `dm_p` |
| Kalman vs non-seasonal (CSR): +28.4 (2.7e-10) / +7.5 (0.018) / -4.0 (0.13) / -13.4 (4.3e-6) / -20.3 (7.8e-9) / -27.1 (5.9e-10) | same | `kalman_ar1` vs `li_lstm_nonseas` |
| ERA5 flat-12 vs full (CSR): +21.5 (4.0e-4) / -1.8 (0.74) / -12.1 (0.032) / -19.6 (1.7e-3) / -26.4 (7.3e-5) / -34.6 (4.8e-7) | same | `ridge_own_era5_flat12` vs `li_lstm_full` |
| ERA5 flat-12 vs non-seasonal (CSR): +32.8 (CI +24.0..+40.8, 2.4e-10) / +11.2 (CI +3.5..+18.5, 2.1e-3) / -0.2 (0.94) / -10.1 (3.2e-3) / -18.0 (6.7e-6) / -26.4 (3.3e-8) | same | `ridge_own_era5_flat12` vs `li_lstm_nonseas` |
| Damped persistence vs full product (CSR strict, ours first): +11.9 (p 0.029) / -16.9 / -30.9 / -42.2 / -47.9 / -54.3 [audit] | same | `damped_persistence_rho` vs `li_lstm_full`, `skill`, `dm_p` |
| "gives up 10 to 35 % of skill" at leads 4-6 (ours first: -10.1 to -35.4; the same rows product-first read +9.2 to +26.1) [audit] | same | the four pairs of Table 5 at h4-h6, both directions |
| Lead-by-lead pattern vs the product (tie = DM p >= 0.05): CSR full win/tie/loss from h3; CSR non-seasonal win/win/tie/loss from h4; JPL full win/tie/tie/loss from h4; JPL non-seasonal win/win/tie/tie/loss from h5 [audit] | both headline files | `dm_p`, subset `joint_full_cells` |
| Per-basin: full product beats Kalman in 98 (h1) ... 157 (h6) of 209; beats ERA5 flat-12 in 87 (h1) ... 162 (h6) | `results/phase6_li_comparison_perbasin.csv` | `a_better` summed by `model`, `vs`, `horizon` (Li first, so `a_better` = Li better) |
| JPL Kalman vs non-seasonal (ours first, converted 1 - 1/(1 - s) from stored -1.1238 / -0.2857 / -0.0750 / +0.0581 / +0.1525 / +0.2178): +52.9 (1.3e-13) / +22.2 (2.1e-5) / +7.0 (0.15) / -6.2 (0.16) / -18.0 (1.1e-4) / -27.9 (2.8e-6) [audit] | `results/jpl/phase6_li_comparison_headline.csv` | subset `joint_full_cells`, `li_lstm_nonseas` vs `kalman_ar1`, `skill` converted, `dm_p` |
| JPL Kalman vs full (ours first, converted from stored -0.7768 / -0.1091 / +0.0580 / +0.1604 / +0.2431 / +0.2952): +43.7 (7.1e-14) / +9.8 (0.087) / -6.2 (0.31) / -19.1 (1.7e-3) / -32.1 (1.2e-6) / -41.9 (1.5e-8) [audit] | same | `li_lstm_full` vs `kalman_ar1`, converted |
| JPL damped persistence vs full product at h1 (ours first): +43.2 (p 1.2e-13), converted from stored -0.7609 [audit] | same | `li_lstm_full` vs `damped_persistence_rho`, h1 |
| JPL Kalman vs damped at h1: +0.56 % (p 0.60), the lead-1 tie [audit] | `results/jpl/paper_baseline_contrasts.csv` | `kalman_ar1` vs `damped_persistence_rho`, h1 |

## Weighting sensitivity and Appendix C

| Number in paper | Source file | Column / row |
|---|---|---|
| Pooled RMSE cm: flat-12 5.118 / 5.634 / 5.915 (h1-3); Kalman 5.128 (h1), 5.128..6.690; ERA5 flat-12 5.231..6.774; state ridge 6.064 / 6.288 / 6.603 (h4-6); damped 5.321..6.935 | `results/conventional_metrics_summary.csv` | `pooled_rmse_cm` |
| Table C1 (all cells) | `results/conventional_metrics_summary.csv` | `pooled_rmse_cm` (3 dp), `rmse_cm_med`, `cc_anom_med`, `cc_full_med`, `nse_anom_med`, `nse_full_med` (2 dp) |
| Damped persistence lead-1 full-signal CC 0.86, NSE 0.72 | `results/conventional_metrics_summary.csv` | `cc_full_med`, `nse_full_med`, `damped_persistence`, h1 |
| Stacked LSTM ensemble (not a retained model) pooled RMSE 5.144 / 5.743 / 5.935 / 6.019 / 6.226 / 6.510 cm, below the state ridge at h4-h6 [audit] | `results/conventional_metrics_summary.csv` | `pooled_rmse_cm`, `stacked_ens` |
| Fig. 3c: filter better in 158 of 234 basins, worse in 76 (lead 1, RMSE in cm, AR(1) damped variant) [audit] | `results/conventional_metrics_perbasin.csv` | `rmse_cm`, `kalman_ar1` vs `damped_persistence`, horizon 1 |
| Fig. 4b: 3 basins below the -42 % axis limit [audit] | `results/flat12_ridge_predictions.csv` via `scripts/make_figures.py` | per-basin lead-1 skill |

## Dropped neighbor experiments

| Number in paper | Source file | Column / row |
|---|---|---|
| CSR correlation neighbor +0.31 % over state ridge at h1; 50 of 50 placebos | `results/phase3b_summary.csv` | `skill_vs_own_ridge`, `placebo_beaten`, `kalman_corr_top1`, h1 |
| CSR 99 of 99 surrogates at h1 | `results/phase4_surrogate_summary.csv` | `beats_n_of`, h1 |
| JPL correlation neighbor -0.52 / -0.97 / -1.16 / -1.14 / -1.16 / -1.28 %; 0 of 50 placebos at every lead | `results/jpl/phase3b_summary.csv` | `skill_vs_own_ridge`, `placebo_beaten`, `kalman_corr_top1` |
| JPL distance neighbor placebos: 0 / 0 / 0 / 8 / 26 / 9 of 50 | `results/jpl/phase3b_summary.csv` | `placebo_beaten`, `kalman_geo_top1` |
| JPL surrogates 0 of 99 at every lead | `results/jpl/phase4_surrogate_summary.csv` | `beats_n_of` |

## Appendix B

| Number in paper | Source file | Column / row |
|---|---|---|
| Table B1, all RMSE and p cells | `results/paper_baseline_ladder.csv` | `rmse_std` (4 dp), `dm_p_vs_damped` (2 s.f.); weaker damped row is `damped_persistence_reg` at h1 and `damped_persistence_rho` at h2-6 |

## Numbers that appear in the abstract and conclusions

All are repeats of entries above: +5.0 / +8.8 / +2.5..+5.6 % (ladder, `kalman_ar1`), +7.6 % (`ridge_own_era5_flat12` vs `kalman_ar1`, h1), the row-matched LSTM p values (0.094 / 2.7e-10 / 4.3e-9), the lead-by-lead win/tie/loss pattern against the product, the +5.0 % (CSR) and +0.56 % (JPL, p 0.60) filter-over-damped margins at lead 1, 234 basins, five folds, 2019 to 2026 issue window.

## Figure numbering (2026-09-11)

Figures are numbered by first citation: Fig. 1 basins, Fig. 2 ladder, Fig. 3 filter mechanism
(`fig03_filter_mechanism`), Fig. 4 where ERA5 helps (`fig04_era5_where`), Fig. 5 sequence models
(`fig05_sequence_models`), Fig. 6 crossing against the published product (`fig06_crossing`).
