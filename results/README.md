# results/

Result tables written by the pipeline scripts (`scripts/run_chain.py` and the scripts it calls).
CSR results are in this directory and JPL results in `jpl/`. `docs/ARCHIVE_MANIFEST.md` gives
the file and column behind each table, figure and quoted number in the paper.

The files behind the main results:

| File | Contents |
|---|---|
| `paper_baseline_ladder.csv` | The benchmark forecasts, KF, KF-R1, KF-R12 and KF-R12E at leads 1 to 6 on matched rows, with skill against the stronger damped-persistence variant at each lead (Table 2, Appendix B) |
| `paper_baseline_contrasts.csv` | Pairwise contrasts behind the ladder: skill, bootstrap interval, DM statistic and p value |
| `flat12_ridge_summary.csv` | KF-R1, KF-R12 and KF-R12E against KF and against the own-basin ridge (Table 3) |
| `r0_ablation_summary.csv`, `kalman_mission_summary.csv` | The zero-r and two-variance variants of KF (Fig. 4); `kalman_mission_summary.csv` also holds the fitting diagnostics |
| `phase6_li_comparison_headline.csv` | KF, KF-R1, KF-R12, KF-R12E, damped persistence and the per-basin lag ridge against GRACE-FCast full and non-seasonal, in both orientations, on the `all_matched`, `coverage_ge_0.5` and `joint_full_cells` subsets (Table 5) |
| `conventional_metrics_summary.csv` | RMSE in cm, correlation and NSE for damped persistence, KF, KF-R1, KF-R12 and KF-R12E (Appendix C); `stacked_ens` is added when the extended stacked-model prediction files exist |
| `phase7_lstm_summary.csv`, `phase7_resmlp_summary.csv` | Nonlinear models (Table 4, Fig. 6) |
| `jpl/` | The same tables from the JPL run, mainly `jpl/paper_baseline_ladder.csv` and `jpl/phase6_li_comparison_headline.csv`; the run record is `jpl/RUN_PROVENANCE.md` |

File names follow a pattern. `*_summary.csv` holds scores per model and lead,
`*_headline.csv` the rows the paper quotes, `*_predictions.csv` one row per model, basin, issue
month and lead, and `*_placebo_*.csv` the null draws of the neighbour and surrogate tests.
`chain_<step>.log` is the console output of a chain step and is not archived.

Skill is 1 - MSE(model)/MSE(vs); a positive value means the model in the `model` column has the
lower error. Errors are in standardized units unless the column name says cm.

The large row-level files ship in the data archive of the Zenodo deposit instead of git:
`*_predictions.csv`, `*_placebo_basin.csv` and `phase3b_placebo_monthly.csv` here, and in
`jpl/` the predictions, all placebo tables and `kalman_fold_params.pkl`.
`SHA256_MANIFEST_LIVE.csv` (here and in `jpl/`) records their sizes and SHA-256 checksums. Verify them with

```bash
python scripts/make_manifest.py --check
GRACEFC_SOURCE=jpl python scripts/make_manifest.py --check
```

`kalman_fold_params.pkl` caches the fitted KF parameters per fold. The `flat12_ridge` and `phase3b`
steps rewrite it when it is missing or out of date.
