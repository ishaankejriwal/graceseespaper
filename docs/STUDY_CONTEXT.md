# Project Context — Global TWSA Forecasting Study

Last updated: 2026-09-10.

What a new collaborator needs today: what the study is, what it claims, where each number comes
from, how to run it, and what is still open. Per-run history lives in `results/RUN_LOG.md`; code
layout in `docs/CODE_MAP.md`; the plain-language introduction in the top-level `README.md`.

> Naming note: `docs/history/africa_pilot_jpl.md` and `docs/history/africa_pilot_csr.md` are the
> older Africa-only work (JPL and CSR respectively). This file covers the current global study.

## The study

We forecast deseasonalized terrestrial water storage anomalies for 234 global HydroSHEDS basins
at leads 1 to 6 months. The target is the CSR RL06.3 mascon basin mean with the long-term trend
and the annual cycle removed and the residual standardized per basin, all fitted on training
months only. Evaluation is 5 expanding-window folds split on issue dates. The issue windows
run from 2019-06 to 2026-04, so the last scored target month is 2026-05
(`results/kalman_predictions.csv`). Diebold-Mariano tests and FDR correction throughout. The
same pipeline runs on the JPL RL06.3Mv04 mascons with `--source jpl`.

## The claims

### 1. The reference forecast for this problem should be a filter, not persistence

A per-basin Kalman AR(1) plus observation-noise filter is proposed as the reference forecast for
deseasonalized basin TWSA. It handles two things damped persistence cannot: observation noise,
which persistence carries forward intact, and the 2017 to 2018 mission gap, across which the
filter simply propagates its state.

Against the stronger damped variant at each lead, on matched rows (19,422 at lead 1 down to
18,252 at lead 6), from `results/paper_baseline_ladder.csv`:

| lead | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `kalman_ar1` skill | +4.98% | +8.79% | +5.62% | +3.07% | +2.55% | +3.63% |
| DM p | 3.5e-12 | 6.3e-19 | 5.7e-19 | 2.6e-11 | 7.9e-07 | 2.0e-09 |

It also beats a per-basin ridge at all six leads, by +1.99/+3.04/+1.43/+1.04/+2.17/+3.92%, and
significantly at five of them (`results/paper_baseline_contrasts.csv`, `kalman_ar1` against
`ridge_own_perbasin`). Lead 4 is the one that misses significance: a gain of +1.04% at p =
0.078, not a loss. The win is the noise removal, not better drift estimation: with `r` forced to
zero the filter gives up 5.35% at lead 1, rising to a peak of 12.65% at lead 5 and easing to
12.27% at lead 6, and that stripped version is worse than damped persistence at leads 2 to 6,
ranging from +0.90% at lead 1 to -11.57% at lead 5 (`results/r0_ablation_summary.csv`).

Identification caveat, unchanged: `r` is whatever part of the signal the AR(1) state cannot carry
forward, not demonstrably measurement noise. 259 of 1170 single-variance fits sit at the r = 0
boundary (`results/kalman_mission_summary.csv`, diagnostic `n_r_one_at_boundary`).

### 2. At leads 1 to 4 the strongest own-basin model is the filter plus a flat-12 ridge correction

`ridge_own_era5_flat12`: a ridge correction on a flat 12-month history of the filtered state and
the 11 ERA5 variables, predicting the Kalman forecast's error. From
`results/paper_baseline_ladder.csv` and `results/paper_baseline_contrasts.csv`:

| lead | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| skill vs damped | +12.24% | +13.71% | +9.61% | +6.16% | +4.58% | +4.47% |
| skill vs `kalman_ar1` | +7.65% | +5.39% | +4.22% | +3.19% | +2.09% | +0.87% (ns, p = 0.12) |

It is the strongest own-basin model only at leads 1 to 4. At leads 5 and 6 the own-state ridge
correction `kalman_own_ridge` is marginally better: `ridge_own_era5_flat12` scores -0.58%
against it at lead 5 and -0.73% at lead 6 (`results/flat12_ridge_summary.csv`, column
`skill_vs_ridge_own`), which in RMSE is 1.3804 against 1.3764 at lead 5 and 1.4421 against
1.4369 at lead 6 (`results/paper_baseline_ladder.csv`).

"Strongest" here means lowest pooled MSE in standardized units, where every basin counts
equally. In raw centimetres, where the largest basins dominate, the ranking changes: the plain
flat-12 ridge without ERA5, `ridge_own_flat12`, has the lowest pooled RMSE at leads 1 to 3
(5.118 cm at lead 1 against 5.128 for the filter and 5.231 for the ERA5 ridge), the own-state
ridge correction is lowest among retained models at leads 4 to 6, and the ERA5 ridge sits behind
the plain filter at every lead (`results/conventional_metrics_summary.csv`, column
`pooled_rmse_cm`). The ERA5 gain is therefore concentrated in basins with small storage
variance, and the paper has to state which weighting a ranking refers to.

Every sequence model trained in this repository loses to it once history length is equalized.
At lead 1 it has the lowest RMSE of every arm in `results/phase7_lstm_summary.csv`, and with the
training window matched at 85% it is ahead of the two-seed LSTM ensemble by +1.02/+2.76/+2.27%
at leads 1 to 3 (`results/flat12_train85_sensitivity.csv`).

### 3. Against Li and Kusche's GRACE-FCast: both models on CSR, the Kalman reference on JPL, one protocol

The JPL run predates the flat-12 step, so only `kalman_ar1` has JPL comparison rows so far
(`results/jpl/phase6_li_comparison_headline.csv`); the JPL rerun with the current default chain
is pending.

One strict subset rule for both products: `joint_full_cells` keeps a basin only if it fully
contains at least one native mascon of the product being scored and at least one valid 1-degree
Li cell. That is 209 of 227 basins on CSR and 67 on JPL
(`results/phase6_li_comparison_summary.csv`, `results/jpl/phase6_li_comparison_summary.csv`).

CSR, 209 basins over 60 months, skill of ours over Li so positive means we are ahead
(`results/phase6_li_comparison_headline.csv`):

| pair | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| `kalman_ar1` vs `li_lstm_full` | +16.4% (p 2.2e-3) | -6.0% (0.23) | -16.3% (1.4e-3) | -23.2% | -28.9% | -35.4% |
| `ridge_own_era5_flat12` vs `li_lstm_nonseas` | +32.8% (2.4e-10) | +11.2% (2.1e-3) | -0.2% (0.94) | -10.1% | -18.0% | -26.4% |

JPL, 67 basins over 59 months, on the collaborator's run and reported Li-first, so a negative
skill means we are ahead (`results/jpl/phase6_li_comparison_headline.csv`): `li_lstm_nonseas`
against `kalman_ar1` scores -1.124 at lead 1 (p 1.3e-13), -0.286 at lead 2 (p 2.1e-5), -0.075 at
lead 3 (p 0.15, a tie), then +0.152 at lead 5 and +0.218 at lead 6. The crossover sits in the
same place on both products.

One number in the JPL tables must not be quoted as a statement about Li's accuracy. On
`all_matched`, `li_lstm_full` reads rmse_std 5.471 at lead 1 against 0.926 on
`joint_full_cells`, a factor of about 5.9, not the 8x the RUN_LOG entry's title still says
(`results/jpl/phase6_li_comparison_summary.csv`; the title is left as written because the log is
append-only). That is not a bug and not a leak. It is the fold standardization by our own
product's residual std on partially covered, low-variance basins; the diagnosis, with the three
negative leak checks, is in the 2026-09-10 RUN_LOG entry "Diagnosis: why Li's all_matched RMSE
inflates 8x on JPL and not at all on CSR". The remedy is the sample restriction, which is what
`joint_full_cells` is.

## What we tested and dropped

- **Neighbouring-basin information.** It does not replicate on JPL, and that is the reason it is
  not a claim. Every neighbour variant there is worse than own-basin at every lead. The placebo
  count is 0 of 50 for the correlation-selected neighbour at every lead and for every arm at
  leads 1 to 3; the distance-selected arm is the exception and beats at most 26 of 50, at lead 5
  (`kalman_geo_top1` beats 8, 26 and 9 of 50 at leads 4, 5 and 6, and `kalman_geo_top2` beats 1
  of 50 at lead 5). IAAFT surrogates are 0 of 99 at every lead
  (`results/jpl/phase3b_summary.csv`, `results/jpl/phase4_surrogate_summary.csv`). What it was on
  CSR: a neighbour's propagated state used as a correction, worth +0.31% at lead 1 over the
  own-basin ridge, beating 50 of 50 seed-matched placebo graphs and 99 of 99 IAAFT surrogates
  (`results/phase3b_summary.csv`, `results/phase4_surrogate_summary.csv`). Working
  interpretation: JPL's 3-degree mascons with the CRI filter already perform the spatial
  denoising that a CSR neighbour supplied. The experiments are retained as extended chain steps,
  not deleted.
- **A separate GRACE-FO observation-noise variance.** It hurts at every lead, -1.84% at lead 1
  (p 1.7e-7) through -0.38% at lead 6 (p 6.0e-3), and fold 1 cannot fit it at all, so 234 of
  1170 basin-folds fall back to one variance (`results/kalman_mission_summary.csv`). Per-basin
  at lead 1, BH-corrected: 0 helped, 0 hurt, 234 tested. The one-variance filter is kept.
- **Sequence models.** LSTM, residual MLP, graph network and the stacked combination all lose to
  the flat-12 ridge once history length is equalized, as under claim 2.

RUN_LOG pointers. The JPL non-replication is tabulated, per lead and per arm, in the entry "JPL
neighbor experiments: non-replication record (from results/jpl/, collaborator run
2026-09-05/06)". The CSR neighbour runs it is contrasted with are under the 2026-08-12 to
2026-08-17 entries, which is where phase 3b, the surrogates, the jump screen and their audit
passes were logged while the effect was still a live claim. The mission-split and sequence-model
decisions, and the `all_matched` diagnosis, are under the two 2026-09-10 entries
("Kalman-benchmark reframe" and the JPL all_matched diagnosis).

## How to run it

`scripts/run_chain.py` with no arguments runs the 12-step default list: `build_basin`,
`build_era5`, `build_li`, `phase2`, `kalman`, `flat12_ridge`, `kalman_mission`, `r0_ablation`,
`ladder`, `li_comparison`, `conventional_metrics`, `manifest`. It ends at `manifest`.
Everything else is extended, reached with `--extended`, which runs the extended list only and
not the default list first, or by naming steps with `--steps`; no script was deleted. The
default list needs no torch. `--list` prints the plan and marks extended steps.

`figures` is an extended step as of 2026-09-10, because it reads outputs only extended steps
produce. It declares sixteen inputs: the basin mask, `basin_meta.csv` and fourteen results
files. Two of the fourteen are default outputs, `paper_baseline_ladder.csv` and
`paper_baseline_contrasts.csv`. The other twelve are extended-only:
`phase8b_li_comparison_headline.csv`, `phase8b_li_comparison_perbasin.csv`,
`phase8b_h16_ensemble_headline.csv`, `phase8b_h16_headline.csv`, `phase8_stratification.csv`,
`phase3b_summary.csv`, `phase3b_placebo_monthly.csv`, `phase4_surrogate_summary.csv`,
`phase5_perbasin_fdr_h1.csv`, `phase6_era5_headline.csv`, `phase6_era5_predictions.csv` and
`phase4_conditioned_predictions.csv`.

`scripts/make_figures.py` additionally opens four undeclared files at runtime, all extended
outputs: `phase8_lstm_combined_predictions.csv`, `phase8b_lstm_h46_predictions.csv`,
`phase8_lstm_combined_placebo_monthly.csv` and `phase8b_lstm_h46_placebo_monthly.csv`, at around
lines 375 and 392. The dependency check cannot report those missing, so the script fails on the
open instead. The manuscript figures have not been rebuilt for the reframe; `figures` returns to
the default list when they are. `--extended --source jpl` skips it rather than failing on it.

Default-tail wall times on the 2026-09-10 CSR rerun: `build_li` 5.9 min, `flat12_ridge` 2.0 min,
`kalman_mission` 29 min, `ladder` 1.0 min, `li_comparison` 6.0 min, `conventional_metrics`
0.7 min, `manifest` 0.2 min. Extended steps run for hours to days.

`pytest tests/ -q` is 28 passing tests in about 30 seconds and needs no data.

## Data state

- CSR is the machine of record: everything directly under `results/` is a live CSR run.
- JPL: only the compact headline and summary CSVs are versioned, under `results/jpl/`.
  Row-level predictions, the JPL mascon file and Li's JPL-FCast archive exist only on the
  collaborator's machine. Provenance and checksums are in `results/jpl/RUN_PROVENANCE.md` and
  `results/jpl/SHA256_MANIFEST_LIVE.csv`.
- GSFC has not been run.

## Where things stand

The manuscript `paper/main.tex` has **not** been rewritten for this reframe. It still describes
the earlier three-finding structure with the neighbour effect as a contribution, and its figures
and its `paper/notes/REWRITE_LEDGER.md` numbers still match that structure. Do not read
`main.tex` as a description of the current claims. `docs/ARCHIVE_MANIFEST.md` carries the same
warning about its table and figure mapping.

## Open items

- Rewrite `paper/main.tex` to the three claims above, then rebuild the figures and regenerate
  the `docs/ARCHIVE_MANIFEST.md` mapping.
- `scripts/run_phase8b_merge.py` and `scripts/run_phase6_hybrid.py` are extended steps and still
  gate `joint_full_cells` on `source() == "jpl"`. The rest of the pipeline now applies the subset
  rule to both products; these two have not been updated. Anything read out of
  `phase8b_li_comparison_*` or `phase6_hybrid_*` on CSR is therefore not on the strict subset.
- `results/jpl/paper_baseline_ladder.csv` records `kalman_ar1` at +0.56% over damped persistence
  at lead 1 with p = 0.60, against +6.35% at lead 2 (p 2.7e-10). Claim 1's lead-1 margin does not
  replicate on JPL at that lead, and the reframed write-up has to say so.
- Zenodo DOI for the code and data availability statement.
