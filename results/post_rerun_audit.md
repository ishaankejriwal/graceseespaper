# Post-rerun audit — corrected foundational phases (2026-08-14)

Adversarial recomputation of every headline number in the 2026-08-13 (late) and 2026-08-14
RUN_LOG entries, from the prediction-level CSVs in `results\` using
`src/gracefc/stats.py` (pooled_monthly_dm / diebold_mariano / _paired_losses), plus code
inspection of the protocol sites. All recomputations run with `.venv\Scripts\python.exe`.
Audit scripts: scratchpad `audit_s12.py`, `audit_s235.py`, `audit_s6.py` (session scratchpad,
not part of the repo).

Overall verdict: **PASS WITH NOTES.** Every recomputed number matches the claimed value to
the stated precision. No protocol violations found in the rerun artifacts. Several
cache-hygiene footguns remain live and should be closed before the torch phases (ranked
list at the bottom).

---

## Section 1 — Baseline ladder & contrasts: **PASS**

Recomputed independently from `results\phase2_baseline_predictions.csv` +
`results\phase3b_predictions.csv` (matched-row intersection rebuilt from scratch, not via
`build_paper_ladder.py`).

**Row counts.** Both files have exactly the claimed n per horizon, for every model
(7 phase2 models, 11 phase3b models):

| h | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| n (both files, every model) | 19422 | 19188 | 18954 | 18720 | 18486 | 18252 |

The matched-row intersection drops **0 keys** at every horizon — phase2 and phase3b are
already on identical (name, issue_date) row sets, and targets agree across all 10 models
to 0.0 (exact). Issue-date consistency: `target_date == issue_date + h months` on all rows,
in both files; issue windows per fold exactly match `DEFAULT_FOLDS`
(src/gracefc/evaluate.py:25-31); zero rows outside declared windows; zero duplicate
(model, name, target_date, horizon) and (model, name, issue_date, horizon) keys — no
fold-boundary double counting under the new overhang semantics.

**kalman_ar1 vs stronger damped variant** (recomputed / claimed, skill %):

| h | recomputed | claimed | DM p (recomputed) |
|---|---|---|---|
| 1 | +4.979 | +4.98 | 3.54e-12 |
| 2 | +8.791 | +8.79 | 6.34e-19 |
| 3 | +5.625 | +5.62 | 5.66e-19 |
| 4 | +3.070 | +3.07 | 2.57e-11 |
| 5 | +2.549 | +2.55 | 7.87e-07 |
| 6 | +3.633 | +3.63 | 1.99e-09 |

Damped reference per horizon (rho at h1, reg at h2-6) matches the stored ladder.

**kalman_ar1 vs ridge_own_perbasin** (recomputed / claimed):

| h | skill % | claimed | DM p | claimed p |
|---|---|---|---|---|
| 1 | +1.986 | +1.99 | 6.6e-4 | 6.6e-4 |
| 2 | +3.036 | +3.04 | 7.7e-7 | 7.7e-7 |
| 3 | +1.427 | +1.43 | 0.0131 | 0.013 |
| 4 | +1.037 | +1.04 | **0.0783** | 0.078 |
| 5 | +2.175 | +2.17 | 0.0051 | 0.0051 |
| 6 | +3.916 | +3.92 | 8.3e-6 | 8.3e-6 |

"Significant at 5 of 6 leads, h4 p=0.078" — **confirmed.** Also confirmed:
kalman_ar1 vs ridge_own_lags significant only at h1-h2 (+3.86% p=9.1e-6, +4.21% p=1.2e-11),
h3-h6 ns with h4/h5 nominally negative (-0.59%, -0.64%) — the RUN_LOG's correction of the
pre-audit pooled-ridge claim is right.

Notes:
- `build_paper_ladder.py:3-9` docstring is stale (describes the old 19,656-row files and
  "phase2 shrinks with horizon"); the current inputs are already row-aligned. Cosmetic.
- At h4, the moving-block-bootstrap CI on skill (+0.0004..+0.0218) excludes zero while the
  DM p is 0.078. Not a bug (different statistics), but the paper must quote the DM test as
  the significance criterion and not let the CI imply significance at h4.

## Section 2 — Neighbor null, placebos, resolution stratification: **PASS**

**kalman_corr_top1 vs kalman_own_ridge** via `pooled_monthly_dm` on
`phase3b_predictions.csv` (recomputed / claimed):

| h | skill % | claimed | DM p |
|---|---|---|---|
| 1 | +0.3076 | +0.308 | 0.1986 (claimed 0.199) |
| 2 | -0.0971 | -0.097 | 0.718 |
| 3 | -0.1915 | -0.192 | 0.481 |
| 4 | -0.3580 | -0.358 | 0.116 |
| 5 | -0.3488 | -0.349 | 0.070 |
| 6 | -0.3208 | -0.321 | 0.157 |

h2-h6 all negative, all ns — confirmed. (h5 at p=0.070 is nominally the closest to
significance *against* the neighbor; consistent with the log's "doubtful for phase-8" read.)

**Placebo ranks**, recomputed from `phase3b_placebo_monthly.csv` pooled MSE vs the real
arm, cross-checked against `phase3b_summary.csv`:
- kalman_corr_top1: h1 **50/50 beaten**, p_rank 0.0196; h2-h6 **0/50**, p_rank 1.0 — matches claim.
- kalman_corr_min300_top1: h1 50/50 (p_rank 0.0196) — matches "min300 also 50/50";
  h2 is 2/50 (p_rank 0.961), h3-6 0/50. The log's "0/50 at h2-6" is exactly true for
  corr_top1 only; if the min300 variant is quoted at h2-6, say "≤2/50".

**Resolution stratification at h1** (kalman_corr_top1 vs kalman_own_ridge, split on
`basin_meta.csv` `below_resolution`):
- resolved (≥90k km², 199 basins): **+0.099%, p=0.715** (claimed +0.099/0.715)
- sub-resolution (<90k km², 35 basins): **+1.033%, p=0.00065** (claimed +1.033/0.00065)
- min300 variant: resolved +0.140% p=0.608, sub-res +0.964% p=0.00087
  (claimed +0.140/p=.61 and +0.964/p=.00087) — all confirmed; the effect is entirely
  carried by sub-resolution basins.

## Section 3 — Li crossing: **PASS**

Recomputed from `phase6_li_comparison_predictions.csv`; agrees with the stored
`phase6_li_comparison_headline.csv` and the claims:

**li_lstm_full vs kalman_ar1** (positive = Li worse... sign convention: negative skill =
Li worse than us; DM sign consistent):

| h | skill % | claimed | DM p |
|---|---|---|---|
| 1 | **-17.13** | -17.1 | **0.0039** |
| 2 | +6.26 | +6.3 | 0.154 (ns) |
| 3 | +14.01 | +14.0 | 8.6e-4 |
| 4 | +18.48 | +18.5 | 3.4e-5 |
| 5 | +22.02 | +22.0 | 1.8e-6 |
| 6 | +25.27 | +25.3 | 4.9e-8 |

Crossing between leads 2 and 3 — confirmed. Also confirmed: li_lstm_full vs
damped_persistence_rho h1 = -11.09%, p=0.057; li_lstm_nonseas vs kalman_ar1 h1 = -35.7%
(p=2.5e-10); kalman_ar1 vs damped on the matched sample = +5.16/+9.07/+10.80/+12.86/
+12.41/+11.87% (log said +5.2/+9.1/+10.8/+12.9/+12.4/+11.9).

**Matched sample:** exactly 227 basins × 60 months = 13,620 rows per model at *every*
horizon; zero duplicate (model, name, target_date) keys per horizon.

**Offset estimation is train-only:** `run_phase6_li_comparison.py:71` sets
`is_train = target_date < fold.test_start`; the offset (line 80) is the mean of
(pred - obs) over train-flagged, finite rows only; test membership (lines 72-73) is by
issue date, matching `evaluate.split_fold`. The climatology and train_std used to place Li
in our space (lines 59-61) are fit on `wide[wide.index < fold.test_start]`. No test
information enters. The target-consistency guard vs phase3b (lines 116-120, tol 1e-6) is
active and passed in `rerun_li.log`.

## Section 4 — Protocol correctness in code: **PASS WITH NOTES**

Full-repo grep (src, scripts, notebooks) for target-date-based test membership: the only
remaining `target_date`-vs-`test_start` comparisons are **train-side** (legitimate):
`evaluate.py:53,68`, `run_phase6_li_comparison.py:71`. All engines and all five previously
inline-splitting runners now clip test rows by issue date
(`run_kalman_baseline.py:49-50`, `run_phase5_coupled.py:41`, etc.). Nothing reads
`archive\` paths. Verified empirically: every rerun output file (phase2, phase3b, era5,
li, kalman) has fold issue-windows exactly matching DEFAULT_FOLDS and zero out-of-window
or duplicated rows.

Live footguns (none currently corrupting results, all verified clean *today*):

1. **`run_kalman_baseline.py:30-34` resume shortcut is still present** — reuses
   `results\kalman_predictions.csv` on mere existence. This exact shortcut served a stale
   protocol file once already this session. The current file is verified bit-identical to
   phase3b's kalman_ar1 rows (113,022 rows joined, max |target diff| = max |pred diff| = 0),
   so today's numbers are fine, but the guard is only the downstream stats.py assertion.
2. **`results\kalman_fold_params.pkl` is keyed by fold name only**
   (`experiment_kalman.py:49-54`; same `PARAMS_CACHE` read blindly by
   `run_phase3b_kalman_neighbors.py:53`, `run_phase5_coupled.py:57`,
   `run_phase5_fusion.py:63`, `run_phase5_nonlinear.py:32`, `run_phase6_era5.py:50`,
   `run_phase7_gnn.py:47`, `run_phase7_lstm.py:47`, `run_phase7_resmlp.py:47`,
   `run_phase8_lstm_combined.py:51`). No content hash: if the basin table or
   deseasonalization ever changes again, every one of these phases silently reuses stale
   MLE params. I verified the current pkl is fresh by refitting fold f1 params from the
   corrected table for 6 basins — exact match to all pkl digits (including an
   r-at-boundary basin at r=1.0e-10).
3. **`run_phase8b_merge.py:65-76`** reads phase7/phase8 prediction CSVs from `results\`.
   Those were archived, so a premature phase8b run fails loudly (FileNotFoundError) rather
   than silently mixing protocols — acceptable, but it hard-orders the reruns:
   phase7 and phase8 must complete before 8b. It also reads
   `phase6_li_comparison_predictions.csv` (fresh, verified).
4. **`notebooks\02_baselines_and_kalman.ipynb`** reads `results\kalman_predictions.csv`
   and `results\phase2_baseline_summary.csv` and compares RMSEs across files. On today's
   row-aligned files that comparison is no longer wrong, but the notebook's stored outputs
   are pre-audit. Re-execute or banner it as stale before anyone screenshots it.
5. Phase 4 / phase 5 (surrogates, fusion, coupled) code is fixed but **not rerun**; their
   old outputs are in archive only. `results\phase5_analysis.md`, `phase5_audit.md`,
   `phase7_analysis.md`, `phase8_*.md`, `phase6_analysis.md` still describe pre-audit
   numbers — do not quote them until their phases rerun.

## Section 5 — ERA5 claims: **PASS**

Recomputed from `phase6_era5_predictions.csv` (h1-h3 only, 57,564 rows per model =
19422+19188+18954; issue windows, key uniqueness, and target=issue+h all verified):

- **ridge_own_era5 vs ridge_own**: h1 +4.621% p=9.74e-5 (claimed +4.62, p=1e-4);
  h2 +1.391% p=0.0920 (claimed +1.39, 0.092); h3 +0.331% p=0.608 (claimed +0.33, ns). ✓
- **No-ERA5 arms reproduce phase3b**: ridge_corr_top1 vs ridge_own = +0.3076/-0.0971/
  -0.1915%, DM p 0.1986/0.7179/0.4805 — identical to phase3b's kalman_corr_top1 vs
  kalman_own_ridge to all recomputed digits. ✓ **Caveat on interpretation:** the underlying
  predictions are bit-for-bit identical (57,564 joined rows, max |pred diff| = 0) because
  both runners share `kalman_fold_params.pkl` and the same engine path. This is a
  determinism/regression check, not an independent replication — the RUN_LOG's "two
  independent runners, same numbers" overstates it slightly.
- **Neighbor under ERA5 conditioning**: ridge_corr_top1_era5 vs ridge_own_era5 h1
  +0.322% p=0.115 (claimed +0.322/0.115). ✓
- **gbm/mlp vs ridge on identical features**: gbm_own_era5 +0.144 ns / +0.596 ns /
  -0.713% p=0.0366; mlp_own_era5_s0 -0.534 ns / -2.558% p=3.1e-5 / -3.145% p=7.2e-5. ✓
  The gbm_corr_top1_era5 vs ridge_corr_top1_era5 h2 anomaly reproduces (+1.115%, p=0.0185)
  and is correctly flagged in the log as an unreplicated single cell.

## Section 6 — Rebuilt basin table: **PASS**

`data\processed\basin_month_twsa_global.csv`:
- **257 unique months, 2002-04 .. 2026-05**; 2011-11 and 2015-05 present; zero duplicate
  (name, date) rows; 284 basins × 257 months = 72,988 rows, complete grid. ✓
- Table months exactly equal `assign_solution_months()` output from the raw NetCDF. ✓
- **Old-vs-new convention diff, computed from the raw file:** midpoint binning and the
  official months_missing sequence disagree on exactly **2 of 257 solutions** —
  solution 110 (arc mid 2011-10-31: old→2011-10, new→2011-11) and solution 144
  (mid 2015-04-27: old→2015-04, new→2015-05). Under the old convention exactly two months
  (2011-10, 2015-04) held two solutions each (which were averaged). Therefore only
  2011-10/2011-11 and 2015-04/2015-05 differ from the pre-audit table; every other month
  maps to the same single solution with identical aggregation. This is a proof by
  construction, stronger than spot-checking against the (overwritten) old table.
- Aggregation spot-check from raw for 3 months × 3 basins (2005-06, 2019-01, 2023-03 ×
  Amazon Delta, Aleutians, N. Gulf of Oman): max abs diff 6.0e-4 cm on values up to
  115 cm (float32 summation-order noise). ✓
- **Kalman diagnostics** (`results\kalman_fold_params.pkl`): 1170/1170 fits converged;
  259 fits at r<1e-6; 31 basins at the boundary in all five folds — matches the log's
  1170/1170, 259/1170, 31 exactly.

---

## Ranked fixes before rerunning torch phases (7 / 8 / 8b)

1. **Delete or guard the `run_kalman_baseline.py` resume shortcut (lines 30-34).** At
   minimum, assert the cached file's per-horizon row counts and fold issue-windows match
   DEFAULT_FOLDS before reuse. It has already served a stale-protocol file once; the only
   thing that caught it was a downstream assertion.
2. **Content-address `kalman_fold_params.pkl`.** Key the cache by (fold name + hash of the
   fold's training residual matrix or of `basin_month_twsa_global.csv`), or have
   `build_basin_series.py` delete the pkl on rebuild. Phases 7/8 read this pkl blindly at
   `run_phase7_lstm.py:47` / `run_phase7_gnn.py:47` / `run_phase7_resmlp.py:47` /
   `run_phase8_lstm_combined.py:51`; today it is verified fresh, but nothing enforces that.
3. **Rerun order and stale-doc quarantine:** phase4 surrogates and phase5 fusion/coupled
   must be rerun before any of their numbers are cited (code fixed, outputs archived);
   phase8b requires fresh phase7+phase8 outputs (it will crash, not silently mix — but
   plan the order). Mark `phase5_analysis.md`, `phase6_analysis.md`, `phase7_analysis.md`,
   `phase8_*.md` as pre-audit until regenerated.
4. **Manuscript wording nits confirmed by this audit:** h4 vs per-basin ridge must be
   quoted from the DM test (p=0.078, ns) even though the bootstrap CI excludes zero;
   "two independent runners" for the phase6/phase3b consistency check should be softened
   to a determinism check (predictions are bit-identical via the shared params cache);
   min300 placebo at h2 is 2/50, not 0/50, if that variant is quoted beyond h1.
5. **Cosmetic:** stale docstring in `build_paper_ladder.py:3-9` (describes pre-fix row
   counts); re-execute or banner `notebooks\02_baselines_and_kalman.ipynb`.

No recomputed number disagreed with the RUN_LOG claims. The corrected foundation is sound
to build the torch phases on, once items 1-2 are closed.
