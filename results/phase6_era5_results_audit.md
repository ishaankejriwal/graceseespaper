# Phase 6 ERA5 Results Audit — 2026-08-12 (post-winsorization rerun)

Adversarial audit of the phase6_era5 RESULTS (CSVs timestamped 15:17, i.e. the rerun after
the ±10σ winsorization fix) and the corresponding RUN_LOG.md entry. Every numeric claim was
recomputed from the raw `phase6_era5_predictions.csv` / `phase6_era5_placebo_monthly.csv`
rows with independent pandas/scipy code (own implementations of pooled RMSE, skill,
Harvey-corrected DM with Bartlett HAC, moving-block bootstrap, placebo rank — no gracefc
imports for checks 1–3/5/6). The winsorization fix (check 4) was audited by reading
`era5_fold_features` and by recomputing the ERA5 features for all five folds with the
project code and measuring clip rates directly. This supersedes nothing in the earlier
CODE audit (`phase6_era5_audit.md`); it extends it to the rerun outputs and the one code
change that postdates it.

Verdict up front: **every headline number in RUN_LOG.md and the CSVs reproduces
digit-for-digit; the row sets are airtight; the winsorization is causal, tiny in footprint
(≤0.06% of feature values), and does exactly what the log says it does. Three low-impact
wording issues in the RUN_LOG entry should be tightened before the write-up (details in
Discrepancies): the "p≤4e-5" bound is off by 8% at one cell, the "three-way tie" quotes the
best MLP seed without saying so, and the Libya mechanism sentence names the wrong folds'
test windows.**

---

## CHECK 1 — Pooled RMSE, n, and row-set identity: PASS

- All 63 (model, horizon) cells (21 models × h1–h3) recomputed from raw prediction rows:
  max |rmse_summary − rmse_recomputed| = **2.2e-16** (float noise; digit-for-digit).
  `skill_vs_ridge_own` recomputed as 1 − (rmse/rmse_own)²: max diff **5.3e-16**.
- **n = 19,656 in every cell** (= 234 basins × 84 test months). File decomposes exactly:
  1,238,328 data rows = 63 × 19,656.
- Row-set identity across arms: within each horizon, the md5 hash of the sorted
  (name, issue_date, target_date) key list is **identical for all 21 models** (one distinct
  hash per horizon). Targets are identical across models per key (nunique ≡ 1).
- Fold composition identical for every model: f1–f4 = 3,978 rows each (17 months × 234),
  f5 = 3,744 (16 months × 234). Every (model, horizon, target_date) cell has exactly 234 rows.

## CHECK 2 — RUN_LOG headline claims: PASS (all reproduce exactly)

Independent DM (Harvey-corrected, HAC max_lag = max(h−1,1)) and block bootstrap
(2000 draws, block 3, seed 0) on cross-basin monthly mean losses, 84 months per contrast:

| claim (RUN_LOG) | recomputed | headline.csv |
|---|---|---|
| ridge_own_era5 vs ridge_own h1 +5.17%, CI +2.94..+7.18, p<1e-5 | **+5.1658%**, CI (+2.941, +7.175), p=4.332e-07 | identical |
| … h2 +1.81%, p=.0099 | **+1.8100%**, CI (+0.265, +3.311), p=9.948e-03 | identical |
| … h3 +0.65% ns | **+0.6539%**, CI (−0.461, +1.771), p=.2624 | identical |
| ridge_corr_top1_era5 vs ridge_own_era5 h1 +0.485%, CI +0.15..+0.86, p=.0096 | **+0.4849%**, CI (+0.152, +0.858), p=9.632e-03 | identical |
| "essentially identical to the unconditioned +0.488%" | ridge_corr_top1 vs ridge_own h1 = **+0.4879%**, p=.0218 | identical |
| GBM ties ridge at h1 (p=.86), h3 (p=.91); suggestive h2 (−1.87, p=.065) | gbm_corr_top1_era5 vs ridge_corr_top1_era5: h1 DM=−0.173 p=**.8629**; h2 DM=−1.867 p=**.0654**; h3 DM=+0.109 p=**.9134** | identical |
| "~10× the neighbor effect" | 5.166 / 0.485 = **10.6×** | — |
| MLP: 2/3 seeds tie at h1, all worse at h3 | s0 p=.885, s1 p=.905 (ties), s2 p=.0083 (worse); h3 all three DM>0, p ≤ 1.6e-05 | confirmed |

My DM/bootstrap implementations agree with `stats.py pooled_monthly_dm` /
`block_bootstrap_skill_ci` outputs (headline.csv) to all printed digits, so the library and
the numbers cross-validate.

## CHECK 3 — Placebo accounting: PASS

- Exactly **20 seeds (rand0..rand19) per family**, four families as designed:
  ridge_corr_top1_era5, gbm_corr_top1_era5, mlp_corr_top1_era5, ridge_corr_top1 (the
  no-ERA5 re-anchor). Placebo file = 20,160 rows = 4 × 20 × 3 horizons × 84 months.
- Monthly `count` ≡ 234 in every row; pooled counts = 19,656 per (placebo, horizon) —
  placebo arms sit on the identical row set as the real arms. No negative loss sums.
- Pooled placebo RMSEs recomputed from sum/count; every `placebo_n`, `placebo_beaten`,
  `p_rank` in summary.csv reproduced exactly. Real arms at 20/20 give
  p_rank = (1+0)/(1+20) = **1/21 = .047619** — the claimed 20-seed floor. Confirmed at the
  floor for: ridge_corr_top1_era5 h1/h2/h3, gbm_corr_top1_era5 h1/h2/h3,
  ridge_corr_top1 h1/h2/h3, mlp_corr_top1_era5_s1 h1. (s0 19/20 at h1, s2 4/20 —
  matches summary and the seed-instability narrative.)
- No placebo detonation: e.g. mlp placebo pooled RMSE range at h1 is [1.0156, 1.0264] —
  the winsorization protects placebo arms identically (they share the era5 feature block).

## CHECK 4 — Winsorization fix: PASS, with two design notes

Code (`era5.py`, `era5_fold_features`): standardized residual `r/std` uses a **train-only**
std (`r[r.index < test_start].std()`), then `.clip(-10, 10)` with **constant** CLIP_SIGMA=10
applied to the full series **before lagging** — one transform, same bound, train and test
alike. No test statistic enters; the clip is causal. The zero guard (`std > 1e-8` else
`r*0.0`) also uses only the train std.

Empirical recompute (project code, all five folds, 234 basins × 33 feature columns):

| fold | test_start | clipped values (of 2,347,488) | train-window frac | test-window frac |
|---|---|---|---|---|
| f1 | 2019-06 | 1,496 (6.4e-04) | 8.7e-05 | 7.8e-04 |
| f2 | 2020-11 | 1,250 (5.3e-04) | 1.0e-04 | 2.1e-04 |
| f3 | 2022-04 | 1,166 (5.0e-04) | 9.9e-05 | 3.4e-03 |
| f4 | 2023-09 | 470 (2.0e-04) | 1.0e-04 | 1.1e-03 |
| f5 | 2025-02 | 287 (1.2e-04) | 1.2e-04 | 2.3e-04 |

- Footprint is tiny (≤0.064% of all values; ≤0.34% of any test window) and the clip is
  demonstrably active on TRAIN rows too (~1e-04 in every fold) — identical treatment, as
  claimed.
- **Libya story verified**: E_Libyian_Desert `subsurf_runoff_cm` train std is 4.4–5.0e-08
  under f1–f4 stats; unclipped max |σ| = **45,265 (f4) to 51,924 (f3)** — the "~50,000σ"
  in RUN_LOG is right. Under f5 (whose train window contains the flood) std jumps to
  1.4e-04 and max |σ| falls to 15.9. In f4, 23 months of that one series clip; the top
  f4 clippers are exactly the near-zero-runoff deserts (E_East_Gobi_Desert 101σ,
  E_Upper_Lake_Chad 53σ, E_Libyian_Desert runoff/surface-runoff ~45–50σ).
- **Guard interaction — no pathology.** The 58 zero-guarded series in f4 are numerically
  dead: tropical SWE constants of ~1.7e-25 std (raw values ±7e-23 — float dust) and
  E_Egyptian_Western_Desert ssro which is exactly 0.0 in train AND test — zeroing discards
  no information. The dangerous series (Libya, std 5.03e-08) sits 5× ABOVE the 1e-8 guard,
  so it is clipped rather than zeroed — the fix, not the guard, is what handles it.
- Emitted features contain **0 NaNs** in any fold's test window, consistent with no rows
  being dropped by the ERA5 join (check 5 proves n is unchanged from earlier phases).

Design notes (disclose, not bugs):
1. The 1e-8 guard threshold is in physical units (cm), so it is unit-dependent, and the
   Libya std (5e-8) is only 5× above it — a future variable rescale could silently flip a
   series from "clipped event flag" to "zeroed". 28 series in f4 live in the near-guard
   band (1e-8 < std < 1e-4); for these the "standardized anomaly" is quantization noise of
   near-zero ERA5-Land output and, after clipping, the feature behaves as a bounded ±10
   event indicator rather than a Gaussian-ish anomaly. Causal and harmless, but the paper
   should describe the winsorization as also converting degenerate desert-runoff series
   into bounded event flags.
2. Series std just below vs just above 1e-8 get qualitatively different treatment
   (zeroed vs ±10 spikes). Cosmetic discontinuity; both branches are causal.

## CHECK 5 — Backbone identity with earlier phases: PASS (with one task-brief correction)

- `phase5_nonlinear_predictions.csv` contains **no kalman_ar1 rows** (its 12 models are the
  ridge/gbm/mlp arms only), so the comparison named in the brief is impossible as stated.
  Instead, kalman_ar1 (h1–h3, 58,968 rows) was compared against **phase3b, phase5_coupled,
  phase5_predlag, and phase5_fusion**: outer-join on (name, issue_date, target_date,
  horizon) is exactly 1:1 in all four (matched = 58,968, zero unmatched on either side),
  and **max |pred diff| = max |target diff| = 0.0** — bitwise identical, row sets exact.
- Additionally, the no-ERA5 twins that DO exist in phase5_nonlinear — ridge_own,
  ridge_corr_top1, gbm_own, gbm_corr_top1, mlp_own_s0, mlp_corr_top1_s1 — are **bitwise
  identical** to their phase6 counterparts (max |pred diff| = 0.0, 58,968/58,968 matched).
  This simultaneously proves (a) the ERA5 join dropped zero rows (n stays 19,656 = phase5's
  count), (b) the winsorization rerun did not perturb any no-ERA5 arm, and (c) the twin
  comparisons in the summary are apples-to-apples.

## CHECK 6 — Anomaly sweep: PASS

- 0 NaN predictions, 0 NaN targets, 0 duplicate (name, issue_date, target_date, model,
  horizon) keys, 0 negative placebo loss sums.
- No detonation residue: max |pred| over all 1.24M rows is **7.01** (kalman_ar1);
  max |target| 10.69; zero rows with |pred| > 15. The pre-fix pathology (pooled RMSE 12–26)
  is gone from every arm including placebos.
- Fold balance exact (see check 1). Placebo seed base = crc32("era5_corr_top1") mod 1e6 =
  **287,374** — distinct from all six earlier experiment bases (352208, 86661, 876756,
  242860, 185052, 800102): no shared placebo draws across experiments.
- MLP seed spread: pooled-RMSE range across seeds is 0.0007–0.0098 per (arm, horizon); the
  largest (0.0098) is mlp_corr_top1_era5 at h1, where s2 (1.0249) sits 4/20 against
  placebos while s0/s1 sit 19–20/20 — this is the seed instability the log reports, and it
  is material to how the h1 "tie" is quoted (Discrepancy 2).

---

## DISCREPANCIES (all LOW, none touches a conclusion)

1. **RUN_LOG: GBM "significantly worse (no-ERA5 twins, DM p≤4e-5)".** The max twin p among
   the four GBM no-ERA5 cells at h1–h3 is gbm_corr_top1 vs ridge_corr_top1 at h1:
   **p = 4.304e-5 > 4e-5**. Same class of nit the phase5 audit flagged; write "p ≤ 4.4e-5"
   or "p < 5e-5".
2. **RUN_LOG: "gbm/ridge/mlp corr_top1_era5 in a three-way tie at RMSE ≈1.015, +5.7% over
   ridge_own."** True only with the best MLP seed: s1 = 1.01503 but s0 = 1.01586 and
   s2 = 1.02487 (s2 loses to 16/20 placebos). And ridge's skill is +5.63%, not +5.7%
   (gbm/mlp_s1 are +5.70%). Quote as "gbm and ridge tie at 1.015 (+5.7%/+5.6%); the best
   MLP seed joins them, the worst does not" — otherwise the tie contradicts the
   seed-instability sentence two lines above it.
3. **RUN_LOG mechanism sentence: "the 2023/24 Libya flood months standardized to ~50,000σ
   in f2–f4 test windows."** The ~50,000σ magnitudes arise under f1–f4 TRAIN-window stats
   (verified: 45k–52k), but the flood months (2023-09 onward; peak unclipped σ at 2024-08)
   fall **only inside f4's test window** (2023-09..2025-01). f2/f3 test windows end 2022-03
   and 2023-08, so those folds never touch the flood rows at train or test time; f5 trains
   through the flood and its σ collapses to ~16. The voided first run's explosion therefore
   came from f4 (and its 12–26 pooled RMSE cannot be re-verified — the file was
   overwritten). Fix the sentence to "under f1–f4 train-window statistics; the affected
   months sit in f4's test window."

No errors found in the CSVs themselves.

---

## SUMMARY TABLE

| # | Check | Verdict | Key recomputed numbers |
|---|---|---|---|
| 1 | Pooled RMSE / n / row sets | **PASS** | max RMSE diff 2.2e-16; n≡19,656; 1 key-set hash per horizon across 21 models |
| 2 | RUN_LOG headline claims | **PASS** | +5.1658/+1.8100/+0.6539% (h1/h2/h3); +0.4849% p=.0096; GBM tie p=.863/.065/.913 — all exact |
| 3 | Placebo accounting | **PASS** | 4 families × 20 seeds × 3 horizons; counts ≡234/19,656; p_rank floor 1/21=.047619 reproduced everywhere |
| 4 | Winsorization fix | **PASS** (2 design notes) | causal constant clip; ≤0.064% values clipped; Libya 45k–52k σ unclipped, guard NOT triggered (std 5e-8 > 1e-8) |
| 5 | Backbone identity | **PASS** | kalman_ar1 bitwise = phase3b/coupled/predlag/fusion; all 6 no-ERA5 twins bitwise = phase5_nonlinear; 0 row-set drift |
| 6 | Anomaly sweep | **PASS** | 0 NaN/dup/negative; max abs pred 7.01; seed base 287374 distinct; MLP s2 h1 outlier is real and disclosed |

## WRITE-UP DISCLOSURES (carry into the paper)

1. Winsorization: state it as "standardized ERA5 anomalies winsorized at ±10σ (a constant,
   train-window-only transform affecting <0.07% of feature values); for a handful of
   degenerate desert-runoff/tropical-SWE series this converts quantization-scale anomalies
   into bounded event indicators." (Check 4 notes 1–2.)
2. The first full run was voided and overwritten after the Libya blowup; only the winsorized
   rerun exists on disk, so the pre-fix RMSEs (12–26) are quoted from the log, not
   reproducible. Say so if the episode is narrated.
3. Rewrite the three RUN_LOG sentences per Discrepancies 1–3 (p-bound, MLP-seed-conditional
   tie, Libya fold attribution).
4. Placebo floor: 20/20 gives p_rank=.0476 and is hit at 12 correlated cells — report it as
   the attainable floor with 20 seeds, per the phase5 audit's precedent.
5. kalman_ar1 does not exist in phase5_nonlinear_predictions.csv; backbone-identity claims
   should cite phase3b (or coupled/predlag/fusion), where it is bitwise identical.
