# Phase 8 Analysis — Stacked LSTM + Neighbor-Only Residual MLP (2026-08-13)

Interpretation pass over `phase8_lstm_combined_*` (stage 1 = phase 7 shared-encoder LSTM on
12-month windows of Kalman-filtered own state + 11 ERA5 channels; stage 2 = sklearn MLP fit on
the stage-1 TRAIN residual from the propagated corr_top1 neighbor state only; final = kalman +
lstm + mlp). All numbers computed fresh from the predictions/placebo CSVs (scripts
`p8_analysis.py`, `p8_followup.py` in the session scratchpad; derived tables saved there as
`p8_headline_computed.csv`, `p8_folds.csv`, `p8_placebo_increments.csv`, `p8_perbasin_fdr.csv`,
`p8_perbasin_fdr_ens_h{1,2,3}.csv`, `p7_resmlp_perbasin_h{1,2,3}.csv`, `p8_nbrin_contrasts.csv`,
`p8_ladder.csv`, `p8_followup_contrasts.csv`). Audit constraints applied throughout: the
summary's `dm_vs_ridge_twin` is never quoted for lstmres arms (information-mismatched); the
correction is described as "honest evaluation, plausibly attenuated correction", never
"provably conservative"; placebo ranks are graph-specificity evidence, not proof of
information transfer (≥300 km and conditioning controls have not been run for this arm).

---

## 1. Data integrity — clean

- **Zero NaN** predictions and targets; every one of the 13 arms has exactly **19,656 rows per
  horizon**, and per horizon all arms share literally **one** (name, issue_date, target_date)
  row-set hash — DM/bootstrap comparisons are exactly paired.
- **Bit-identity to phase 7:** all 9 arms shared with `phase7_lstm_predictions.csv`
  (kalman_ar1, 4 ridges, lstm_own_era5_s{0,1}, lstm_corr_top1_era5_s{0,1}) merge 530,712/530,712
  rows with max |pred diff| = max |target diff| = 0.0; the 5 reference arms shared with
  `phase7_resmlp_predictions.csv` likewise (294,840 rows, 0.0). Every cross-phase comparison
  below is therefore exact, not approximate.
- Summary-file RMSEs reproduce from raw rows to 2e-16. The run log's headline numbers all
  verify; one nit: it says "all DM p<1e-6" for the per-seed headline — s1 h2 is p = 2.0e-6.

## 2. Headline: the stage-2 neighbor correction is real, seed-stable, and fold-stable

`lstmres_corr_top1` vs `lstm_own_era5` (identical stage-1 net within seed — the contrast
isolates the correction; ensemble = per-row mean of the two seeds on both sides):

| h | s0 | s1 | 2-seed ensemble (CI) | DM p (ens) |
|---|---|---|---|---|
| 1 | +0.90 | +0.90 | **+0.93** (+0.70..+1.18) | 1.0e-13 |
| 2 | +1.12 | +0.99 | **+1.09** (+0.80..+1.40) | 4.1e-11 |
| 3 | +1.28 | +1.20 | **+1.27** (+0.94..+1.62) | 6.5e-9 |

The seed agreement is remarkable: at h1 the increment is +0.9029 (s0) vs +0.9046 (s1) even
though the two stage-1 nets differ by ~0.9% skill. **This is the first neighbor-under-ERA5
result in the project that is seed-stable** (phase 7's within-LSTM and resMLP-ERA5 neighbor
increments were sign-flipping across seeds).

Ladder contrasts (ensemble): vs `ridge_corr_top1_era5` +2.40/+2.95/+2.62% (p ≤ 3.0e-10); vs
`kalman_ar1` +7.72/+3.95/+3.67%; vs `ridge_own` +7.89/+4.81/+3.32% (all p ≤ 1.1e-7).

**Per-fold: 45/45 cells positive.** Skill vs the same-seed stage-1 is positive in every
fold × horizon × {s0, s1, ens} cell (weakest: s1 f4 h1 +0.22, s0 f3 h1 +0.25; table
`p8_folds.csv`). No fold drives the result. The weakest folds at h1-h2 are f3/f4 (the 2022-23
La Niña windows, the known weak ERA5 folds), but even there the correction stays positive.

## 3. The key placebo question: seed-matched increments (task 3)

The summary's placebo ranks compare each real arm's raw pooled RMSE against 20 placebo arms
that ride the **seed-0** stage-1 net. That is seed-matched for s0 (20/20 beaten at all
horizons) but mechanically handicaps s1, whose stage-1 is ~0.9-1.1% worse than seed-0's.
The seed-matched quantity is the **increment**: skill of the stacked arm relative to its own
stage-1 (for placebos, relative to `lstm_own_era5_s0`, which they ride). Computed from
`phase8_lstm_combined_placebo_monthly.csv` (`p8_placebo_increments.csv`):

| family | h | placebo increment mean [min, max] | real s0 | real s1 | beats all 20? |
|---|---|---|---|---|---|
| lstmres_corr_top1 | 1 | −0.01% [−0.24, +0.21] | **+0.90** | **+0.90** | yes / yes |
| lstmres_corr_top1 | 2 | −0.12% [−0.40, +0.03] | **+1.12** | **+0.99** | yes / yes |
| lstmres_corr_top1 | 3 | −0.12% [−0.39, +0.10] | **+1.28** | **+1.20** | yes / yes |

- **A junk-neighbor MLP correction does nothing** (mean increment ≈ 0 to −0.1%, best single
  draw +0.21%): the two-stage architecture itself confers no skill, exactly mirroring phase
  7's finding that the placebo resMLP family pools to the `ridge_own` level. All of the gain
  requires the real corr_top1 neighbor.
- **Seed-matched, both seeds beat all 20 placebo increments at every horizon** — the real
  increments are 4-9x the best placebo draw. The proper statement is 20/20 at all horizons
  for BOTH seeds, not the summary's 6/20-12/20 for s1.
- **The s1 raw ranks are fully explained by the stage-1 asymmetry.** Comparing s1's raw RMSE
  to the seed-0-riding placebo distribution reproduces the summary's 6/20 (h1), 12/20 (h2),
  20/20 (h3) exactly. The stage-1 seed gap (s0 better than s1 by 0.93/1.06/0.67% skill at
  h1/h2/h3) is the same size as the correction at h1-h2 — so s1+real-correction lands inside
  the s0+junk-correction distribution at h1-h2 — and smaller than the correction at h3, where
  s1 clears 20/20 even on the handicapped comparison. Nothing else is needed to explain the
  ranks. (nbrin family increments below, §5.)

Scope caveat (audit finding 3, inherited): 20/20 certifies "this specific graph beats chance
graphs" on this backbone; the ≥300 km and index-conditioning controls that closed the
leakage/shared-climate holes for the linear effect have not been run for the stacked arm.

## 4. Per-basin FDR and geography: the stack re-finds the LINEAR neighbor map, not resMLP's

Per-basin DM (monthly losses pooled across folds, 84 test months/basin, BH q=0.10;
`p8_perbasin_fdr*.csv`), lstmres vs same-seed lstm_own_era5, win/lose counts:

| h | s0 | s1 | ensemble | basins favoring stack (ens) |
|---|---|---|---|---|
| 1 | 13/0 | 18/1 | **19/1** | 157/234 |
| 2 | 21/1 | 12/2 | **26/5** | 149/234 |
| 3 | 15/1 | 17/3 | **22/4** | 153/234 |

Winner-dominated at every horizon and seed — unlike the phase 5 linear effect's 9/7 split.
Cross-seed per-basin DM geography: Spearman 0.80/0.65/0.82 at h1/h2/h3 (seed-stable).

**Geography (ensemble h1 winners, 19):** Africa-heavy — Niger, Lower Lake Chad, Rift Valley,
Shebelli-Juba, Lake Rukwa, East/Central Africa Coast, Indian Ocean Africa Coast, South Africa
South Coast (8/19), plus the Brazil pair (East Brazil, São Francisco), Dniester, Sumatra,
Ganges-Brahmaputra, Yellow Sea, Junggar Pendi, Svalbard, Novaya Zemlya, Hudson Strait. Sole
h1 FDR loser: **R_Yenisey_River** — the same basin significantly hurt by the neighbor in
phases 3-6. Recurrent h2-h3 losers: Amur, Yukon, Great Britain, South Mediterranean.

**Concordance — the surprise of the phase:**

- vs the **phase 5 linear** neighbor DM (corr_top1 vs own_ridge, h1): Spearman **+0.46 (s0) /
  +0.51 (s1)**, p ≤ 8e-14. Five of the 19 ensemble h1 winners are phase 5 linear FDR winners
  (Niger, São Francisco, East Brazil, Rukwa, Hudson Strait); ZERO phase 5 linear losers
  appear among the stack's winners; Yenisey is the shared loser.
- vs the **phase 7 resMLP** neighbor DM (resmlp_corr_top1_s0 vs ridge_corr_top1): Spearman
  **−0.21/−0.24 at h1** (p ≤ .0014), ≈ 0 at h2-h3.

So the stacked correction is **NOT finding the same nonlinear channel resmlp found**. resMLP's
gains were the mirror image of the linear map (rho = −0.90); the stack's gains are aligned
with the linear map (+0.5) and consequently anti-aligned with resMLP's (−0.2). Reading: the
LSTM-ERA5 stage 1 already absorbs the forcing-related structure that resMLP's channel
exploited where the linear map failed; what survives in the stage-1 residual is the classic
TWSA-specific regional-denoising signal — the same basins (Africa, Brazil, with Yenisey hurt)
that carried the linear +0.5% all along, now worth +0.9-1.3% against a much stronger backbone.
This is geography-level evidence that the correction is picking up the established real
signal rather than an artifact of the stacking.

## 5. The nbrin family: stacking on the in-LSTM-neighbor stage 1 is dominated (task 5)

`lstmres_corr_top1` vs `lstmres_nbrin_corr_top1` (same seed; ensemble): +0.69/+1.07/+1.05%
(DM p = .009/1e-6/2e-9); per-seed s0 +1.00/+1.67/+1.44 (all sig), s1 +0.32/+0.28 ns at h1-h2,
+0.65 (p=.009) at h3. The clean own-state stage 1 plus correction dominates.

The nbrin stage-2 increment itself (vs `lstm_corr_top1_era5`, same seed) is **negative at h1**
(s0 −0.13 ns, s1 −0.26, p=3e-4 — significantly harmful), ns at h2, positive at h3
(+0.56/+0.45, p=.006/.037). Seed-matched placebo increments for the family (mean −0.15/−0.11/
−0.09%): the real s0 increment at h1 beats only 11/20 placebo increments — indistinguishable
from junk — but clears 20/20 at h2-h3 (s1: 4/18/20 at h1/h2/h3). Interpretation: when the neighbor is already
inside the encoder, a second neighbor-only correction double-counts at h1 (where the in-LSTM
channel still carries some signal) and only helps at h3 where the encoder wastes it. Both
phase 7 conclusions are confirmed: the in-LSTM neighbor channel costs skill under ERA5
(lstm_corr_top1_era5 vs lstm_own_era5: s0 −0.78/−0.73% at h2/h3, p ≤ .004), and the correct
delivery is the dedicated correction stage.

## 6. Cross-architecture ladder (task 6) — pooled skill (%) vs ridge_own; matched 19,656-row sets

| model | h1 | h2 | h3 |
|---|---|---|---|
| kalman_ar1 | +0.18 | +0.89 | −0.36 |
| ridge_own | 0 | 0 | 0 |
| ridge_corr_top1 | +0.49 | +0.16 | +0.11 |
| ridge_own_era5 | +5.17 | +1.81 | +0.65 |
| ridge_corr_top1_era5 | +5.63 | +1.92 | +0.72 |
| resmlp_corr_top1 (3-seed ens, phase 7) | +1.20 | +1.84 | +1.78 |
| resmlp_own_era5 (3-seed ens, phase 7) | +7.90 | +4.44 | +2.98 |
| lstm_own_era5 (2-seed ens) | +7.02 | +3.76 | +2.07 |
| **lstmres_corr_top1 (2-seed ens)** | **+7.89** | **+4.81** | **+3.32** |

(`p8_ladder.csv`; resMLP ensemble rows verified to sit on the identical row set, max target
diff 0.0.)

**New overall best, with one honest nuance.** `lstmres_corr_top1_s0` is the lowest-RMSE single
arm at all three horizons across every phase 7+8 arm and ensemble (1.0028/1.1571/1.2257), and
the lstmres ensemble is the best ensemble at h2-h3. At h1, however, the 3-seed
`resmlp_own_era5` ensemble — which uses **no neighbor at all** — is statistically
indistinguishable from the stack: lstmres_ens vs resmlp_own_era5_ens = −0.01% (p=.98) at h1,
+0.39% (p=.35) at h2, +0.35% (p=.26) at h3 (`p8_followup_contrasts.csv`). So the correct
model-ranking claim is: lstmres_corr_top1 is the new nominal best at every horizon and clearly
best-in-class at h2-h3 among same-backbone comparisons, but it is not DM-separable from the
neighbor-free resmlp_own_era5 ensemble. The controlled within-backbone contrast (§2), not the
between-backbone ranking, is the evidence that the neighbor adds skill.

## 7. Anomalies and sanity (task 7)

1. **Seed spread vs effect size:** stage-1 seed spread is 0.93/1.06/0.67% (skill of
   lstm_own_era5_s0 over s1) — comparable to the correction itself. But the correction's
   INCREMENT is seed-stable to 0.00-0.14 pp (h1: +0.9029 vs +0.9046). The stack inherits the
   backbone's seed noise in its level, not in its increment. Report increments per-seed or
   the 2-seed ensemble; never a single seed's level.
2. **No fold where the correction hurts:** 45/45 fold-cells positive (min +0.22). The f3
   (La Niña/ERA5-misinformation) weakness appears only as a dampened gain (+0.25-0.61 at h1),
   not a reversal — the correction is robust to the backbone's known bad fold.
3. **The h1 s1 placebo rank (6/20) is FULLY explained by the stage-1 asymmetry** (§3): the
   0.93% seed-0 stage-1 advantage ≈ the 0.90% correction, and the seed-matched increment
   beats 20/20. The run log's caution ("seed-matched placebo increments needed before quoting
   s1 ranks") is resolved — quote 20/20 both seeds, all horizons, on increments.
4. **Is the neighbor channel's size preserved as the backbone strengthens?** Stacked gain
   +0.93/+1.09/+1.27% (ens) on the LSTM-ERA5 backbone vs resMLP-over-ridge +0.71/+1.69/+1.67%
   (3-seed ens vs ridge_corr_top1; per-seed range +0.50..+1.78) and resMLP-vs-ridge_own
   +1.20/+1.84/+1.78%. So: h1 preserved or slightly larger; h2-h3 attenuated to roughly
   two-thirds. A ~1% neighbor channel survives a backbone that is itself 7-8% above ridge_own
   — the channel shrinks slowly, consistent with partial (not total) absorption by ERA5
   sequence modeling. Caveat: the phase 8 correction is "plausibly attenuated" by in-sample
   stage-1 residuals, so the true h2-h3 attenuation may be smaller than it looks.
5. **Marginal cell to disclose:** lstmres_ens vs lstm_corr_top1_era5_ens at h1 is +0.51%
   p=.075 (the in-LSTM neighbor arm is not significantly beaten at h1, because that channel
   still helps s1 at h1); at h2-h3 it is +1.21/+1.58, p ≤ 8e-6.
6. **Placebo increments are slightly negative on average** (−0.01 to −0.12%) — a junk stage-2
   correction mildly overfits; nothing about the architecture manufactures skill.

## 8. For the paper

### (a) Numbers to quote (with audit phrasing constraints)

- Headline (controlled, same stage-1 net; honest evaluation, plausibly attenuated
  correction): "a neighbor-only residual correction adds **+0.93% / +1.09% / +1.27%**
  (2-seed ensemble; per-seed +0.90-0.90 / +0.99-1.12 / +1.20-1.28) at h1/h2/h3 over the
  LSTM-ERA5 stage 1, DM p ≤ 6.5e-9 (ens), bootstrap CIs excluding zero (h1 +0.70..+1.18);
  positive in 45/45 fold-cells; per-basin FDR 19/1 -- 26/5 -- 22/4 winners/losers (ens)."
- Placebos, seed-matched: "the real correction's increment exceeds all 20 degree-matched
  random-graph increments at every horizon for both seeds (placebo family mean increment
  ≈ 0: −0.01 to −0.12%)" — state as graph-specificity, not proven information transfer.
- Ladder: lstmres_corr_top1 vs ridge_corr_top1_era5 +2.40/+2.95/+2.62% (ens, p ≤ 3e-10); vs
  kalman_ar1 +7.72/+3.95/+3.67%; new best single model at all horizons (RMSE 1.0028/1.1571/
  1.2257, s0); NOT DM-separable from the neighbor-free resmlp_own_era5 3-seed ensemble
  (h1 −0.01% p=.98) — disclose this alongside any "best model" claim.
- NEVER quote the summary's `dm_vs_ridge_twin` for lstmres arms (pairs against no-ERA5
  ridge_corr_top1; information-mismatched). Use the contrasts above.
- Geography: correction's per-basin DM correlates +0.46/+0.51 with the phase 5 LINEAR
  neighbor map (and −0.21/−0.24 with resMLP's mirror-image map); Africa 8/19 h1 FDR winners;
  Yenisey again the lone h1 loser.

### (b) Ranked next steps

1. **Extend lstmres_corr_top1 + lstm_own_era5 to h4-h6** (one seed suffices; keep the
   placebo family). Now doubly motivated: it is the best model AND the correction GROWS with
   horizon (+0.93→+1.27), on a collision course with Li's h4+ edge. Feeds the centerpiece
   skill-vs-horizon ladder figure.
2. **Neighbor correction on the resmlp_own_era5 backbone.** The h1 tie between the stack and
   neighbor-free resmlp_own_era5 is the open ranking question; if the same stage-2 correction
   lifts that backbone too, "the neighbor channel survives any strong backbone" becomes a
   two-backbone result and likely yields an outright best model. Cheap (sklearn stage only).
3. **Port the linear-effect controls to the stacked arm:** ≥300 km graph exclusion and
   ENSO/IOD conditioning (audit finding 3). These closed the leakage/shared-climate holes for
   the +0.5% linear claim; the +1% stacked claim needs them before submission.
4. **Attenuation bound:** refit stage 2 on out-of-fold (within-train K-fold) stage-1
   residuals, one seed — turns "plausibly attenuated" into a measured bound.
5. **Third LSTM seed** (s2) for stage 1 + stack, to match resMLP's 3-seed ensembles and
   firm up the h1 ranking against resmlp_own_era5.

### (c) Verdict on "the neighbor is redundant under ERA5"

**Overturned in its general form; survives only as a statement about delivery mechanism.**
Phase 7 concluded the neighbor channel is absorbed once ERA5 forcing is supplied, because
every within-architecture neighbor increment under ERA5 was ns or seed-sign-flipping. Phase 8
shows that conclusion was an artifact of HOW the neighbor was delivered: fed to the encoder
as an input channel it is redundant-to-harmful (confirmed again here — nbrin family, §5), but
delivered as a dedicated residual-correction stage it adds a seed-stable, fold-stable,
placebo-clean (seed-matched 20/20, both seeds, all horizons) **+0.9-1.3% on top of the
strongest ERA5-sequence backbone in the project — larger than the original +0.5% linear
effect, growing (not dying) from h1 to h3**, and concentrated in the same basins that carried
the linear effect (Spearman +0.5), with Yenisey still the lone hurt basin. Neighbor
information therefore survives sequence-ERA5 conditioning; ERA5 absorbs the resMLP-style
mirror-image channel but not the core regional-denoising signal. The paper's neighbor claim
can be restored from "mechanism only, not deployable" to "deployable via two-stage
correction", scoped by: honest-but-plausibly-attenuated construction, graph-specificity (not
transfer) certification pending the ≥300 km/conditioning ports, and the disclosed h1 tie
with the neighbor-free resmlp_own_era5 ensemble.
