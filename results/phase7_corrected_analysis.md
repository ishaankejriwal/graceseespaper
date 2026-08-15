# Phase 7 Corrected-Pipeline Analysis (2026-08-14)

Recomputed from `phase7_{resmlp,lstm,gnn}_predictions.csv` (corrected pipeline: CSR month fix
+ issue-date fold membership) with `src/gracefc/stats.py` (paired rows enforced). Skill % =
1 − MSE_a/MSE_b on paired rows; DM = pooled monthly (Harvey-corrected); CI = moving-block
bootstrap (1000 draws) on the seed-ensemble. Per audit rule P0-6, every contrast is reported
per-seed AND as the seed-ensemble (predictions averaged across seeds per row, then scored).
Seeds: resmlp/gnn s0-s2, lstm s0-s1. `phase7_analysis.md` (2026-08-13) is STALE and used only
as the pre-audit baseline.

**Integrity checks (Q5/Q6): all pass.**
- n per arm per horizon is exactly 19,422 / 19,188 / 18,954 (h1/h2/h3) in all three files;
  zero NaN predictions. (Pre-audit files had 19,656 — the issue-date protocol trims
  fold-boundary months, as expected.)
- The five shared reference arms (`kalman_ar1`, `ridge_own`, `ridge_corr_top1`,
  `ridge_own_era5`, `ridge_corr_top1_era5`) are row-identical across the resmlp/lstm/gnn files
  (pandas row-hash equality on sorted predictions).
- Digit-level cross-phase check: `ridge_own_era5` RMSE 1.0253533779/1.1929806049/1.2685130189
  and `ridge_corr_top1_era5` 1.0236989492/1.1935290418/1.2696690468 match
  `phase6_era5_predictions.csv` to 1e-9 at all three horizons. (phase3b uses the `kalman_*`
  arm naming, so no same-name arm exists there to compare; the phase6 match covers the shared
  ridge stack.)

---

## Q1. LSTM sequence gain (lstm_own_era5 vs ridge_own_era5) — SURVIVES, slightly larger

| h | s0 | s1 | ens | ens DM p | ens 95% CI |
|---|-----|-----|-----|---------|------------|
| 1 | +2.35 | +1.27 | **+2.17** | 4.6e-5 | [+1.30, +3.18] |
| 2 | +2.85 | +1.32 | **+2.36** | 3.4e-9 | [+1.66, +3.02] |
| 3 | +1.68 | +0.87 | **+1.43** | 1.5e-5 | [+0.80, +2.01] |

Pre-audit: s0 +2.05/+2.24/+1.60, s1 +1.13/+1.19/+0.94, ens +1.95/+1.99/+1.43. Same story,
marginally stronger after correction. Seed spread is large (s1 ≈ half of s0) but sign-stable
everywhere — no flags. This remains the biggest architectural effect of the phase.

GRACE-only control (lstm_own vs ridge_own): ens +0.32 (p=.10) / +0.93 (p=1.5e-4) / **−1.03**
(p=.008). Sequence modeling without ERA5 is not a robust win and is significantly *worse* at
h3 — the LSTM gain lives in the ERA5 forcing sequence, as before.

## Q2. resMLP architecture gain — SURVIVES, strengthened at h2-h3

**(a) GRACE-only, identical features (resmlp_corr_top1 vs ridge_corr_top1):**

| h | s0 | s1 | s2 | ens | ens DM p | ens 95% CI |
|---|-----|-----|-----|-----|---------|------------|
| 1 | +0.66 | +0.90 | +0.79 | **+0.81** | 6.7e-5 | [+0.41, +1.17] |
| 2 | +2.00 | +2.15 | +2.18 | **+2.15** | 2.8e-12 | [+1.59, +2.69] |
| 3 | +2.00 | +2.12 | +2.35 | **+2.17** | 8e-10 | [+1.65, +2.73] |

Pre-audit s0 was +0.50/+1.45/+1.74 — the corrected gain is ~0.2-0.6 pp larger at every
horizon, seed-stable (spread ≤0.35 pp), all seeds individually significant. Vs `ridge_own`:
ens +1.12/+2.05/+1.99, all p ≤ 6e-10.

**(b) ERA5 twin (resmlp_corr_top1_era5 vs ridge_corr_top1_era5):** ens +2.39/+3.07/+2.48
(p ≤ 9.3e-6, CIs well clear of 0), but **SEED-INSTABILITY FLAG: s2 is an outlier**
(−0.06/+0.83/+0.44, none significant, sign flip vs s0/s1 at h1) while s0/s1 sit at
+1.1 to +2.1. The ens beats every individual seed — classic variance-averaging of a noisy MLP
stage. Quote the ens with the s2 caveat, never s0 alone.

**(c) Neighbor-free architecture gain (resmlp_own_era5 vs ridge_own_era5):** ens
**+3.35/+3.58/+2.84** (p ≤ 4.6e-9), per-seed +1.9-2.1/+2.4-2.8/+0.8-2.4 (s0 h3 +0.80 p=.34 is
the weak cell; sign never flips). The two-stage nonlinear correction is worth ~3% with *no
neighbor at all* — the largest resMLP number in the phase, and the key confound for phase 8
(see recommendation).

## Q3. Neighbor increments — null-to-negative on every backbone, with ONE surviving exception

Same-architecture, same-seed contrasts (a-question), ens rows with per-seed range:

| backbone | contrast | h1 | h2 | h3 |
|---|---|---|---|---|
| resMLP+ERA5 | corr_top1_era5 vs own_era5 | **−0.67** (p=.012; seeds −1.9..−0.2) | **−0.62** (p=.012; −1.9..−0.35) | **−0.55** (p=.047; −2.1..+0.5) |
| LSTM (GRACE-only) | corr_top1 vs own | +0.25 (p=.19) | **−0.87** (p=.019) | **−0.51** (p=.015) |
| LSTM+ERA5 | corr_top1_era5 vs own_era5 | +0.36 (p=.09) | **−0.42** (p=.038) | −0.43 (p=.067) |
| resMLP | top2 vs top1 | +0.13 (ns) | −0.28 (ns) | −0.15 (ns) |
| GNN | top2 vs top1 | +0.16 (ns) | **−0.48** (p=.013) | −0.33 (p=.081) |

**SEED FLAGS:** LSTM+ERA5 h2 flips sign across seeds (s0 −1.46 p=3e-5 vs s1 +0.71 p=.031) —
do not quote either seed alone; the ens is −0.42. GNN top2_era5 h2 also flips (s2 +0.63 vs
s0 −0.40). resMLP top2_era5 vs top1_era5 is +0.82 at h2 ens (p=.006, all seeds positive) but
chains with the negative top1-vs-own increment to ≈ +0.2 vs own_era5 — consistent with the
neighbor null once ERA5 is present.

**Placebo ranks (b-question; real graph vs 20 degree-matched rand graphs):**
- **lstm_corr_top1: 0/20 at h2 and h3 (ens and s1)** — the real neighbor is *worse* than every
  random graph; actively harmful. h1 is 19-20/20 with a +0.25-0.36 ns increment: a weak
  h1-only hint, not evidence.
- lstm/resmlp/gnn ERA5 arms: seed-level ranks are mediocre (resmlp_corr_top1_era5 seeds:
  10,7,1/20 at h1; 8,8,0/20 at h2). Ens ranks look better (20/20 in places) but the ens is a
  variance-reduced 2-3-seed average scored against single-seed placebos — a biased comparison;
  do not headline it.
- **THE EXCEPTION — resmlp GRACE-only: every seed of resmlp_corr_top1 and _top2 beats 20/20
  placebos at every horizon (p_rank = .048, 12/12 seed×h cells for top1).** The placebo family
  itself pools to ridge_own level (family RMSE 1.0503/1.2018/1.2711 vs ridge_own
  1.0499/1.2014/1.2706), while the real-graph arm sits at 1.0448/1.1899/1.2591. So on the
  corrected pipeline the pre-audit mechanism finding *survives in full*: a junk neighbor
  through the two-stage MLP adds nothing, the real neighbor adds +0.8/+2.1/+2.2% — despite the
  linear neighbor effect being null (h1 +0.31% p=.20). This is the single robust
  neighbor-information signal in the study, it is nonlinear-only and GRACE-only, and it
  disappears (goes negative) as soon as ERA5 enters the backbone.

## Q4. GNN — still never beats ridge

gnn_corr_top1 vs ridge_corr_top1: negative at every seed and ens, every horizon (ens
−0.90/−0.24/−0.61; h1 and h3 significantly worse). ERA5 twin: negative or ns everywhere
(best ens +0.44 h3, p=.077). top2 ≤ top1 as before (ens −0.48 h2, p=.013). Exhaustive scan of
all 4 gnn arms × 4 seeds/ens × 3 horizons vs their ridge twins finds exactly one nominal win:
gnn_corr_top2_era5_ens vs ridge_corr_top2_era5 at h2, +0.65% (p=.035) — no individual seed is
significant (s0 −0.40, s1 +0.00, s2 +0.63 p=.073), it is against the *weakest* ridge twin, and
placebo ranks for that family are poor (2-14/20). Noise. Conclusion unchanged: message-passing
adds nothing here.

## What changed vs pre-audit (stale phase7_analysis.md)

| claim | pre-audit | corrected | verdict |
|---|---|---|---|
| LSTM ERA5 sequence gain | s0 +2.05/+2.24/+1.60; ens +1.95/+1.99/+1.43 | s0 +2.35/+2.85/+1.68; ens +2.17/+2.36/+1.43 | **survives, slightly larger** |
| resMLP beats ridge twin (GRACE-only) | s0 +0.50/+1.45/+1.74 | s0 +0.66/+2.00/+2.00; ens +0.81/+2.15/+2.17 | **survives, stronger at h2-h3** |
| resMLP 20/20 placebo wins, all seeds (GRACE-only) | yes | yes (12/12 cells, p_rank .048) | **survives** |
| placebo family ≈ ridge_own (junk graph adds nothing) | yes | yes (Δ ≤ 0.0005 RMSE) | **survives** |
| neighbor increment under ERA5 (resMLP) | mixed, ≈0 (−1.2..+0.8 by seed) | ens −0.67/−0.62/−0.55, all p<.05 | **sharpened: now significantly negative** |
| LSTM neighbor increment | ns / mixed | h1 ns positive; h2-h3 negative, real graph 0/20 vs placebos | **null confirmed, h2-h3 harmful** |
| GNN never beats ridge, top2≤top1 | yes | yes (one ens-only h2 blip, no seed support) | **survives** |
| n per horizon | 19,656 | 19,422/19,188/18,954 | protocol change, expected |

## Recommendation for Phase 8 — RUN, but MODIFIED; do not run as-is

Phase 8 (stacked LSTM: stage-1 lstm_own_era5 + stage-2 neighbor residual correction,
`lstmres_corr_top1`) is **not foreclosed** — h4-6 is untested territory and the pre-audit
phase-8 run claimed a +1.2/+1.1/+1.3% stage-2 increment there — but the corrected phase-7
evidence predicts its headline confound precisely:

1. Every in-model neighbor increment on an ERA5 backbone is now null-to-negative
   (resMLP −0.6, LSTM −0.4 at h2-h3). The one robust neighbor signal (resMLP two-stage) exists
   only *without* ERA5, and phase 8's stage 1 is an ERA5 LSTM.
2. The neighbor-free two-stage correction alone is worth +2.8-3.6% (resmlp_own_era5 vs
   ridge_own_era5). A pre-audit phase-8 "stage-2 gain" of ~1% could therefore be pure
   second-stage nonlinear correction with zero neighbor content.

**Modification (required):** add a neighbor-free stage-2 control arm — same stage-2 MLP fed
own-basin features only (`lstmres_own`) — so the stage-2 increment decomposes into
(architecture: lstmres_own vs lstm_own_era5) + (neighbor: lstmres_corr_top1 vs lstmres_own),
each with the existing 20 rand-graph placebos on the neighbor part. Keep ≥2 seeds and report
per-seed + ens (the resMLP s2 outlier and the LSTM h2 sign flip show single-seed phase-8
numbers would be untrustworthy). Prediction from phase 7: the architecture term survives, the
neighbor term does not — but the GRACE-only resMLP exception at h2-h3 leaves a real
possibility that neighbor residual correction works at long leads where own-state information
decays, which is exactly what the control arm will adjudicate.

Priority ranking: (1) phase 8 rerun with the lstmres_own control, (2) resMLP GRACE-only
mechanism write-up (the surviving nonlinear neighbor signal is the paper's novelty candidate),
(3) GNN: drop from the paper's main line, keep as a negative control.
