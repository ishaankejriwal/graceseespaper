# Manuscript rewrite ledger — corrected pipeline (2026-08-15)

Every number in `main.tex` predates the 2026-08-13 repository audit and is STALE. This file
is the single source of truth for the rewrite. Numbers here are copied from the corrected
result CSVs and were independently reproduced by audit subagents
(`results/post_rerun_audit.md`, `results/phase8_corrected_audit.md`).

**Sample:** 234 basins, 5 expanding-window folds, **issue-date** fold membership.
n per lead (identical for every model): **19,422 / 19,188 / 18,954 / 18,720 / 18,486 / 18,252**.

---

## 1. Baseline ladder — `results/paper_baseline_ladder.csv`

Skill (%) vs the stronger damped-persistence variant at each lead, matched rows:

| Model | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| Climatology (zero) | −125.60 | −70.73 | −58.93 | −51.36 | −42.69 | −33.62 |
| Persistence | −20.75 | −22.68 | −21.39 | −19.49 | −16.04 | −15.80 |
| Damped persistence (stronger) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Pooled ridge (own lags) | +1.17 | +4.78 | +5.08 | +3.64 | +3.17 | +2.49 |
| Pooled ridge + indices | +0.99 | +4.52 | +5.14 | +3.74 | +3.17 | +2.38 |
| Per-basin ridge | +3.05 | +5.94 | +4.26 | +2.05 | +0.38 | −0.29 |
| **Kalman (filtered-state persistence)** | **+4.98** | **+8.79** | +5.62 | +3.07 | +2.55 | +3.63 |
| Ridge on filtered states | +4.82 | +7.87 | **+6.04** | **+5.36** | **+5.14** | **+5.16** |

Note: the *stronger* damped variant is rho-damping at h1 and regression-damping at h2+;
`damped_persistence_rho` degrades badly at long leads (−6.4 to −12.0 at h3–h6).

### Contrasts — `results/paper_baseline_contrasts.csv`
Kalman vs **per-basin ridge**: +1.99 (p=6.6e-4) / +3.04 (7.7e-7) / +1.43 (0.013) /
**+1.04 (p=0.078, NOT SIGNIFICANT)** / +2.17 (0.0051) / +3.92 (8.3e-6)
Kalman vs **pooled ridge**: +3.86 (9.1e-6) / +4.21 (1.2e-11) / +0.57 (0.45) / −0.59 (0.46) /
−0.64 (0.44) / +1.17 (0.19)

### MANDATORY framing changes
- The old claim "beats per-basin ridge at **all six** leads (p≤0.042)" is now **FALSE**.
  Write: "at five of six leads; lead 4 is not significant (p=0.078)."
- The old claim about the pooled ridge ("significant at 1–3 and 6, nominal at 4–5") is now
  **wrong in both directions**. Truth: significant at **h1–h2 only**; h4/h5 nominally negative.
- **The headline is STABILITY, not growth.** Pre-audit was +4.8/+8.1/+5.3/+3.2/+2.5/+3.5;
  corrected is +5.0/+8.8/+5.6/+3.1/+2.6/+3.6. The benchmark result is essentially unchanged
  by both defect fixes — state this as robustness. Do NOT claim the margin grew.
- The filtering class still owns h3–h6 through *ridge on filtered states* (+4.8 to +6.0), which
  is the strongest available long-lead statement.

---

## 2. ERA5 forcing — `results/phase6_era5_headline.csv`
ridge_own_era5 vs ridge_own: **+4.62% (CI +1.91,+6.84; p=1.0e-4)** / +1.39% (p=0.092) /
+0.33% (p=0.61).
**Change:** ERA5 is now a **lead-1 result only**. Old text claimed +5.2/+1.8 at h1/h2 with h2
significant — h2 has LOST significance.
GBM/MLP still never beat ridge: gbm_own_era5 +0.14 ns / +0.60 ns / −0.71 (p=0.037, worse);
mlp_own_era5_s0 −0.53 ns / −2.56 (p<1e-4) / −3.14 (p=1e-4). This convergence with Nie et al.
(2025) is now on FIRMER ground than pre-audit.

---

## 3. Cross-basin signal — LINEAR TIER IS NULL

`results/phase3b_predictions.csv`, kalman_corr_top1 vs kalman_own_ridge:
h1 **+0.308% (DM p=0.199 — NOT SIGNIFICANT)**; h2 −0.097; h3 −0.192; h4 −0.358; h5 −0.349;
h6 −0.321 (all ns, all negative).
Pre-audit was +0.488%, p=0.022 (significant). **The registered capacity-matched test no longer
reaches significance. Contribution 3 at the linear tier is a NULL RESULT.**

### Two questions that must be stated SEPARATELY (do not blur)
1. *Does correlation-selection beat a random neighbor?* **Yes at h1** — beats 50/50
   degree-matched random graphs (p_rank=0.0196); ≥300 km variant also 50/50.
2. *Does adding a neighbor beat not adding one?* **No** (p=0.199).
At h2–h6 the real graph is beaten by **0/50** placebos — the selected neighbor is worse than a
random basin beyond one month. (Minor: the ≥300 km arm at h2 is 2/50, not 0/50.)

### Resolution stratification (PRIMARY sensitivity, lead 1) — SUPERSEDED FRAMING 2026-08-15
| Stratum | Basins | Skill | DM p |
|---|---|---|---|
| Resolved ≥90,000 km² | 199 | +0.099% | 0.715 |
| Sub-resolution <90,000 km² | 35 | +1.033% | 0.00065 |
Under ≥300 km exclusion: resolved +0.140% (p=0.608); sub-resolution +0.964% (p=0.00087).
These numbers stay in the paper, but the interpretation changed (resolution audit,
RUN_LOG 2026-08-15). **BANNED SENTENCE: "no detectable cross-basin effect in basins GRACE
can resolve" — the contamination 2×2 refutes it.** Required replacements:
- **Contamination**, not size, is the real stratifier: per-basin `contamination` = share of
  the basin's mascon-weighted signal contributed by land OUTSIDE the basin
  (Spearman vs area only −0.325, a nearly independent axis). 2×2 at h1 (linear arm,
  `results/resolution_cross_2x2.csv`): resolved×cont-high n=57 **+1.161% p=0.0045**;
  resolved×cont-low/mid n=142 **−0.500% p=0.082**; sub×cont-high n=21 +1.166% p=0.0071;
  sub×cont-low/mid n=14 +0.913% p=0.036. Correct sentence: *the linear neighbor effect is
  confined to basins whose mascon footprint is shared with land outside the basin,
  regardless of basin size* — footprint sharing (leakage), with "small basins" as its proxy.
- Linear effect by contamination tercile across leads (ridge_corr_top1_era5 vs
  ridge_own_era5, `results/phase8_stratification.csv`): cont-high h1 +1.11% p=2.3e-4,
  h2 +0.59% p=0.037, dead h3–h6; mid/low terciles negative at every lead. The linear
  effect is short-horizon AND contamination-confined.
- 90,000 km² is ~**7** native mascons, not one resolution element: empirical tile recovery
  gives 42,107 global tiles, median 12,123 km² (~1° equal-area). Only 3/234 keep-basins
  span <2 tiles. Do not justify the threshold as "one resolution element".
- **Area provenance**: basin areas are OUR cos-latitude mask integration; CSR supplies no
  tile/area geometry with the gridded product. The paper must say so.
- Still claim "characterization with null-model controls", not "isolation and validation".

### Stacked-correction stratification (NEW, 2026-08-15 — the leakage-artifact test)
`results/phase8_stratification.csv`. lstmres_corr_top1_ens vs lstm_own_era5_ens is
**positive in all 60 stratum×lead cells and does NOT concentrate in high-contamination
basins**: cont-low tercile (n=78) +0.69/+1.31/+1.29/+1.28/+1.31/+1.84% (all p≤0.007);
resolved×cont-low/mid (n=142, where the linear effect is −0.50%) +0.89/+1.57/+1.46/+1.38/
+1.49/+2.00% (all p<1e-4); sub-resolution (n=35) +0.64..+1.05% (all p≤0.043). Per-seed
worst cell is still positive (+0.22%, tiny-n stratum). **Required framing: the linear
effect and the stacked correction have DIFFERENT signatures — the linear one is confined
to high-contamination basins at h1–2 (a leakage signature); the stacked correction is
positive in every stratum at every lead and NOT concentrated in shared-footprint or
small basins, so the footprint-sharing artifact does not explain it.** Do NOT overclaim
mechanism ("genuine teleconnection"); claim that the leading artifact explanation is
ruled out by stratification. **BANNED WORD for this result: "uniform"** (audit
2026-08-15): the SIZE axis has a real gradient — sub-resolution basins gain LESS
(interaction h1 p=0.026, h6 p<0.001), which is the OPPOSITE of what leakage predicts,
so state it as claim-safe texture ("if anything the small-basin stratum gains least").
Formal contamination high-vs-low interaction: ns at every lead (h1 p=0.099, trending
toward MORE gain in cont-high; h3 p=1.0; h6 p=0.66) — quote it to convert "flat rather
than concentrated" from descriptive to tested (numbers to land in
`results/phase8_stratification_interactions.csv`).
(Diagnostic, optional: plain resMLP neighbor-vs-own ERA5 twin is NEGATIVE overall,
−0.55..−0.67% p<0.05, harm concentrated in LOW/MID contamination — naive neighbor
features import noise except where footprints overlap; only the two-stage stack converts
neighbor state into uniform gains.)

### ERA5 conditioning
ridge_corr_top1_era5 vs ridge_own_era5 h1 **+0.322% (p=0.115)** vs unconditioned +0.308%
(p=0.199). The control still does its job (11 local-meteorology variables leave the estimate
unmoved → not a shared-weather proxy), but **both arms are non-significant**. Correct phrasing:
"conditioning does not explain the effect away", NOT "the effect survives conditioning".

---

## 4. Architectures — `results/phase7_corrected_analysis.md` (all 2-seed/3-seed ensembles)
- LSTM sequence gain **survives, slightly larger**: lstm_own_era5 vs ridge_own_era5
  **+2.17/+2.36/+1.43%** (all p<2e-5). GRACE-only LSTM still not a win (h3 worse than ridge).
- resMLP **strengthens**: vs ridge twin +0.81/+2.15/+2.17%.
  Largest architectural number in the study is **neighbor-free**: resmlp_own_era5 vs
  ridge_own_era5 **+3.35/+3.58/+2.84%**.
- **GNN never beats ridge** anywhere; two-hop ≤ one-hop. Question stays closed.
- **SURVIVING NONLINEAR NEIGHBOR SIGNAL:** GRACE-only resMLP neighbor beats **20/20** placebos
  at every seed and horizon, with the placebo family sitting exactly at ridge_own. This is the
  one place the neighbor survives at the linear-tier information set, and it supports the
  "matters most where forcing data is absent" reading.
- Seed flags to disclose: resmlp_era5 s2 outlier; LSTM h2 neighbor sign flip s0 vs s1.

---

## 5. Delivery / stacked correction — FINAL, `results/phase8b_h16_ensemble_headline.csv`

**THE skill claim** — neighbor correction vs its own stage-1 LSTM (2-seed ensemble):
| h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|
| **+0.91** | **+1.45** | **+1.33** | **+1.32** | **+1.42** | **+1.96** |
All p ≤ 3.14e-08 (max at h4; h1 = 1.10e-10, h6 = 1.21e-13 — the old "all p ≤ 1.2e-10" was
a transcription slip from the h1 value, caught by the figure build 2026-08-15);
CI lower bounds +0.64/+1.07/+1.02/+0.98/+1.11/+1.67 (exclude zero everywhere).
Per-seed: s0 +0.64/+1.33/+1.29/+1.26/+1.51/+1.90; s1 +1.19/+1.56/+1.36/+1.37/+1.28/+1.93.
vs strongest linear system with identical information (ridge_corr_top1_era5):
+2.75/+3.86/+2.92/+2.52/+1.78/+2.23% (p ≤ 6.5e-8, max at h1 — the old "≤ 1.5e-9" quoted
the h6 value; caught by the manuscript audit 2026-08-15).

**Placebos:** beats **20/20** random-graph correction placebos in **all 12** lead×seed cells.
The pre-fix "16/20 at s1 h2" was an artifact of placebos riding the seed-0 network while being
scored against seed-1 arms; fixed, and it disappeared. Placebos cost only −0.27% to −0.05%
(pure-variance behaviour), so they are the correct **primary** control.
`p_rank = 0.0476` is the 1/21 floor, **not a measured p-value** — quote the z-score instead
(−6.8 to −18.6 placebo sd).

**DELIVERY RESULT (now honestly at ensemble level).** Same information, two deliveries:
- as an LSTM **input channel**: +0.36% (p=0.090) at h1; **negative** at h2–h6
  (−0.42/−0.43/−0.61/−0.37/−0.38, p=0.038/0.067/0.010/0.169/0.332)
- as a **dedicated correction stage**: +0.91 to +1.96% at every lead
Old text said "dead or harmful" from a single seed — that phrasing is **banned**; the honest
statement is "no robust ensemble gain as an input channel".

**The own-state control (`lstmres_own`) — DIAGNOSTIC ONLY, never a skill number.**
Control vs stage-1: **−1.16/−1.46/−1.90/−2.12/−1.84/−1.80%** (it HARMS, p ≤ 2.5e-7).
Neighbor vs control: +2.05/+2.87/+3.18/+3.37/+3.20/+3.70% — **NEVER quote this as the
neighbor's skill, nor as "the architecture term".** It is inflated because the control is
actively harmful.
*Why it harms (audit-verified, and NOT "it overfits noise"):* its correction is
**anti-correlated** with stage-1 error (corr −0.08 to −0.16); only 0.16–0.54pp of the harm is
variance cost, so **79–85% is a systematically wrong-direction correction**, an artifact of the
in-sample stage-2 residual protocol. It is therefore an inferentially **too-easy** control.
Its legitimate use: evidence that the gain is the neighbor's **information**, not the
two-stage **architecture**.
Verified fair mechanically: the self-graph reproduces the basin's own propagated state exactly
(max|diff| = 0.000e+00); 0 of 234 basins lack a top-1 neighbor, so there is **no** zero-fill
asymmetry between arms.

---

## 6. The crossing — `results/phase8b_li_comparison_headline.csv`
Matched sample **227 basins × 60 months = 13,620 rows per lead** (was 61 months pre-audit).
lstmres_corr_top1_ens vs li_lstm_full:
| h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|
| **+20.01** (p=3.4e-4) | −3.49 (p=0.50) | −12.94 (p=0.015) | −17.44 (p=0.0021) | −23.64 (p=1.3e-4) | −30.34 (p=1.5e-6) |

**Pre-audit: +20.2/−3.8/−12.4/−16.5/−21.6/−26.7 → the crossing is ROBUST to both fixes.**
Crossing point remains between leads 2 and 3.
- Bare Kalman vs their full product: h1 **+14.63% (p=0.0039)**.
- vs their non-seasonal variant: h1 +30.94% (p=3.1e-11), h2 +8.96% (p=0.0083), h3 −1.74 (ns),
  then −8.91/−16.20/−23.38 — crossing moves to h3–h4.
- **Sharpest single statement in the study**, vs damped persistence: ours +11.14/+11.78/+13.38/
  +16.59/+15.55/+14.16% at h1–h6, while **theirs is −11.09% at h1** (worse than damped
  persistence, p=0.057) before rising to +14.76/+23.30/+28.97/+31.70/+34.14%.
- **Scope the h1 deficit; do NOT harden it** (added 2026-08-15 after a provenance check).
  p=0.057, and per-basin it is a near-tie: their product still beats damped persistence in
  **115 of 227** matched basins (50.7%), so the pooled −11.09% is magnitude in the losing
  basins, not breadth. Source: `results/phase8b_li_comparison_predictions.csv`, per-basin MSE
  at h1. Three supporting checks, all clean:
  (a) NOT a deseasonalization-convention artifact — calendar month explains only **0.12%** of
  their h1 error variance, *less* than damped persistence's own 0.38%; a seasonal-convention
  mismatch would leave an annual signature and does not;
  (b) trend handling agrees — their non-seasonal is de-seasoned *and* de-linearized
  (Li et al. 2020 WRR, doi:10.1029/2019WR026551) and our decomposition also removes a linear
  slope, so the two targets match in definition;
  (c) mechanism matches their architecture — their LSTM ingests no GRACE lags, so at h1 it has
  no access to the strongest available predictor. Their RMSE grows only 13.2% from h1→h6 while
  damped grows 47.1%; near-lead-independent error is exactly what a forcing-only model predicts.
- **Two framing obligations attached to that number** (now in main.tex §data\_li and §crossing):
  (i) Li & Kusche **never benchmark against persistence/climatology at any lead** — this is a
  floor they did not test, not a result we contradict; (ii) damped persistence at h1 is not
  operationally available to their user, since mascon latency is 2–3 months and that latency is
  the stated motivation for their product.
- **Product-identity caveat:** their published skill figures are for the **mean of three**
  FCast products (JPL/CSR/GSFC); we use CSR-FCast alone because our system is CSR-based
  (mascon-matched), but it is the member their own training-residual diagnostic ranks least
  certain (3.0 vs 2.8 cm). Our numbers characterize that member, not their headline product.
- Keep saying "empirical crossing **consistent with**" the IC/forcing crossover — the two
  systems differ in many ways at once, so this is not a causal source decomposition.

---

## 7. Kalman identification — unchanged caveat
1170/1170 fits converge, but **259 hit the r<1e-6 boundary** and **31 basins are at the boundary
in all five folds**. Keep "AR(1) state-space filtering"; never "measured cost of forecasting
measurement noise". Keep the identification-caveat paragraph and the limitation.

---

## 8. Data / protocol facts for Methods
- CSR **RL0603M** (file title; `product_version` = RL06.2). **257** monthly solutions.
- Solutions are assigned to their official calendar month via the product's documented
  missing-month list, with a span-overlap assertion. Midpoint binning mislabels the
  **November 2011** and **May 2015** solutions (arcs centred 2011-10-31 and 2015-04-27) and
  silently drops both months — this was a real defect, now fixed.
- Fold membership is on **issue dates**: every transform is fit strictly before the fold's first
  issue date, and all test forecasts are issued at or after it. The earlier target-date design
  let leads 2–6 use transforms fit up to h−1 months past the issue date.
- All results are **retrospective potential skill** (no GRACE-FO latency or ERA5 availability
  modelling).
- Twelve NOAA indices: conditioning controls **plus** one baseline-ladder arm; AMM and PDO are
  dropped from that arm (they end before the test window).

---

## 8b. VOICE AND READER RULES (user directive 2026-08-15 — binding on every writing pass)

1. **Fresh-reader test (the biggest rule).** The paper must read cleanly for someone who
   has never seen this project's workflow. Banned from rendered text: internal phase
   numbers ("phase 8b", "phase 3b" — papers do not have phases; describe the analysis
   instead), code arm names with underscores (`lstmres_corr_top1_ens`,
   `resmlp_own_era5`...), "RUN_LOG"/"ledger"/fold codenames, and the words "arm(s)" and
   "backbone" as workflow jargon. Give every model a plain-English name in ONE Methods
   table (e.g. "the two-stage model", "per-basin ridge on own-basin lags", "the
   residual-MLP variant"), then use those names exclusively and consistently
   (Steinhardt: one term per concept, never synonyms). Internal codenames may appear
   once, in the reproducibility appendix, as a name→code mapping. "Stage 1"/"stage 2"
   are fine after the two-stage architecture is defined.
2. **Human, professional, clear, simple.** Follow `.claude/skills/ml-paper-writing/SKILL.md`:
   one-sentence contribution stated in the intro (narrative principle); Gopen & Swan
   sentence rules (subject–verb proximity, stress position at sentence end, old-before-new);
   kill filler words (actually/very/quite/essentially/basically); no hedging "may/can"
   unless genuinely uncertain; no intensifiers; verbs not nominalizations ("we analyzed",
   not "we performed an analysis"); specific words ("RMSE", not "performance"). Abstract
   and introduction get as much polish as everything else combined.
3. Percent-comment provenance notes (`% source: ...`) are exempt — they never render.

## 9. Sections that must be RESTRUCTURED, not just renumbered

1. **Contribution 3 becomes two-sided.** Linear tier: a null overall, whose apparent
   pockets are confined to high-contamination (footprint-sharing) basins at h1–2 — a
   leakage signature, with "small basins" retired as a proxy framing. Nonlinear tier: a
   real, placebo-validated correction at every lead 1–6 that is positive in every
   contamination/size stratum and not concentrated in shared-footprint or small basins
   (survives the leakage stratifier; never say "uniform" — small basins gain least),
   plus the GRACE-only resMLP survivor. The paper's claim is "cross-basin information exists but is invisible
   to standard linear methods and to naive deep-model inputs; here is the delivery that
   finds it, here is the control proving it is the information not the architecture, and
   here is the stratification showing it is not the leakage artifact."
2. **Promote the random-graph placebo to primary control; demote `lstmres_own` to diagnostic.**
3. **Stack stays labelled EXPLORATORY** (architecture chosen after inspecting these folds; no
   untouched evaluation period exists).
4. **Ensemble reporting is mandatory everywhere.** Never a single seed as a headline.
5. Keep: "not detectably different" (not "tie") at lead 2; LOFO described as cross-fold
   stability, not an operational recipe; Newey–West + Künsch citations.

## 10. Known limitations that MUST remain
- Stage-2 residuals are in-sample (out-of-fold is the ranked next experiment).
- 20 placebo draws are reused across folds **and** horizons → the six leads are **not**
  independent events.
- Post-selection inference on the stack.
- The `contamination` metric treats the union of all 284 mask units as "land", so a coastal
  tile's ocean fraction is not counted as foreign — island/coastal basins score LOW; the
  metric is a lower bound on footprint sharing. Linear 2×2 is h1 + linear arm;
  stacked stratification covers h1–6 but reuses the same test months as the headline.
- Kalman `r` not identified as measurement noise.
- One mascon product (CSR); post-2019 test era only; 2 LSTM seeds / 3 resMLP seeds.

## 11. Numbers NOT yet final (reruns in flight, 2026-08-15)
IAAFT surrogates, jump screen, phase-5 fusion/coupled/nonlinear, phase-6 basin analysis
(Africa, LOFO, geography, RF covariates), phase-6 hybrid splice. **Leave explicit
`\todo{RERUN}` markers for these; do not carry pre-audit values forward.**

---

## 10. Phase-8 corrected rerun, leads 1--3 (landed 2026-08-16 01:41). h4--6 PENDING.

Source: `results/phase8_lstm_combined_headline.csv` + `_summary.csv`, seed-matched
per-cell placebo draws. **Read this section before touching the delivery framing.**

### 10a. The headline survives the placebo-seed repair, unchanged

`lstmres_corr_top1` vs `lstm_own_era5`, per seed at h1/h2/h3:
s0 **+0.64 / +1.33 / +1.29**, s1 **+1.19 / +1.56 / +1.36**, every p <= 1.8e-5.
Seed mean +0.92 / +1.45 / +1.32 -- matches the quoted ensemble +0.91/+1.45/+1.33.
Placebos: **20/20 beaten in all six (seed x lead) cells**, p_rank 0.0476. The claim is intact.

### 10b. In-sample stage-2 caveat resolves FAVOURABLY -- rewrite that limitation

`lstmres_oof_corr_top1` (out-of-fold stage-2 residuals) vs `lstm_own_era5`:
s0 +0.91 / +1.64 / +1.57, s1 +1.40 / +1.69 / +1.47, all p <= 4.3e-12; 20/20 placebos.
Direct OOF vs in-sample: **+0.27/+0.32/+0.29 (s0), +0.22/+0.13/+0.11 (s1)**, significant
in 5 of 6 cells. The correction is BIGGER once the in-sample-residual mechanism is removed,
exactly the direction the limitation predicted. So the in-sample design was CONSERVATIVE and
the headline is if anything an underestimate. Delete the hedge; cite the OOF number.

### 10c. The representation-matched arm CONTRADICTS the pure "delivery" reading

This is the important one. All three arms scored against the same baseline (`lstm_own_era5`):

| arm | representation | delivery | h1 | h2 | h3 |
|---|---|---|---|---|---|
| `lstmres_corr_top1` | propagated **scalar** | correction stage | +0.64\* / +1.19\* | +1.33\* / +1.56\* | +1.29\* / +1.36\* |
| `lstmres_corr_top1_hist12` | **12-month history** | correction stage | -0.78\* / +0.10 | -0.18 / +0.13 | -1.03 / -0.44 |
| `lstm_corr_top1_era5` | **12-month history** | input channel | +0.31 | -1.45\* | -1.06\* |

(s0 / s1 where both exist; \* p<0.05.)

Direct contrast, holding delivery fixed at "correction stage" and varying only representation:
`hist12` vs scalar = **-1.43 / -1.53 / -2.35 (s0)** and **-1.10 / -1.46 / -1.82 (s1)**,
every p <= 1.5e-3. Placebos for hist12 are erratic (s0: 18/20, 20/20, **2/20**).

**Implication:** once representation is equalized, the correction stage stops outperforming the
input channel -- both 12-month-history arms are ns-to-negative. The separating variable is the
**propagated scalar representation**, NOT the correction-stage delivery. The manuscript's
disclosed representation confound is therefore REAL and material, and any wording that reads
"the same information delivered as a correction works, as an input channel does not" is no
longer supportable as stated. Recommended reframing: the neighbour signal helps only when it
arrives as a single rho-propagated state aligned to the forecast target; handed over as raw
history it is not usable by either architecture.

**What still holds:** the information is genuinely the neighbour's, not stage-2 capacity.
Neighbour-free control `lstmres_own` vs `lstm_own_era5` = -1.03/-1.45/-1.84 (s0),
-1.30/-1.45/-1.98 (s1), all significant NEGATIVE; the OOF twin `lstmres_oof_own` likewise
(-1.04/-0.93/-1.59, -1.24/-1.40/-1.83). A second stage with no neighbour information actively
hurts, so the gain is not a free-parameter artifact.

**Do not finalize the reframing until h4--6 land** (`phase8b_lstm_h46`, running).

### 10d. Leads 4--6 confirm; two-seed ensemble, all six leads (2026-08-16)

Source: `results/phase8b_h16_ensemble_headline.csv` (hist12/oof ensemble pairs added to
`run_phase8b_merge.py` in this batch so they are quoted on the same footing as the headline).

| contrast | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| scalar correction vs baseline | **+0.91\*** | **+1.45\*** | **+1.33\*** | **+1.32\*** | **+1.42\*** | **+1.96\*** |
| hist12 correction vs baseline | +0.05 | +0.35 | −0.42 | −0.88 | −0.78 | +0.25 |
| input channel vs baseline | +0.36 | −0.42\* | −0.43 | −0.61\* | −0.37 | −0.38 |
| hist12 vs scalar correction | −0.87\* | −1.11\* | −1.78\* | −2.24\* | −2.23\* | −1.74\* |
| **OOF** scalar vs baseline | **+1.14\*** | **+1.70\*** | **+1.54\*** | **+1.66\*** | **+1.62\*** | **+2.28\*** |
| OOF vs in-sample scalar | +0.24\* | +0.25\* | +0.21\* | +0.34\* | +0.20\* | +0.33\* |
| no-neighbour control vs baseline | −1.16\* | −1.46\* | −1.90\* | −2.12\* | −1.84\* | −1.80\* |

The two raw-history arms are indistinguishable from each other and from zero at every lead;
only the propagated scalar works. hist12-vs-scalar is significant in **12/12 seed×lead cells**.

**Framing changes MADE in main.tex on this evidence (2026-08-16):**
- Sect. title "Delivery decides…" → **"Representation decides: the neighbor is usable only as
  a propagated state"**; contribution-3 bullet and intro sentence reworded to match; abstract
  now says the propagation, not the correction stage, is what makes the neighbour usable.
- **Gradient starvation (Pezeshki 2021) demoted**, not deleted. It predicts a dedicated stage
  with its own objective should rescue the weak feature; the hist12 arm HAS its own objective
  and nothing else to fit and still recovers nothing, so training dynamics cannot be the
  binding constraint. Now written as "may contribute to the input channel's instability but
  cannot explain a failure that survives the separation of objectives."
- New positive claim, and it links contribution 3 back to contribution 1: rho^h x_j(t) is the
  neighbour carried forward by its own estimated dynamics, i.e. temporally aligned with the
  target. Recovering that alignment from 12 raw monthly values is itself a learning problem
  and is not solved at this sample size. **The filter does not only clean the target; it also
  puts cross-basin information into the only form our models can use.** Framed as a claim
  about sample size and inductive bias, NOT about optimization pathology.
- **In-sample stage-2 limitation REWRITTEN as resolved-and-favourable** (OOF is larger at all
  six leads); **shared-placebo-draws limitation DELETED** (draws are now per fold×horizon),
  replaced by a narrower note that the rank statistic is pinned at its 1/21 floor.

**What did NOT change:** the headline +0.91..+1.96 numbers, the 20/20 placebo claim, the
stratification/leakage result, and the two-sided linear-vs-nonlinear structure. The finding is
the same size; only its attributed cause moved.
