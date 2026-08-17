# Run Log

One entry per experiment run: what was launched, why, and the key numbers once it finished.
Earlier phases (0-4) predate this log; their commands and outcomes are recorded in the plan file
and in `context_global_study.md`.

---

## 2026-08-12 — ERA5-Land download complete

- Command: `scripts/download_era5.py` (background, resumed after licence acceptance)
- Outcome: 26 files, 2001–2026 (2026 = Jan–May), 590 MB total in `data/raw/era5/`.
  Server-side 0.5° regridding honored. Not yet ingested by anything.

## 2026-08-12 — phase5_predlag: predictive-lag graphs on the Kalman backbone

- Command: `scripts/run_phase3b_kalman_neighbors.py --cells pred_lag1:1,2 --seeds 50 --tag phase5_predlag`
- Why: Phase 3 tested pred_lag graphs only on the ridge backbone; the audit flagged that they
  were never carried to the Kalman backbone where the headline result lives.
- Outputs: `phase5_predlag_{predictions,placebo_monthly,placebo_basin,summary}.csv`
- Key numbers: decisively negative. pred_lag1_top1 loses to kalman_own_ridge at every horizon
  (h1 skill −1.19%) and is beaten by 50/50 random placebo graphs (p_rank = 1.0) at h1–6.
  Worse than random = the lag-1 selection step itself overfits train noise. Correlation
  selection remains the only graph kind that carries signal on the Kalman backbone.

## 2026-08-12 — phase5_fusion: neighbor-fused Kalman filter (novel model candidate)

- Command: `scripts/run_phase5_fusion.py --seeds 50`
- Why: the Phase 4 mechanism finding says the neighbor is a second noisy sensor of the same
  regional signal. This model fuses that sensor inside the filter (y_j = c·x_i + noise, c and
  r_j fit on train only) instead of bolting it on with ridge afterwards. Same information set
  as Phase 3b. Closed-form fit is primary (placebos refit identically); MLE polish is a
  real-arm sensitivity.
- Outputs: `phase5_fusion_{predictions,placebo_monthly,placebo_basin,summary}.csv`
- Key numbers: negative, informatively so. Fusion loses to plain kalman_ar1 at every horizon
  (h1 −1.83%, DM p=3.5e-5) and to 50/50 random-neighbor fusions at h1–h2/h5–h6 (46–49/50 at
  h3–h4); the MLE fit is worse still
  (h1 −5.60%). Diagnosis: the model assumes y_j = c·x_i + white noise, but the neighbor's
  deviation from c·x_i is persistent (its own local signal), so the filter over-trusts it.
  Sharing is partial, not total — motivates the bivariate coupled filter below.

## 2026-08-12 — phase5_coupled: bivariate coupled Kalman filter (novel model, iteration 2)

- Command: `scripts/run_phase5_coupled.py --seeds 50`
- Why: the single-latent fusion's failure mode says the neighbor has persistent local signal,
  so the right structure is two latent states coupled through correlated process noise —
  regional shocks hit both, everything local stays separate. One new parameter (c, joint MLE
  on train only); at c = 0 it reduces exactly to the own filter. Hand-coded 2x2 filter
  verified against a numpy matrix reference to 1e-9 (states and loglik) including NaN-gap
  patterns; c recovery confirmed on synthetic shared-shock pairs (0.72 fit vs 0.69 true).
- Outputs: `phase5_coupled_{predictions,placebo_monthly,placebo_basin,summary,coupling}.csv`
- Key numbers: architecture works, no metric boost. Beats kalman_ar1 at all six horizons
  (h1 +0.24%) and beats 50/50 placebos at h1/h3/h4/h6 (p_rank .0196), 45/50 (ns) at h2/h5 —
  but the gain is never DM-significant vs the own filter (h1 p=.56) and stays below the
  ridge bolt-on's +0.49%. Audit correction: phase3b corr_top1 ALSO beats 50/50 placebos at
  h1–h3 (dying at h4–h5), so the honest contrast is only that coupled keeps a non-significant
  placebo edge at h4–h6 where corr_top1 loses it — and coupled is significantly worse than
  own_ridge at h4–h6 (−1.8 to −2.0%). Headline treatment remains kalman_corr_top1.

## 2026-08-12 — phase5_nonlinear: GBM + MLP heads and a 2-hop arm on the Kalman backbone

- Command: `scripts/run_phase5_nonlinear.py --placebo 20`
- Why: all Phase 3b arms were linear, so the null was only "no linearly usable information."
  GBM and MLP heads on the same features test nonlinearity; the corr_top1 2-hop chain tests
  multi-hop propagation (the GNN question) in feature form. Placebos zero-pad exactly where
  the real 2-hop chain does, keeping capacity matched. Horizons 1–3 only.
- Outputs: `phase5_nonlinear_{predictions,placebo_monthly,summary}.csv`
- Key numbers: the linear null is now a general null. GBM and MLP significantly underperform
  their ridge twins on identical features at every horizon (DM p ≤ 3e-4); ridge_corr_top1
  stays best overall. The 2-hop chain is slightly worse than 1-hop everywhere (linear and
  GBM) — no multi-hop signal to propagate, answering the GNN question in feature form.
  Robustness bonus: at h1 every real-graph nonlinear arm still beats 20/20 placebo twins,
  so the neighbor signal exists across function classes — it is just small, linear, one-hop.

## 2026-08-12 — phase5_stats: consolidated significance table

- Command: `scripts/run_phase5_stats.py` (run after the three experiments above)
- Why: every headline comparison (skill, block-bootstrap CI, DM/HLN p) in one table, computed
  on strictly matched row sets across runs; per-basin DM + FDR at h1.
## 2026-08-12 — phase5 audit (subagent)

- Report: `results/phase5_audit.md`. 13/16 numeric claims verified digit-for-digit from raw
  CSVs; 0 leakage, 0 unfair comparisons, 0 join misalignments; kalman_ar1 rows bitwise
  identical across all four runs. Three overstatements in this log corrected in place (marked
  above). Write-up disclosures the audit requires: (1) everything in Phase 5 is exploratory
  relative to the registered Phase 3b headline — coupled's p_rank .0196 is the 50-placebo
  floor on correlated tests; (2) ~48% of basins have no distinct 2-hop node, and the 2-hop
  placebo randomizes both hops, so it tests the joint graph rather than the 2-hop increment
  in isolation; (3) fusion's MLE variant was ranked against closed-form placebos
  (conservative direction only).

## 2026-08-12 — phase5_stats: consolidated significance table (result)

- Outputs: `phase5_headline_table.csv`, `phase5_perbasin_fdr_h1.csv`
- Key numbers: headline reproduced exactly on matched rows — kalman_corr_top1 vs
  kalman_own_ridge at h1: +0.4879%, CI (+0.14, +0.90), DM p=.0218; 132/234 basins favor it,
  16 pass FDR q=0.10. Every other arm ranked below it at h1 (see entries above). Caution for
  the write-up: at h2 coupled shows +0.92% vs own_ridge (p=.007), but own_ridge is WORSE than
  the plain filter at h2, so the strongest own-only comparator there is kalman_ar1 and vs it
  coupled is +0.03% ns — same comparator trap the audit flagged for corr_top1 at h2. At h4–6
  every apparent corr_top1 gain over kalman_ar1 is the own-state ridge correction resurfacing
  (vs own_ridge it is ~0 ns).

## 2026-08-12 — phase5 analysis (subagent)

- Report: `results/phase5_analysis.md`. New findings from data nobody had examined:
  (1) The 16 FDR basins split 9 helped / 7 HURT — never report "16 pass FDR" as if all
  supportive. The hurt 7 are large rivers / high-latitude coasts (Yenisey hurt under all
  three architectures). (2) Africa: 4 of the 9 winners, pooled h1 skill +1.32%
  (CI +0.38..+2.19) — ~3x the global effect; gains concentrate in high-SNR (+5.2%) and
  glaciated (+1.7%) strata. (3) Fitted coupling c is a stable basin property (median 0.78,
  230/234 sign-stable across folds) but only weakly predicts where skill lives (r=+0.13) —
  descriptive map only. (4) Cross-architecture concordance: per-basin DM stats correlate at
  Spearman 0.68–0.82 across the three neighbor architectures — where the neighbor helps is a
  property of the basins, not the model. Ranked next steps in the report; experiments phase
  closed.

## 2026-08-12 — phase6: Li & Kusche (2026) head-to-head (result, audited)

- Scripts: `scripts/build_li_basin_series.py`, `scripts/run_phase6_li_comparison.py`
- Inputs: PANGAEA 973113 CSR-FCast.zip (md5 verified, `data/raw/li2026/`), 174 init files
  (2009-12..2024-05), 1° grid, leads 1-12. Aggregated to our 234 basins by exact 0.25°→1°
  cell-containment weights (`data/processed/li2026_csr_basin_forecasts.csv`); 7 island
  basins have zero Li land coverage and drop → 227 matched basins.
- Design: Li forecasts moved into our fold-specific deseasonalized standardized space
  (our climatology + per-(fold,basin,horizon) train-window mean offset, both pre-test only);
  matched sample = identical (name, target_date) rows for all 9 models per horizon;
  61 test months (2019-06..2024-11), folds f1-f4 only (Li hindcast ends init 2024-05).
- Outputs: `phase6_li_comparison_{predictions,summary,headline,perbasin}.csv`
- HEADLINE: crossing pattern. h1: kalman_corr_top1 BEATS li_lstm_full (Li skill vs us
  −16.9%, DM p=.004; li_lstm_nonseas −34%; Li h1 nonseas is worse than raw persistence —
  their LSTM takes no GRACE lags as input, only exogenous hydromet predictors). h2: ~tie
  (+7.6%, p=.075). h3-6: Li wins big and growing (+13.8/+16.1/+18.8/+22.1%, p≤.001;
  165-170/227 basins; skill_vs_damped 0.22-0.31 vs our 0.10-0.14). Same conclusions in the
  coverage≥0.5 subset and for the nonseasonal-only variant (which wins h3-6 by ~0.02-0.08
  less than full — the increment is their monthly re-extrapolated seasonal/trend vs our
  fold-frozen climatology).
- Audit (subagent, PASS all items): spatial aggregation verified by independent recompute
  (exact match incl. NaN renormalization); time alignment verified incl. December rollover;
  no leakage (all transforms pre-test; 83,082 matched keys × exactly 9 models each, target
  identical across models within key); skill_vs_damped reproduced from raw rows. Staleness
  check: Li's edge does NOT grow with months-since-fold-start (h3 ρ=.14 p=.61) — expanding
  window refits unlikely to explain h3-6. Caveats to carry: their training uses Yin 2023
  gap reconstruction (hindsight channel, likely minor); month sets differ slightly by
  horizon at window edges; our h1 win partly reflects Kalman being tuned to exactly this
  target definition.

## 2026-08-12 — era5 ingestion: basin-month forcing table

- Command: `scripts/build_era5_basin_table.py`
- Why: ERA5-Land (11 variables, 0.5°, 2001-01..2026-05) was downloaded but never ingested.
  Aggregates to basin level with the same cos-lat, NaN-aware weighted means as the CSR
  aggregation; 0.25° mask cells map to nearest 0.5° ERA5 cell. Unit traps handled:
  accumulated vars (tp, e, ro, sro, ssro) are mean daily rates in m/day → ×days-in-month
  ×100 → cm/month; evaporation sign flipped (positive = loss); t2m → °C; sd → cm WE.
- Outputs: `data/processed/era5_basin_month.csv` (284 basins × 305 months, gapless),
  `data/processed/era5_basin_coverage.csv`
- Key numbers: single-basin manual recomputation matches to float32 precision (Amazon
  2010-06 precip 11.9715 cm). 13 keep-basins have <50% land coverage (island/archipelago
  slivers; Nusa_Tenggara .25 lowest, Sulawesi .49 highest) — the code audit
  (`phase6_era5_audit.md`, all checks PASS) undercounted this as 5; corrected per
  phase6_analysis.md. Screening all 13 changes nothing material (+5.42% ERA5 gain,
  +0.506% neighbor).

## 2026-08-12 — phase6_era5: ERA5 exogenous features on the Kalman backbone

- Command: `scripts/run_phase6_era5.py --placebo 20`
- Why: all prior phases used TWSA-only features, so "GBM/MLP lose to ridge" had a
  possible excuse (features too thin) and the neighbor effect had an unclosed confound
  (shared local meteorology, only partially controlled by ENSO/IOD indices). ERA5 lag
  features (11 vars × lags 0,1,2, deseasonalized + standardized on the train window per
  fold, winsorized at ±10σ) are appended to own/neighbor arms; placebos randomize ONLY
  the neighbor with ERA5 held fixed. All arms share identical row sets.
- FIRST RUN VOID: an initial run (results overwritten) had ridge/MLP+ERA5 arms explode
  (pooled RMSE 12–26) — E_Libyian_Desert subsurface runoff is ~0 for every train window,
  so the 2023/24 Libya flood months standardized to ~45,000–52,000σ under f1–f4 train
  stats (the flood months fall in f4's test window; audit-corrected);
  tree-based GBM was immune, linear heads detonated. Fix: winsorize standardized ERA5
  features at ±10σ (`era5.py CLIP_SIGMA`, constant causal transform, train and test
  alike). Smoke fold f1 missed it because its test window predates the event.
- Outputs: `phase6_era5_{predictions,placebo_monthly,summary,headline}.csv`,
  log `phase6_era5_run.log`
- Key numbers (pooled, std units, n=19,656/arm/horizon):
  (1) ERA5 is the largest skill source found in this project: ridge_own_era5 vs
  ridge_own +5.17% at h1 (CI +2.94..+7.18, DM p<1e-5), +1.81% at h2 (p=.0099), +0.65%
  ns at h3. ~10× the neighbor effect.
  (2) Nonlinear heads still do NOT beat ridge on identical enriched features: GBM goes
  from significantly worse (no-ERA5 twins, DM p≤4.4e-5) to a statistical tie (h1 p=.86,
  h3 p=.91); suggestive-only edge at h2 (−1.87, p=.065). MLP remains seed-unstable,
  2/3 seeds tie at h1, all worse at h3. "Ridge is enough" survives feature enrichment.
  (3) NEW CONTROL, paper-grade: the neighbor effect survives full local-meteorology
  conditioning — ridge_corr_top1_era5 vs ridge_own_era5 at h1 +0.485% (CI +0.15..+0.86,
  DM p=.0096), essentially identical to the unconditioned +0.488%, and 20/20 placebos
  beaten (p_rank=.0476, the 20-seed floor). The neighbor is NOT shared weather; it adds
  TWSA-specific spatial information, consistent with the spatial-denoising reading.
  (4) Best overall h1 models: gbm_corr_top1_era5 and mlp_corr_top1_era5_s1 at RMSE
  ≈1.0150 (+5.70%), ridge_corr_top1_era5 at 1.0154 (+5.63%) — tie holds only for the
  best MLP seed (s2 = 1.0249); audit-corrected. Neighbor increment at h2–h3 stays ns, as before.

## 2026-08-12 — Phase 6: hybrid splice (ours h1-2 + Li LSTM h3-6)
- Command: `scripts/run_phase6_hybrid.py`
- Why: quantify the complementarity claim from the Li head-to-head — if short leads
  are a filtering problem (we win h1) and long leads need exogenous forcing (Li wins
  h3-6), a best-of-both splice should beat either alone. Same matched sample as the
  head-to-head (227 basins × 61 test months per horizon), same skill-vs-damped
  framework; ERA5 arm merged onto matched keys (target consistency asserted <1e-6,
  ERA5 run only covers h1-3 so its solo pooled row is omitted).
- Hybrids: {kalman_corr_top1, ridge_corr_top1_era5} at short leads + li_lstm_full at
  long leads; primary splice h1-2/h3-6 per plan, _h1 variants splice h1/h2-6 because
  the head-to-head h2 "tie" nominally favors Li (li_full h2 skill .141 vs our .070-.080).
- Outputs: `phase6_hybrid_{summary,headline}.csv`
- Key numbers (pooled all-horizon skill vs damped persistence; row-pooled, so months
  with all 6 horizons weigh more — equal-weight-month pooling shifts levels ~1pt,
  e.g. era5_li_h1 22.6% not 23.7%, li_full 20.2% not 21.6%, rankings unchanged):
  kalman_corr_top1 alone +10.4%, li_lstm_full alone +21.6%, hybrid_kalman_li +22.3%,
  hybrid_era5_li +22.9%, hybrid_era5_li_h1 +23.7%.
  Hybrid vs our model alone: +13.3% (DM p=.001) — decisive.
  Hybrid vs Li alone: pooled gain small — h1-2 splice +0.9% ns (p=.60) because the
  pooled metric is dominated by long-lead losses AND the h2 leg gives back −8.3%
  (p=.075); h1-only splice +2.2% (p=.063) kalman / +2.7% (p=.047) era5. The honest
  claim: complementarity is real but concentrated at h1 (+14.5%/+18.0% over Li there,
  p≤.004); spliced products beat each component at its weak leads, and the pooled
  gain over Li alone is modest because Li is only clearly beaten at one of six leads.
- Selection caveats (audit): splice points were chosen on the SAME 61-month test
  sample the hybrids are scored on, and "+23.7%" is the best of 4 post-hoc variants
  with no multiplicity control — treat marginal pooled gains over Li (p=.047/.063)
  as optimistic; the pre-planned h1-2 splice is the primary result. Pooled "all" DMs
  are descriptive (differential is exactly 0 at shared horizons, diluting ~6:1, and
  10/66 edge months have partial horizon mix); per-horizon DMs are the evidence.
- Audit (subagent): PASS, no blockers — splice RMSEs reproduce to <1e-12, matched
  sample intact (no merge fan-out), targets shared across input files to 9e-16,
  skill convention matches upstream. Fixed post-audit: h3 ERA5 coverage guard,
  HAC-lag comment (Newey-West max_lag=5, not 6), numeric sort, and dropped
  trivially-identical hybrid-vs-component headline rows (skill≡0 noise).

## 2026-08-12 — phase6_era5 results audit (subagent)

- Report: `results/phase6_era5_results_audit.md`. ALL SIX CHECKS PASS: all 63
  (model,horizon) RMSE cells reproduce to machine precision; n≡19,656 with identical
  row-set hashes across all 21 models per horizon; every headline DM/bootstrap number
  reproduces exactly; placebo accounting exact (4 families × 20 seeds, p_rank floor
  1/21 confirmed); kalman_ar1 bitwise identical to phase3b/coupled/predlag/fusion, and
  all six no-ERA5 twin arms bitwise identical to phase5_nonlinear (ERA5 dropped zero
  rows). Winsorization verified causal; clip footprint ≤0.064% of feature values;
  Libya train std 4.4–5.0e-8 → 45k–52kσ unclipped, and the 1e-8 zero-guard does NOT
  catch it (5× above threshold) — the clip is the operative fix. Three LOW wording
  errors in the entry above corrected in place (marked audit-corrected). Disclosures:
  the voided first-run RMSEs (12–26) are unreproducible (files overwritten); the 1e-8
  std guard is unit-dependent; 28 near-guard series act as bounded ±10 event flags
  rather than Gaussian anomalies.

## 2026-08-12 — phase6 analysis (subagent)

- Report: `results/phase6_analysis.md` + 5 derived CSVs (`phase6_perbasin_era5_skill_h1`,
  `phase6_perbasin_fdr_era5_h1`, `phase6_perbasin_fdr_neighbor_{era5,noera5}_h1`,
  `phase6_era5_vargroup_ablation_h1`). Fold-ridge refits reproduce archived per-fold
  skills exactly — ablations are engine-exact.
- Key findings: (1) ERA5's geography is the MIRROR IMAGE of the neighbor's: h1 gains
  NA +7.8% / Europe +7.3% / Asia +6.4%, but Africa +0.30% ns (Congo −66%, Lower Lake
  Chad −32%) and Amazon −99% (worst basin); gains live in low-SNR strata, high-SNR
  stratum hurt −24.7% — opposite the neighbor. Per-basin ERA5 and neighbor gains
  anti-correlate (rho −0.22, p=.0007): the two sources cover each other's dead zones.
  (2) Per-basin FDR q=.10: 27 helped / 3 hurt — far cleaner than the neighbor's 9/7.
  (3) Variable ablation: soil moisture carries ~half the gain (−2.5pp when dropped,
  +3.1% alone), precip ~a quarter, temperature and snow nothing. (4) Fold-stable
  except f3 (2022-04..2023-08, −0.9% vs +5..+9% elsewhere), traced to ERA5 precip
  misinformation in the La Niña tropics (dropping precip turns f3 to +3.6%). Libya
  winsorization held. (5) Neighbor geography invariant to conditioning (per-basin DM
  rho .857; all 16 phase-5 FDR basins keep sign, set grows to 24 = 14 helped/10 hurt);
  Yenisey still significantly hurt by the neighbor (p=.001) yet is ERA5's #2 winner
  (+32%) — local forcing delivers what the neighbor imports as noise. Africa's
  neighbor effect STRENGTHENS under conditioning: +1.91% (p=.0002) vs +1.32%.
  (6) MLP seed spread triples with ERA5 (1.84pp on the fullest arm).
- Write-up implications ranked in the report; the ERA5/neighbor complementarity
  (mirror geography + anti-correlation) is the strongest new paper material.

## 2026-08-12 — phase6 basin analysis: who benefits from neighbors (result)

- Command: `scripts/run_phase6_basin_analysis.py`
- Why: per-basin synthesis across phase3b/phase4/phase6_era5 + Li per-basin DM — what predicts
  neighbor benefit, is selective application better, geographic pattern, does ERA5 change who wins.
- Outputs: `phase6_basin_analysis_summary.csv` (234-basin covariate+skill table),
  `phase6_basin_analysis.md` (report), `_spearman/_rf_importance/_selective/_q4.csv`
- Key numbers: best predictors of h1 benefit are neighbor stability across folds (ρ=-0.27;
  stable +1.18% vs unstable -1.33% median) and basin area (ρ=+0.28; largest tercile -1.37%,
  great rivers hurt; 42% of large basins pick a >1000 km spurious neighbor). RF on all
  covariates: CV R² only 0.02-0.04 — static covariates predict poorly. Honest LOFO selective
  application (neighbor only where it helped in other folds) lifts pooled h1 skill
  +0.49% → +1.13% (DM p<1e-4, ~132 basins), 64% of oracle (+1.77%); same on ERA5-conditioned
  backbone (+1.09%). Africa median +1.43% (+2.94% ERA5-conditioned); Europe/Australia negative.
  ERA5 conditioning leaves ranks intact (ρ=0.857, flips only near-zero basins; phase4 index
  conditioning ρ=0.997). Li h4 edge uncorrelated with neighbor benefit (ρ=-0.14) — complementary
  mechanisms. Replication: phase6_era5 no-ERA5 ridge arms == phase3b arms exactly (max DM diff 0).
- Audit (subagent, PASS all 7 items): per-basin skill/DM reproduced to full precision on spot
  checks; graph reconstruction 234/234 match vs experiments' fold graphs; LOFO selection verified
  leakage-free by independent reimplementation (+1.13314% exact match, DM p=1.5e-9); replication
  claim verified at raw-prediction level (19,656/19,656 rows, max |pred diff| 0.0); phase4 file
  confirmed to be the genuinely conditioned run (preds differ from phase3b, max 0.105). Two
  last-digit rounding nits in the report fixed (-1.02→-1.01%, -0.32→-0.31%); caveat added that
  RF-selection covariates aggregate across folds (structural, not target leakage).

## 2026-08-12/13 — Phase 7: three new architectures on the Kalman backbone

- Commands: `scripts/run_phase7_resmlp.py --placebo 20` → `run_phase7_gnn.py` →
  `run_phase7_lstm.py` (sequential background chain, ~10.5 h total). New engines:
  `src/gracefc/experiment_{resmlp,lstm,gnn}.py` + shared `src/gracefc/phase7.py`.
  PyTorch CPU installed into .venv for the LSTM/GNN.
- Why: import the Africa-study winner (ridge + neighbor-only residual MLP) into the
  global Kalman framework, and give the "ridge is enough" conclusion two stronger
  challengers (sequence memory via LSTM; learned graph aggregation via 1-layer GAT).
  Same 5 folds, horizons 1-3, identical row sets (ERA5-complete, n=19,656/arm/horizon),
  20 degree-matched random-graph placebos, ridge twins re-emitted per experiment and
  verified bit-identical to phase 6 (code audit PASS, no blockers; audit notes: placebo
  draws shared across arm families; torch models train on ~85% of train months with no
  refit after early stop — conservative vs ridge; resmlp keeps sklearn random-split
  early stopping to stay twin-comparable with phase 5/6 MLP; GNN _era5 messages carry
  the NEIGHBOR's ERA5, which its flat ridge twin never sees).
- Outputs: `phase7_{resmlp,lstm,gnn}_{predictions,placebo_monthly,summary,headline}.csv`,
  logs `phase7_*_run.log`; smoke twins with `_smoke` tags. Stale pre-winsorization
  `phase6_era5_smoke_*.csv` deleted per audit.
- Key numbers, resmlp (arch 1): FIRST architecture to beat ridge on identical features —
  resmlp_corr_top1_s0 vs ridge_corr_top1 +0.50/+1.45/+1.74% at h1/2/3 (DM p=.015/5e-11/1e-9),
  vs ridge_own +0.99/+1.61/+1.85% (all p<1e-6); every no-ERA5 seed beats 20/20 placebos at
  every horizon (p_rank=.0476 floor). Second neighbor adds nothing (top2 vs top1 ns; h3
  −0.42% p=.03 against). With ERA5 aboard: still beats its ridge twin at h2-3 (p<.001) but
  the neighbor increment vanishes (vs resmlp_own_era5 ns at all horizons; ERA5 placebo
  ranks decay to 2-17/20) and seed spread widens beyond the increment.
- Key numbers, LSTM (arch 2): the big surprise — sequence modeling of ERA5 forcing gives
  the first decisive nonlinear win: lstm_own_era5_s0 vs ridge_own_era5 +2.05/+2.24/+1.60%
  (DM p≤1.5e-7 at all horizons; seed s1 concordant), and lstm_corr_top1_era5 beats
  ridge_corr_top1_era5 +1.61/+1.37/+0.82% (p≤.006). GRACE-only: lstm_own ≈ ridge_own
  (h1 tie, h2 +0.82% p=1e-4, h3 −0.62%); neighbor channel helps only at h1 (+0.77% vs
  lstm_own p=1.4e-4, 20/20 placebos) and is dead or harmful at h2-3 (placebo ranks 0-7/20;
  with ERA5 the neighbor is actively worse, −0.78/−0.73% at h2/3, p<.004).
- Key numbers, GNN (arch 3): never beats a ridge twin anywhere (best case ns; h1 GRACE-only
  significantly worse, p≤4e-4); no-ERA5 arms sit BELOW ridge_own (−0.1..−0.8% skill) while
  still beating 20/20 random graphs at h1 — the graph signal survives inside the GNN but
  the architecture costs more than the neighbor adds. Graph structure beyond top-1 does not
  help: top2 vs top1 negative wherever significant (h2 −0.92% p=9e-4; era5 h1 −1.22% p=.027,
  h3 −0.80% p=.006). Seed spread (e.g. h1 era5 1.0161→1.0307) exceeds the neighbor effect.
  Closes the graph-architecture question in agreement with phase 5: small/linear/one-hop.

## 2026-08-13 — Phase 8: stacked LSTM + neighbor-only residual MLP (the two Phase 7 winners combined)

- Command: `scripts/run_phase8_lstm_combined.py --placebo 20` (~3 h). New engine
  `src/gracefc/experiment_lstm_combined.py`; `experiment_lstm.py` refactored
  (`fit_lstm` split into `train_lstm` + `lstm_predict`, behavior unchanged — smoke
  verified bit-identical on all 7 shared arms vs `phase7_lstm_predictions.csv`, and
  the code audit re-verified independently, 27,846 rows max |diff| = 0.0).
- Why: the requested "LSTM with both Kalman states AND ERA5" experiment already
  existed (Phase 7 lstm_own_era5 / lstm_corr_top1_era5 / lstm_own ARE those arms —
  not duplicated). The new question: does the resMLP neighbor-only correction still
  add skill when stage 1 is the LSTM that already integrates the ERA5 forcing
  history? Stage 1 = LSTM(own filtered-state seq + 11 ERA5 channels), stage 2 =
  sklearn MLP on stage-1 train residual from propagated corr_top1 state only, final
  = kalman + lstm + mlp (`lstmres_corr_top1`). Second family `lstmres_nbrin_*`
  stacks on the literal variant-2 stage 1 (neighbor channel inside the LSTM).
  2 LSTM seeds, 20 placebos randomizing ONLY the stage-2 neighbor (stage-1 seed-0
  nets shared across draws, mirroring resmlp's shared ridge stage).
- Outputs: `phase8_lstm_combined_{predictions,placebo_monthly,summary,headline}.csv`,
  log `phase8_lstm_combined_run.log`, smoke twins `_smoke`, audit
  `phase8_code_audit.md` (PASS WITH NOTES, no blockers).
- Key numbers: the stack WORKS and is the new best model at every horizon —
  lstmres_corr_top1 vs lstm_own_era5 (same stage-1 net, isolates the correction):
  s0 +0.90/+1.12/+1.28% at h1/2/3, s1 +0.90/+0.99/+1.20% (all DM p<=2.0e-6; seed-stable,
  unlike every previous neighbor-under-ERA5 result; ensemble +0.93/+1.09/+1.27%,
  45/45 fold-cells positive). vs ridge_corr_top1_era5:
  +2.46/+3.23/+2.80% (s0, p<1e-7). Placebos: s0 20/20 beaten at ALL horizons; s1
  6/20 (h1), 12/20 (h2), 20/20 (h3) — but s1 fights placebos built on the BETTER
  seed-0 stage-1 (audited asymmetry, conservative); seed-matched placebo increments
  needed before quoting s1 ranks. nbrin family: worse than lstmres at h1-h2 (in-LSTM
  neighbor still costs, confirming Phase 7), stage-2 correction on top of it ns at
  h1-h2, +0.56% p=.006 at h3; nbrin placebo certifies stage-2 increment only.
- Audit caveats to carry: (1) stage-2 residuals are in-sample for the LSTM —
  evaluation honest by construction (shared stage-1 net; MLP sees no test info) but
  the learned correction is "plausibly attenuated," NOT provably conservative;
  (2) NEVER quote summary's auto `dm_vs_ridge_twin` for lstmres arms (pairs against
  no-ERA5 ridge_corr_top1; information-mismatched) — use the runner CONTRASTS;
  (3) placebo draws shared across the two stacked families within a fold.

## 2026-08-13 — Phase 8b: h4-6 extension of the stacked LSTM + h1-6 Li comparison

- Commands: `scripts/run_phase8_lstm_combined.py --horizons 4-6 --tag phase8b_lstm_h46
  --placebo 20` (~80 min, 15/15 fold-cells) then `scripts/run_phase8b_merge.py`.
  Runner gained a `--horizons` arg (default 1-3 = old behavior); engine untouched,
  h1-3 results NOT re-run. Same protocol as phase 8: 5 folds, 2 LSTM seeds, 20
  stage-2 placebos, ridge baselines included.
- Outputs: `phase8b_lstm_h46_{predictions,placebo_monthly,summary,headline}.csv` +
  `_run.log`; merged h1-6 tables `phase8b_h16_{summary,headline}.csv`; Li matched-
  sample comparison `phase8b_li_comparison_{predictions,summary,headline,perbasin}.csv`
  (227 basins x 61 months, phase 6 protocol, targets verified bit-identical);
  merge log `phase8b_merge_run.log`; audit `phase8b_audit.md`.
- Key numbers, h4-6 (full 234-basin sample): lstmres_corr_top1 is the best arm at all
  three horizons, both seeds, and beats 20/20 placebos at every horizon (both seeds).
  Stage-2 increment vs same-seed lstm_own_era5: s0 +1.20/+1.07/+1.29% at h4/5/6,
  s1 +1.01/+0.98/+1.44% (all DM p<=1e-4) — the neighbor correction is horizon-STABLE
  (~+1% at every h1-6). The stage-1 ERA5-sequence gain is NOT: lstm_own_era5 vs
  ridge_own_era5 +0.78% (p=.0012) at h4 then dead at h5-6 (ns, sign flips); by h5-6
  every learned arm sits within ~1.1% of ridge_own (kalman_ar1 baseline −2.0% at h5)
  — the correction is the only surviving
  architectural gain at long leads. nbrin family again dominated (worse than lstmres
  everywhere; its s1 placebo rank collapses at h5: 0/20).
- Key numbers, Li h1-6 (matched sample): the stack does NOT close the h4-6 gap — it
  widens with lead. lstmres_corr_top1_ens vs li_lstm_full: +20.2% h1 (p=3e-4),
  −3.8% h2 (ns), −12.5% h3, −16.5% h4, −21.6% h5, −26.7% h6 (all p<=.019); per-basin
  wins 133/81/69/65/57/57 of 227. Even li_lstm_nonseas beats us at h4-6. Our stage-2
  correction (~+1%) is an order of magnitude smaller than Li's exogenous-forcing edge
  (~16-27%). Phase 6 framing sharpened, not overturned: filtering wins h1, exogenous
  forcing wins h3-6, and the crossing is now measured against our best architecture,
  not just the Kalman ladder.
- Caveats: phase 8 audit caveats all carry over (in-sample stage-2 residuals;
  dm_vs_ridge_twin column wrong-twin for lstmres arms — use headline contrasts;
  shared placebo draws). Li comparison excludes ALL of f5 (their hindcast ends
  2024-11; f5 test starts 2025-02); full-sample and matched-sample skills are NOT
  interchangeable. Audit `phase8b_audit.md`: PASS WITH NOTES — all headline numbers
  recomputed digit-for-digit from predictions CSVs; matched sample rebuilt
  independently row-for-row identical; kalman_ar1 bit-identical to the phase 6 file
  at all six horizons. New good news: the h1-3 s1-placebo asymmetry does NOT bite at
  h4-6 (stage-1 seed gap ±0.1-0.2%; raw ranks and seed-matched increments agree,
  placebo max increment +0.14% vs real ~+1%). Placebo graphs at h4-6 reuse the same
  per-fold draws as h1-3 (don't sell 20/20-at-six-horizons as independent events);
  >=300km/conditioning controls still unrun for the stacked arm; the
  ensemble≈resmlp_own_era5 equivalence check exists only at h1-3.

## 2026-08-13 — Paper phase: novelty audits, HESS draft, adversarial audit, repo cleanup
- Two novelty audits (main session + independent adversarial subagent, ~25 searches, full
  texts read where reachable). Verdicts: noise/benchmark argument NOVEL (cleanest claim;
  cite Niraula & Goessling 2021 sea-ice analogue — NOTE: memory previously said "Niraula &
  Notz", wrong, CrossRef-corrected); neighbour claim REWORDED to "first isolation+validation
  on observed GRACE" (Steidl & Zhu 2025 NeurIPS-WS basin-graph forecaster exists: reconstruction
  target, no nulls, no isolation — cite proactively); crossing = measurement not concept
  (Wood & Lettenmaier 2008 lineage); residual-delivery = empirical demonstration (cite
  gradient starvation, Pezeshki 2021). New review risk: FLDAS-Forecast hindcast is public —
  manuscript acknowledges + justifies non-comparison. Full detail: memory grace-novelty-audit.md.
- Venue: HESS (open review suits unaffiliated first-time authors; €1800 w/ waiver route;
  fallback J. Hydrology free). Writing-practices agent report informed structure/figures.
- Manuscript written by clean-context subagent (Orchestra ml-paper-writing skill):
  paper/main.tex (copernicus.cls) + references.bib (42 DOI-verified) + FIGURE_PLAN.md (7 figs)
  + DECISIONS.md (35 calls). NEW derived results: scripts/build_paper_ladder.py →
  results/paper_baseline_ladder.csv + paper_baseline_contrasts.csv (ladder recomputed on
  matched phase2∩phase3b rows; corrected headline to kalman-vs-damped +4.8/+8.1% h1/h2,
  vs-ridge +1.2..+3.4% p<=.042).
- Adversarial audit: PASS WITH NOTES. Ladder + crossing + stage-2 numbers reproduced
  digit-for-digit incl. independent DM reimplementation; fold-free match key verified harmless
  (0 dup keys, targets bit-identical). Fixes applied to main.tex: abstract said "time-shift
  surrogates" → IAAFT (MAJOR — control mislabeled); Li init range Apr→May 2024; MLP-with-ERA5
  is significantly WORSE not "tied" (−1.8..−2.5%, p<=.004); no-ERA5 LSTM h2 exception
  (+0.8/+0.9% both seeds) now disclosed; rho −0.22 labeled as vs CONDITIONED neighbour gain
  (unconditioned −0.121 ns) + F6 spec aligned; seed spread 0.93/1.06/0.67%. Upstream fix:
  phase8_related_benchmarks.md §2.2 FLDAS author list was wrong (now Li/Hazra/McNally/
  Slinski/Shukla/Anderson per CrossRef).
- Repo cleanup: context_global_study.md + code_structure.md rewritten to current state
  (were frozen at phase 3b/4); context.md → context_africa_jpl.md, "context1 (1).md" →
  context_africa_csr.md; empty configs/ + src/gracefc/data/ removed; old WRR stub renamed
  results/paper_draft_SUPERSEDED_old_wrr_framing.tex.
- OPEN: figures (specs in FIGURE_PLAN.md), LaTeX toolchain + first compile, authors/Zenodo
  DOI, manual full-text pulls (Niraula & Goessling 2021; Kankanige 2026), CSR release-string
  TODO ("RL0603M" label vs product_version "RL06.2" in file metadata), unresolvable
  J.Hydrology 2025 PII S002216942500890X still \todo in Sect 5.1.

## 2026-08-13 (late) - REPOSITORY AUDIT verified, data/eval defects fixed, paper pipeline rerun started

- Full-repo audit (AUDIT_REPORT.md, root) triaged by independent verification, not taken on faith.
  Verdicts: P0-1 CSR month assignment CONFIRMED (raw file's months_missing attr proves it:
  290 span months - 33 documented missing = 257 solutions exactly; the arcs centered 2011-10-31
  and 2015-04-27 are the Nov-2011 and May-2015 solutions; midpoint binning averaged them into
  Oct/Apr and dropped both months). P0-2 issue-date leakage CONFIRMED structurally (folds split
  on target_date, transforms fit to test_start; leads 2-6 saw up to h-1 post-issue months).
  P0-3 CONFIRMED (kalman_corr_top1 vs kalman_ar1 h1: resolved 199 basins +0.09% p=.71 vs
  sub-resolution 35 basins +1.06% p=.0007; same split under min300). P0-5 CONFIRMED
  digit-for-digit (273/1170 fits r<1e-6; 32 basins all folds; q/r 0.0059..1.1e10).
  P0-6 CONFIRMED digit-for-digit (ERA5-arm input-channel contrast: ens +0.43/-0.12/-0.31% ns;
  s1 POSITIVE h1-2 - "dead or harmful" was seed-0-selective). P0-4 post-selection: process
  fact, remedy = exploratory label.
- FREEZE: all 117 pre-audit result artifacts moved to archive/pre_audit_2026-08-13/results
  with SHA256_MANIFEST.csv; .md analyses/audits + this log stay active. NOTE: the pre-audit
  processed basin table was overwritten in place by the rebuild (regenerable from archived
  code+raw at any time).
- FIXES: (1) basins.py assign_solution_months() - official month sequence from months_missing
  + span-overlap assertions, no duplicate averaging; rebuild gives 257 months (was 255), Nov
  2011 + May 2015 restored. (2) evaluate.split_fold now issue-date membership (model frozen at
  test_start, used forward; all transforms fit < test_start are legal by construction); all
  five split sites (evaluate, experiment_era5, experiment_kalman, experiment_nonlinear,
  phase7) share the one splitter; test-side leakage assertion added. Same issue window across
  horizons now (equal n per lead except record-tail truncation).
- Manuscript claim surgery applied to main.tex (numbers still stale, banner at top says so):
  dead-or-harmful -> seed-unstable everywhere; Kalman language -> filtering-class margin +
  identification caveat paragraph + limitation; resolution stratification added as primary
  neighbor sensitivity (abstract, contributions, controls table row, mechanism paragraph,
  discussion, conclusions); stack labeled exploratory (ranking disclosure + limitation);
  crossing -> "empirical crossing consistent with"; tie -> "not detectably different"; LOFO
  paragraph de-operationalized; issue-date protocol described in Methods; retrospective-skill
  scope note; RL0603M string fixed; indices arm corrected; 45 cells -> 30+15; Newey-West +
  Kunsch citations added to references.bib.
- RERUN started (corrected data + protocol): phase2 DONE (~2.5 min), kalman baseline ->
  phase3b (50 placebo seeds) -> phase6_era5 queued sequentially; torch phases 7/8/8b held for
  a corrected-foundation audit first, per standing audit cadence.

## 2026-08-14 - CORRECTED RERUN, first results (phase2 + kalman + phase3b). MAJOR REFRAMING.

All numbers below are on the corrected basin table (257 months) AND the issue-date fold
protocol. Equal n per lead across all models (19,422 at h1 down to 18,252 at h6).

**Baseline ladder GOT STRONGER.** Kalman vs stronger damped persistence, skill %:
  h1 +6.23 | h2 +11.14 | h3 +9.42 | h4 +8.45 | h5 +9.60 | h6 +12.23
  (pre-audit: +4.8/+8.1/+5.3/+3.2/+2.5/+3.5 - the long-lead margin roughly TRIPLED)
Kalman vs per-basin ridge significant at ALL SIX leads, p = 1.6e-4 / 5.9e-6 / 2.2e-3 /
6.4e-3 / 3.2e-3 / 6.3e-4 (pre-audit max p was .0412 at h4; the audit's fear that h4 would
lose significance was WRONG - it strengthened). Per-basin ridge collapses with lead under
the honest protocol (+3.05 h1 -> -0.29 h6; pre-audit +2.8 -> +0.1). READING: the leakage
was flattering the NON-filtering baselines more than the filter, because they lean harder
on the leaked climatology/scaling; removing it widens the filtering-class margin. The
paper's contribution #1 is stronger, not weaker.

**Neighbor effect LOST SIGNIFICANCE - contribution #3 must be reframed as a null.**
kalman_corr_top1 vs kalman_own_ridge (the registered capacity-matched contrast):
  h1 +0.308% DM p=0.199 (pre-audit +0.488%, p=.0218 SIGNIFICANT)
  h2 -0.097 | h3 -0.192 | h4 -0.358 | h5 -0.349 | h6 -0.321, all ns, all NEGATIVE
Placebo picture is now split and must be stated as two separate questions:
  - "does correlation-selection beat a random neighbor?" YES at h1 (50/50 draws beaten,
    p_rank=.0196; min300 also 50/50)
  - "does adding a neighbor beat not adding one?" NO (DM p=0.199)
  At h2-h6 the real graph is beaten by 0/50 placebos (p_rank=1.0) - i.e. the
  correlation-selected neighbor is WORSE than a random basin at every lead beyond 1.
**Resolution stratification unchanged and decisive** (h1, corrected):
  resolved >=90k km2 (199 basins) +0.099% p=0.715  <- nothing
  sub-resolution <90k (35 basins)  +1.033% p=0.00065 <- the entire effect
  same under min300 (+0.140% p=.61 vs +0.964% p=.00087)
VERDICT: at the linear tier there is no detectable cross-basin effect in basins GRACE can
resolve. The pooled effect was carried by sub-resolution basins and is now not even
pooled-significant. Contribution #3 becomes a controlled NEGATIVE result (+ a caution about
sub-resolution basins), unless the phase-8 stacked correction survives its own rerun -
which should now be treated as doubtful given h2-h6 went negative here.

**Kalman identification caveat stands** (diagnostics now stored): 1170/1170 fits converge,
but 259/1170 hit the r<1e-6 boundary and 31 basins are at the boundary in all five folds.
Language stays "AR(1) state-space filtering", not "measured noise cost".

phase6_era5 running. Torch phases 7/8/8b NOT yet rerun.

### phase6_era5 on corrected data (same session)

**ERA5 forcing gain is now essentially LEAD-1 ONLY.** ridge_own_era5 vs ridge_own:
  h1 +4.62% (CI +1.91..+6.84) p=1.0e-4  <- still the single largest skill source
  h2 +1.39% (CI -0.48..+2.98) p=0.092   <- LOST significance (pre-audit +1.81%, p=.0099)
  h3 +0.33% p=0.61 ns (pre-audit +0.65% ns)
Manuscript claim "ERA5 adds 5.2/1.8% at h1/h2" must become "+4.6% at h1, not significant
beyond".

**Internal consistency check PASSED:** phase6's no-ERA5 arms reproduce phase3b exactly
(ridge_corr_top1 vs ridge_own: +0.3076/-0.0971/-0.1915 at h1-3, DM p .1986/.7179/.4805,
identical to phase3b's kalman_corr_top1 contrast to 4 decimals). Two independent runners,
same numbers - the corrected pipeline is self-consistent.

**Neighbor under ERA5 conditioning: control still informative, effect still null.**
  ridge_corr_top1 vs ridge_own          h1 +0.308% p=0.199
  ridge_corr_top1_era5 vs ridge_own_era5 h1 +0.322% p=0.115
The conditioning control still does its job (the point estimate is unmoved by adding 11
local-meteorology variables, so the neighbor is NOT a shared-weather proxy), but both arms
are non-significant, so the correct statement is "conditioning does not explain away the
effect" NOT "the effect survives conditioning".

**"Nothing beats ridge on identical features" STRENGTHENED.** gbm_own_era5 vs
ridge_own_era5: +0.14 ns / +0.60 ns / -0.71 p=.037 (significantly WORSE at h3).
mlp_own_era5_s0 vs ridge_own_era5: -0.53 ns / -2.56 p<1e-4 / -3.14 p=1e-4. The Nie et al.
2025 convergence argument is on firmer ground than pre-audit.
  One anomaly to NOT over-read: gbm_corr_top1_era5 beats ridge_corr_top1_era5 at h2 by
  +1.12% (p=.0185) - a single cell among many contrasts, no multiplicity control, and the
  same model is flat at h1/h3. Treat as noise unless it replicates.

STATUS after foundational rerun: contribution #1 (benchmark) STRONGER; contribution #3
(neighbor) is a NULL and must be rewritten as a controlled negative result; ERA5 support
narrows to lead 1. Crossing (contribution #2) + torch phases still to run.

### CORRECTION (same session): the ladder numbers two entries above were WRONG

Self-caught while diagnosing an assertion failure in run_phase6_li_comparison.py. The
"+6.23/+11.14/+9.42/+8.45/+9.60/+12.23" ladder reported earlier was computed by comparing
kalman_predictions.csv against phase2_baseline_summary.csv - but those two files were on
DIFFERENT ROW SETS. run_kalman_baseline.py still clipped its test window by target_date
(19,656 rows at every horizon) while phase2/phase3b had already moved to issue-date folds
(19,422 at h1 down to 18,252 at h6). Comparing RMSEs across different row sets is invalid.
DISREGARD those numbers; they were never real.

**AUTHORITATIVE ladder** = results/paper_baseline_ladder.csv (build_paper_ladder.py), which
matches rows across phase2 and phase3b, both issue-date pipelines. Skill % vs stronger
damped variant:
  model                 h1     h2     h3     h4     h5     h6
  kalman_ar1          4.98   8.79   5.62   3.07   2.55   3.63
  kalman_own_ridge    4.82   7.87   6.04   5.36   5.14   5.16
  kalman_corr_top1    5.11   7.78   5.86   5.02   4.81   4.86
  ridge_own_perbasin  3.05   5.94   4.26   2.05   0.38  -0.29
  ridge_own_lags      1.17   4.78   5.08   3.64   3.17   2.49
  persistence       -20.75 -22.68 -21.39 -19.49 -16.04 -15.80

**THE REAL HEADLINE: the benchmark result is essentially UNCHANGED by the fixes.**
Pre-audit was +4.8/+8.1/+5.3/+3.2/+2.5/+3.5; corrected is +5.0/+8.8/+5.6/+3.1/+2.6/+3.6.
That is a robustness result, not a growth result - the earlier "margin tripled" claim was
pure artifact. Good news for the paper (the contribution survives both fixes intact), but
it must be stated as stability, not improvement.

**The audit was RIGHT about lead 4 and I was wrong to say otherwise.** kalman_ar1 vs
ridge_own_perbasin on matched rows: h1 +1.99 p=6.6e-4 | h2 +3.04 p=7.7e-7 | h3 +1.43
p=.013 | **h4 +1.04 p=0.078 NOT SIGNIFICANT** | h5 +2.17 p=.0051 | h6 +3.92 p=8.3e-6.
The manuscript claim "beats per-basin ridge at all six leads (p<=0.042)" MUST become
"at five of six leads; lead 4 is not significant (p=0.078)".
NEW, also needs fixing: vs the POOLED ridge (ridge_own_lags) the filter is significant only
at h1-h2 (+3.86 p=9.1e-6, +4.21 p=1.2e-11); h3-h6 are ns and h4/h5 are nominally NEGATIVE
(-0.59, -0.64). Pre-audit text claimed "significant at leads 1-3 and 6, nominal at 4-5" -
that is now wrong in both directions.
Filtering class still owns h3-h6 via kalman_own_ridge / kalman_corr_top1 (+4.8..+6.0),
which remains the strongest statement available at long leads.

**FIFTH-SITE BUG FOUND (P0-2 was under-fixed).** The issue-date fix covered the five engine
sites in src/gracefc but NOT the inline fold splits in five runner scripts:
run_kalman_baseline.py, run_phase6_li_comparison.py, run_phase4_surrogates.py,
run_phase5_fusion.py, run_phase5_coupled.py. All five clipped test windows by target_date.
Now all fixed to issue_date. Caught only because run_phase6_li_comparison.py has an internal
guard asserting its recomputed targets match phase3b's stored targets (max |diff| was 4.11
standardized units) - that assertion earned its keep. Kalman baseline + Li comparison
rerunning; phase4 surrogates and phase5 fusion/coupled also now need reruns before their
numbers can be quoted.

### Li crossing on corrected pipeline (contribution #2): SURVIVES, same shape

run_phase6_li_comparison.py rerun after its inline fold split was fixed to issue-date; its
internal target-consistency guard vs phase3b now PASSES. Matched sample: 227 basins x 60
months (was 61 - one fold-boundary month reassigned by the protocol change), folds 1-4.
Li skill relative to OUR arms (negative = we win), li_lstm_full vs kalman_ar1:
  h1 -17.1% p=.0039  <- WE WIN lead 1 (bare 3-parameter filter beats their full LSTM)
  h2  +6.3% p=.15    <- not detectably different (nominally Li, was nominally us pre-audit)
  h3 +14.0% p=8.6e-4 | h4 +18.5% | h5 +22.0% | h6 +25.3% (p<=3.4e-5)  <- Li wins, widening
Crossing still sits between leads 2 and 3. Also: their full product is nominally WORSE than
damped persistence at h1 on the matched sample (-11.1%, p=.057), and their nonseas variant
loses h1 by -35% - the "no GRACE lags" design cost at short leads is robust to our fixes.
NOTE: this is vs our LINEAR arms; the pre-audit comparison used the phase-8 stack (not yet
rerun). On the matched sample kalman_ar1 sits at +5.2/+9.1/+10.8/+12.9/+12.4/+11.9 vs
damped - long-lead skill vs damped is much higher on the 227-basin/60-month matched sample
than on the full global sample; state matched-sample numbers only in the crossing section.

### Stale-cache trap caught by the new paired-row assertion

run_kalman_baseline.py has a resume shortcut that reuses results/kalman_predictions.csv if
present; it silently reused the OLD target-date-clipped file. The stats.py paired-row
assertion (added this session, audit P1-5) caught it: targets differed on exactly 4
fold-boundary months x 234 basins (the two protocols assign those months to different
folds, hence different climatology fits, max diff 4.1 sigma). Stale file deleted, refit
running. This is the second time in one session an internal guard caught a real
inconsistency (after the Li target guard) - the audit's cache-hygiene P1 (content-addressed
caches, run manifests) is clearly earning its priority.

### Post-rerun audit (subagent, results/post_rerun_audit.md): PASS WITH NOTES

Every recomputed headline number matched the RUN_LOG claims exactly (ladder, neighbor null,
stratification, crossing, ERA5, basin table - the last proved from raw NetCDF that only
solutions 110/144 changed convention). No remaining target-date fold splits anywhere.
Notes acted on immediately:
1. run_kalman_baseline.py resume shortcut DELETED (bit once, gone now).
2. kalman_fold_params.pkl is now content-addressed: new src/gracefc/cache.py stores a
   SHA256 fingerprint over (basin table bytes + fold spec + protocol tag "issue-date-v1");
   all 11 load/save sites routed through load_params_cache/save_params_cache, which refit
   from scratch on any mismatch. Existing pkl stamped (audit verified it equals a fresh
   refit). run_phase4_surrogates + run_phase6_basin_analysis now assert non-empty cache.
3. Wording nit adopted: min300 arm at h2 beats 2/50 placebos (not 0/50) - only the
   unrestricted corr arm is 0/50 at h2.
4. Audit note kept in mind: phase6==phase3b agreement is a determinism check via the shared
   params cache, not independent replication - do not oversell it in the paper.
Torch phase 7 rerun launched (resmlp -> lstm -> gnn, sequential, fingerprinted cache).
Phase 8/8b to follow after phase 7 lands + is audited. Phase4/5 control reruns queued
behind torch (lower priority: the linear neighbor result they control for is now null).

## 2026-08-14 (day) - Phase 7 corrected rerun complete + analyzed; phase 8 launched MODIFIED

Timeline note: the laptop rebooted ~6:36 PM mid-chain on 8/14; resMLP and LSTM had already
finished cleanly, GNN died mid-f2 (no traceback - machine hang) and was rerun from scratch.

Analysis (subagent, results/phase7_corrected_analysis.md). Integrity: n exact per horizon,
ridge arms bit-identical across files and digit-matched to phase6. Findings:
1. LSTM ERA5 sequence gain SURVIVES, slightly larger: ens +2.17/+2.36/+1.43% (p<2e-5).
   GRACE-only LSTM still no win (h3 significantly worse than ridge).
2. resMLP STRENGTHENS: vs ridge twin ens +0.81/+2.15/+2.17% (all seeds significant).
   Biggest architectural number in the study is NEIGHBOR-FREE: resmlp_own_era5 vs
   ridge_own_era5 ens +3.35/+3.58/+2.84%.
3. Neighbor increments null-to-negative on every ERA5 backbone (resMLP ens -0.55..-0.67
   p<.05; lstm_corr_top1 0/20 placebos at h2-3). ONE robust exception: GRACE-only resMLP
   neighbor beats 20/20 placebos at every seed and horizon (placebo family sits exactly at
   ridge_own) - the nonlinear GRACE-only neighbor signal survives the corrected pipeline.
4. GNN: never beats ridge (sole ens blip at h2 vs weakest twin, no seed support - noise);
   two-hop <= one-hop. Question stays closed.
Seed flags: resmlp_era5 s2 outlier; LSTM h2 neighbor sign flip s0 vs s1.

Phase 8 launched MODIFIED per the analysis recommendation: experiment_lstm_combined.py
gained lstmres_own - a neighbor-free stage-2 control fed each basin's OWN propagated state
through the identical feature path (self-graph through propagated_neighbor_features), same
stage-1 net, same stage-2 capacity. Decisive contrast = lstmres_corr_top1 vs lstmres_own:
architecture term vs information term. Runner contrasts extended. Smoke PASS (f1/h1/s0).
Chain: h1-3 -> h4-6 (tag phase8b_lstm_h46) -> run_phase8b_merge (Li h1-6 comparison).

## 2026-08-15 - PHASE 8 + 8b COMPLETE on corrected pipeline. Neighbor correction SURVIVES.

Chain (h1-3 -> h4-6 -> merge) all exit 0. Compute block for the paper is now closed.

**1. Stage-2 neighbor correction vs its OWN stage-1 LSTM (the within-backbone contrast):**
  seed0  h1..h6: +0.64 / +1.33 / +1.29 / +1.26 / +1.51 / +1.90 %
  seed1  h1..h6: +1.19 / +1.56 / +1.36 / +1.37 / +1.28 / +1.93 %
  all 12 cells positive, all DM p <= 1.8e-5. Pre-audit was ~+1% flat at every lead; the
  corrected version is comparable at short leads and LARGER at long leads (h6 ~+1.9%).
  Horizon-stability claim not only survives, it strengthens.

**2. THE DECISIVE NEW CONTROL (lstmres_own: identical stage-1 net, identical stage-2 model
and capacity, fed the basin's OWN propagated state instead of the neighbor's):**
  own-state correction vs stage-1:      -1.03..-2.30% (HARMS, all p<1e-8, both seeds)
  neighbor correction vs own-state:     +1.65..+4.11% (all p<=8.7e-9, both seeds, h1-6)
  Interpretation: the two-stage ARCHITECTURE alone does not explain the gain - a second
  stage on information stage 1 already consumed actively overfits. The gain is specific to
  the neighbor's INFORMATION. The phase-7 worry (that the ~3% neighbor-free two-stage term
  could account for the correction) is REFUTED.
  DISCIPLINE FOR THE PAPER: the honest "does the neighbor add skill" number is contrast 1
  (+0.6..+1.9% vs stage-1). Contrast 2 (+1.65..+4.11%) answers a DIFFERENT question
  (information vs architecture) and is inflated by the control being actively harmful -
  never quote it as the neighbor's skill contribution.

**3. Placebo integrity:** lstmres_corr_top1 beats 20/20 degree-matched random-graph
correction placebos at every lead and seed EXCEPT s1 at h2 (16/20, p_rank=0.238). 11 of 12
cells clean. Must be disclosed; also the known caveat that placebo draws are reused across
horizons, so the six per-lead results are not independent events.

**4. THE CROSSING IS ROBUST - contribution #2 essentially unchanged by both fixes.**
lstmres_corr_top1_ens vs li_lstm_full (227 basins x 60 months = 13,620 rows/lead):
  h1 +20.0% (p=3.4e-4) | h2 -3.5% ns | h3 -12.9% (p=.015) | h4 -17.4% | h5 -23.6% | h6 -30.3%
  pre-audit: +20.2 / -3.8 / -12.4 / -16.5 / -21.6 / -26.7. Same shape, same crossing point
  between h2 and h3, long-lead gap slightly wider. The bare Kalman filter alone still beats
  their full product at h1 (+14.6%, p=.0039; pre-audit +14.4%).
  vs their non-seasonal variant: we win h1 +30.9% and h2 +9.0% (p=.0083), crossing moves to
  h3-h4. vs damped persistence: ours +11.1..+16.6% at every lead; THEIRS IS -11.1% AT h1
  (worse than damped persistence), +14.8..+34.1% at h2-6 - the cleanest single statement of
  the filtering-vs-forcing split in the study.

Audit + analysis subagents launched on this batch per standing process.

### Phase 8 audit (subagent, results/phase8_corrected_audit.md): PASS WITH NOTES

Reproduction EXACT: all 36 headline cells recompute to <=3.3e-14 (skill) and 3.0e-19 (p);
all 30 Li crossing cells match to 0.0; Li file exactly 227 x 60 = 13,620 rows/horizon;
ridge/Kalman/LSTM twins bit-identical to phase 6/7 at h1-3; fingerprinted cache used; no
duplicate keys; merge lossless. Adding lstmres_own perturbed nothing (all 9 pre-existing
arms bit-identical to phase 7).

CONTROL IS MECHANICALLY FAIR: self_idx reproduces the basin's own propagated state exactly
(max|fs - kalman| = 0.000e+00, every row/fold, h1/h3/h6); same net object, same resid2, same
MLP seed and capacity, same row alignment. The -1 sentinel asymmetry is EMPTY (0 of 234
basins lack a top-1 neighbor in any fold, 0 zero-filled rows either arm) - it explains none
of the gap.

BUT MY INTERPRETATION WAS WRONG AND IS CORRECTED HERE. I wrote that the own-state control
"overfits noise". It does not: its correction is ANTI-correlated with stage-1 error
(corr -0.08..-0.16), and only 0.16-0.54pp of its 1.0-2.3% harm is variance cost - 79-85% is
a systematically WRONG-DIRECTION correction, an artifact of the in-sample residual protocol
(it fits the LSTM's own-state overfit backwards). So lstmres_own is an inferentially TOO-EASY
control and must be demoted to a DIAGNOSTIC.

THE PRIMARY CONTROL IS THE RANDOM-GRAPH PLACEBO, AND IT ALREADY EXISTED. Placebos (novel
feature, wrong identity) cost only -0.27%..-0.05%, matching the pure-variance prediction;
lstmres_own is 5.7-21.4 placebo sd worse. The two defensible numbers AGREE:
  neighbor increment vs its own stage 1:   +0.64 .. +1.90 %
  neighbor increment vs placebo median:    +0.92 .. +1.96 %
NEVER quote the +1.65..+4.11% vs-lstmres_own contrast - not as skill, not as the
architecture term.

TWO PLACEBO DEFECTS, BOTH NOW FIXED IN CODE:
 (1) Placebos rode the seed-0 stage-1 net but were scored against BOTH seeds; the seed-1
     handicap (+0.002..+0.009 RMSE) dwarfs the placebo spread (sd 0.0008), and the lone
     "16/20 at s1 h2" blemish is ENTIRELY this artifact - in skill space all 12 cells are
     20/20. FIXED: experiment_lstm_combined now stores per-seed stage-1 residuals/predictions
     and emits placebos per (arm, seed) as "{stem}_s{s}_rand{n}"; phase7.summarize_and_write
     prefers seed-matched placebo draws and falls back to family draws (backward compatible
     with phase 7 files). Smoke PASS.
 (2) The 20 draws are reused across all 5 folds AND all 6 horizons (verified identical
     graphs) - 12 correlated re-scorings of 20 realizations, NOT 12 independent events.
     Stays as a disclosed limitation. Also p_rank=0.0476 is the 1/21 floor, not a measured
     p; quote the z-score instead (-6.8..-18.6 sd).
 (3) Ensemble within-backbone contrasts were computed nowhere despite the ensemble being the
     designated headline quantity. FIXED: run_phase8b_merge now writes
     phase8b_h16_ensemble_headline.csv. Audit's values: +0.91/+1.45/+1.33/+1.32/+1.42/+1.96%
     at h1-6, all p<1e-9.

Rerun launched with all three fixes. Real-arm numbers are expected bit-identical (same seeds,
same data; only placebo rows are added) - verify that on completion.
RANKED NEXT EXPERIMENT (revision-stage, not blocking): out-of-fold stage-2 residuals. It
would delete paper limitation 1, make the lstmres_own diagnostic interpretable, and the audit
expects it to RAISE the neighbor number.

### Placebo-fix rerun VERIFIED (2026-08-15). Phase 8 numbers are now FINAL.

1. Real arms bit-identical to the pre-fix run (s0 h1 +0.639443, s1 h6 +1.934878, ... every
   digit) - confirms determinism and that adding placebo/control arms perturbs nothing.
2. Seed-matched placebos: **all 12 cells now 20/20**. The 16/20 blemish is GONE, exactly as
   the audit predicted - it was purely the seed-0-net/seed-1-arm mismatch, not evidence.
3. add_ensembles was silently limited to a 2-family module constant, so two ensemble
   contrasts were skipped. Parameterized (fams=...) rather than widening the constant,
   because ENSEMBLES also drives the Li matched-sample n_models - widening it would have
   changed the crossing sample. Verified after: Li crossing identical (13,620 rows/lead,
   +20.0/-3.5/-12.9/-17.4/-23.6/-30.3).

**FINAL ENSEMBLE NUMBERS (phase8b_h16_ensemble_headline.csv) - quote these:**
  neighbor correction vs its own stage-1 LSTM (THE skill claim):
    h1..h6  +0.91 / +1.45 / +1.33 / +1.32 / +1.42 / +1.96 %   all p <= 1.2e-10
    CIs exclude zero at every lead (lo +0.64 .. +1.67)
  vs the strongest linear system with identical information (ridge_corr_top1_era5):
    +2.75 / +3.86 / +2.92 / +2.52 / +1.78 / +2.23 %           all p <= 1.5e-9
  own-state DIAGNOSTIC (not a skill claim): control harms -1.16 .. -2.12 % (p<=2.5e-7);
    neighbor beats it +2.05 .. +3.70 % - report as evidence that the gain is INFORMATION
    not ARCHITECTURE, never as the neighbor's skill.
  DELIVERY, now honestly at ensemble level: neighbor as an LSTM INPUT CHANNEL gives
    +0.36% ns at h1 and is negative at h2-h6 (-0.37..-0.61%, mostly ns/marginal), against
    +0.91..+1.96% for the same information as a correction stage. This is the cleanest
    statement of the delivery result the study has produced.

Compute for the paper is CLOSED. Remaining: manuscript renumbering, then figures.

---

## 2026-08-15 - Resolution stratification audit (scripts/run_resolution_sensitivity.py)

Motivation: the manuscript's PRIMARY sensitivity (audit P0-3) splits basins at
90,000 km2. Both inputs are ours, not CSR's - the area is our own cos-lat
integration of the mask file, the threshold is a convention. Three checks.

**1. External cross-check: NOT AVAILABLE.** `HydroShed+Mascon_Basins_L3.nc`
carries only `mask`, `ID`, `Name`, lat, lon - no area, no mascon id, no cell
count. The CSR grid file carries no tile geometry either. There is nothing from
CSR to validate our areas against; say so in the paper rather than implying a
cross-check. (Internal consistency is exact: area_km2 recomputed through the
tile path matches to 1e-6.)

**2. Native mascon tiles RECOVERED empirically.** The gridded product replicates
one tile value across its 0.25 deg cells, so cells sharing a tile have
bit-identical series. Fingerprinting 8 months and grouping on exact equality
gives **42,107 distinct global tiles, median area 12,123 km2** (IQR 11,518-13,032)
while cell counts run 16 -> 24 with latitude. Near-constant area with varying
cell count is the signature of CSR's ~1 deg EQUAL-AREA mascons - the recovery is
validated. Inventory: results/mascon_tile_inventory.csv.

New per-basin diagnostic `contamination` = share of a basin's mascon-weighted
signal contributed by LAND OUTSIDE the basin, = sum_t (w_t/W)(1 - land_fill_t).
This is the physical leakage quantity; area is only a proxy for it. They are
nearly independent: Spearman(area, contamination) = **-0.325**. Only 3 of 234
basins have n_eff_tiles < 2, so "one resolution element" was never the right
mental model - 90,000 km2 is ~7 mascons.
Files: results/resolution_diagnostics.csv.

**3a. Area threshold is NOT a knife edge.** h1 kalman_corr_top1 vs
kalman_own_ridge, sub-resolution stratum, by cut:
  40k  +0.32% p=.39 (n=15, underpowered) | 60k +0.93% p=.0037 | 80k +1.09% p=.00046
  90k  +1.03% p=.00065 (the manuscript number, reproduced exactly)
  100k +1.34% p=3.0e-5 | 120k +1.19% p=2.6e-5 | 150k +1.22% p=8.1e-5 | 200k +0.93% p=.00086
The complement stratum runs +0.31 -> -0.18% and is never significant. The result
is stable across every cut from 60k to 200k. Files: resolution_sweep_area.csv.

**3b. Contamination separates it better, and by terciles (no threshold choice):**
  tercile_low  -0.27% p=.43 | tercile_mid -0.18% p=.66 | **tercile_high +1.16% p=.00051**
Files: resolution_sweep_contamination.csv.

**DECISIVE 2x2 (results/resolution_cross_2x2.csv) - CLAIM CHANGE REQUIRED:**
```
  resolved       x cont_high     n=57   +1.161%  p=.0045
  resolved       x cont_lowmid   n=142  -0.500%  p=.082
  sub_resolution x cont_high     n=21   +1.166%  p=.0071
  sub_resolution x cont_lowmid   n=14   +0.913%  p=.036
```
Contamination separates the effect INSIDE the resolved basins, at the same
magnitude as in the sub-resolution ones. Therefore the current manuscript
sentence - "no detectable cross-basin effect in basins GRACE can resolve" - is
NOT SUPPORTED and must be replaced with the footprint-sharing version: the h1
neighbor effect is confined to basins whose mascon footprint is shared with
land outside the basin, regardless of basin size. This STRENGTHENS the leakage
reading of contribution 3 and removes the "small basins" framing, which the 2x2
shows was a proxy standing in for the real variable.

Caveats: single horizon (h1) and the linear kalman arm only; the stacked phase-8
correction has NOT been re-stratified this way. Contamination uses the union of
all 284 mask units as "land", so a coastal tile's ocean half is not counted as
foreign - island basins therefore score LOW contamination, which is why the two
stratifiers disagree as much as they do.

## 2026-08-15 - Phase-8 stacked correction stratified by contamination (the leakage-artifact test)

The resolution audit (entry above) left one ranked-highest open item: the paper's
headline neighbor result - the phase-8 stacked correction - had never been stratified
by contamination or area. If it concentrated in high-contamination basins the main
claim was a leakage artifact. It does not.

New script: scripts/run_phase8_stratification.py -> results/phase8_stratification.csv.
Strata replicate run_resolution_sensitivity exactly (contamination terciles over the
234 keep basins; below_resolution area split; 2x2 cross). Horizon is filtered inside
pooled_monthly_dm/block_bootstrap_skill_ci BEFORE the (name, target_date) pairing
merge, so the shared-target-date trap cannot recur. All 234 basins present in phase-8
predictions.

**1. Stacked ensemble (lstmres_corr_top1_ens vs lstm_own_era5_ens) is positive in all
60 stratum x lead cells and shows NO contamination concentration:**
  cont tercile low  (n=78):  +0.69/+1.31/+1.29/+1.28/+1.31/+1.84%  all p<=0.0069
  cont tercile mid  (n=78):  +1.09/+1.58/+1.44/+1.33/+1.61/+2.01%  all p<=1e-4
  cont tercile high (n=78):  +1.00/+1.48/+1.29/+1.35/+1.37/+2.03%  all p~0
  resolved x cont low/mid (n=142, linear effect -0.50% here): +0.89/+1.57/+1.46/+1.38/+1.49/+2.00%, all p<1e-4
  sub-resolution (n=35): +0.64..+1.05%, all p<=0.043
  Tiny-n cells (sub x cont splits, n=14-21) stay positive; a few p>0.05 as expected.
  Per-seed worst cell across every stratum/lead: +0.22% (still positive).

**2. Linear extension h1-6 (ridge_corr_top1_era5 vs ridge_own_era5, same row set):**
the linear effect is BOTH contamination-confined AND short-horizon -
  cont high: h1 +1.11% p=2.3e-4 | h2 +0.59% p=.037 | h3-h6 dead (+0.37..0.00, ns)
  cont mid:  negative at every lead (-0.23..-0.78%) | cont low: negative at every lead
This completes the resolution audit's secondary item (h2-6 on a linear arm).

**3. Diagnostic: plain resMLP neighbor-vs-own ERA5 twin (phase 7, 3-seed ens, h1-3)
is NEGATIVE overall** (-0.55..-0.67%, p<0.05), with the harm concentrated in LOW/MID
contamination (resolved x low/mid: -0.89% p=.011) and flat in high contamination.
Naive neighbor-feature concatenation imports noise except where footprints overlap;
only the two-stage stacked delivery converts neighbor state into uniform gains.

CONCLUSION: the linear neighbor effect and the stacked correction have different
signatures. Linear = leakage signature (high-contamination footprint sharing, h1-2
only). Stacked = uniform across contamination and size at every lead -> the leading
artifact explanation for contribution 3's nonlinear tier is ruled out by
stratification. This is claim-strengthening: the paper now separates a
leakage-driven linear pocket from a delivery-dependent correction that survives the
leakage stratifier. Framing must stop at "artifact ruled out"; mechanism remains
unidentified (no "genuine teleconnection" claim).

Caveats: stacked stratification reuses the headline's test months (not an independent
sample); contamination is a lower bound (union-of-284-masks land definition; island
basins score low); linear 2x2 remains h1-only for the kalman arm. REWRITE_LEDGER.md
amended (stratification section superseded, new stacked-stratification section,
contribution-3 rewording, limitations updated); rewrite agent notified mid-flight.

## 2026-08-15 - Stratification batch AUDIT PASS + formal interaction tests added

Independent audit of the phase-8 stratification batch (subagent, own DM/HAC
reimplementation, no gracefc imports): all 8 spot-check cells reproduce to 4 decimals;
strata are basin-set-identical to run_resolution_sensitivity (qcut boundaries tie-free,
q1=0.0792 q2=0.1350); panel exactly balanced (n_rows = n_basins x n_months) so no
weighting ambiguity; no cross-horizon/fold pairing possible; the both-arms-degrade
alternative reading is EXCLUDED empirically (absolute RMSE by tercile is non-monotonic,
mid easiest, arms track within 0.5% everywhere). Latent code weakness found and FIXED:
add_seed_ensembles used an inner join whose NaN guard was dead code - replaced with a
length-vs-seed0 assertion (did not fire; row sets bit-identical).

WORDING CORRECTION (applied to ledger + manuscript): "uniform across strata" is an
overclaim - the SIZE axis has a real gradient, sub-resolution basins gain LESS, which
is the OPPOSITE of leakage's prediction (claim-safe, now stated openly).

NEW: formal stratum-interaction DM tests folded into run_phase8_stratification.py
(method from the auditor: per-stratum monthly loss differential normalized by the
stratum's time-mean reference MSE, DM on the difference; index-alignment assert added)
-> results/phase8_stratification_interactions.csv, all leads h1-6:
  contamination high-vs-low: ns at EVERY lead (h1 p=.099 trending MORE gain in
    cont-high, i.e. the safe direction; h2-h6 p>=.51) -> "not concentrated" is now a
    TESTED statement, not a description.
  sub vs resolved: significant at EVERY lead (+0.35pp h1 p=.026 ... +1.10pp h6
    p=1.8e-4) - small basins gain less at all six leads.
Sign convention: mean_diff_pp < 0 = first-listed stratum gains more.

Manuscript rewrite agent completed the full rewrite (both mid-flight spec updates
applied; REWRITE_NOTES.md has the change log); the two INTERACTIONS-CSV todos are now
filled with the verified values. MiKTeX installed (user-scope); first compile of the
corrected manuscript succeeded (27 pp). Remaining todos: \todo{RERUN} markers awaiting
the control-phase chain (surrogates/phase5/hybrid/basin-analysis).

## 2026-08-15 — External-audit repair batch (code + geometry + manuscript) and corrected chain launch

- Trigger: external audit of the full codebase/manuscript, verified claim-by-claim by five
  parallel agents (all five blockers confirmed; verification details in the session record).
- Git provenance BEGINS here: repo initialized, baseline commit `c70b7e5` = state before any
  repair; every repair is a tracked commit. data/, archive/, and results files >5 MB are
  gitignored (archive is manifest-pinned; live big files get SHA256_MANIFEST_LIVE.csv after
  the chain).
- **Official CSR geometry (blocker 1):** CSR RL06.3 DOES publish a native-mascon-ID mapping
  (`CSR_GRACE_GRACE-FO_RL0603_mascons_mapping_file.nc`) and v02 land/ocean masks
  (doi:10.15781/cgq9-nh24) — the manuscript's "no external assignment field exists" was wrong.
  Downloaded to `data/raw/csr_ancillary/` (note: the page's displayed mask filenames 404;
  the real files carry `v02_`). `run_resolution_sensitivity.py` rebuilt on official geometry:
  **fingerprint recovery agrees EXACTLY** (42,107 = 42,107 tiles, purity 1.0 both directions,
  ARI 1.0, contamination Spearman 1.0000, max abs diff 0.0000) → all contamination-based
  results stand, now externally validated. New: `csr_geometry_validation.csv`,
  `resolution_cross_2x2_200k.csv` (CSR caution threshold: resolved×cont_high +1.10% p=.033;
  resolved×cont_lowmid −0.80% p=.014 significantly negative; subres×cont_lowmid weakens to
  +0.69% p=.057).
- **Placebo seed repair (blocker 3):** all six experiment engines now fit placebo heads with
  the REAL arm's model seed (was 1000+draw = init/shuffle/early-stop split confounded with
  graph identity); placebo labels carry `_s{s}`; draws seeded per (fold, horizon) cell so
  leads are independent draws (deletes the reused-draws limitation). GBM placebos were
  already seed-matched — convention generalized.
- **Other code repairs:** bootstrap block = max(3, horizon) (stats.py + inline copy in
  run_phase5_stats); IAAFT now FFTs the gap-interpolated full monthly grid (NaNs re-imposed;
  full-record span + real-graph reuse documented as intentional null design); ERA5 zip
  responses merged back to flat per-year .nc + year-contiguity assertion in ingestion
  (latent bug, nothing currently lost); maritime-SE-Asia continent box (Java, Nusa Tenggara,
  Sulawesi, Timor, Maluku → asia; basin_meta.csv refreshed, 5 rows).
- **New experiments:** `run_r0_ablation.py` (r=0 forces gain 1 → damped persistence with MLE
  rho; decomposes the benchmark margin into noise-filtering vs rho-estimation terms — decides
  the title's mechanism claim); combined engine gains OOF stage-2 residuals (3 contiguous
  issue-month blocks, `lstmres_oof_*` arms + seed-matched placebos), the delivery-equalized
  `lstmres_corr_top1_hist12` arm (stage 2 fed the input arm's 12-month history), and the LSTM
  engine gains flat-12-month ridge/MLP twins (deconfounds sequence modeling from history
  length). All three engines smoke-tested PASS.
- **Chain:** `scripts/run_chain.py` (declared inputs/outputs, fail-fast, per-step logs)
  launched with 14 steps: predlag, conditioned (corr:1 corr_min300:1, nino34+dmi), surrogates
  (repaired IAAFT), r0_ablation, phase5_nonlinear, phase5_stats, basin_analysis (both missing
  deps now produced upstream), phase6_era5, phase8 h1-3, phase8b h4-6, merge, stratification,
  phase7_resmlp, phase7_lstm. phase7_gnn defined but excluded (real arms unaffected by the
  seed repair; placebo ranks not quoted anywhere).
- **Manuscript surgery (partial, commits `3525539`/`bd45cb7`):** geometry paragraph rewritten
  around official products + exact validation; CSR 200k caution + 200k 2×2 added; 2×2
  headline softened; "ruled out by stratification" → "no detected concentration under this
  proxy"; delivery section discloses the 12-month-history vs propagated-scalar representation
  confound; abstract/Methods state per-lead issue windows (83→78, last issue 2026-04→2025-11);
  \graphicspath fixed, F1/F2/F5/F8 wired; jump-screen todos filled. PENDING on chain output:
  title/abstract noise-framing decision (r0 ablation), remaining 16 todos, F3/F4/F6, table
  refreshes, limitation rewrites (OOF, reused draws).
- **Tests:** `tests/` pytest suite pinning the repaired invariants (14 pass, 1 skips until
  r0 predictions exist). pytest==9.1.1 appended to requirements-lock.
- Stale pre-audit analyses/audits (12 docs incl. phase6_analysis.md, which the external audit
  missed) + smoke files → `archive/superseded_preaudit_docs/`; results/README.md documents
  the provenance rules.

---

## 2026-08-15 evening — repository cleanup for handoff (no results touched)

Purely structural: made the repo legible to a teammate opening it cold. **No result CSV, no
figure, and no manuscript number changed.** The corrected-pipeline chain was mid-flight
(phase6_era5) throughout, so nothing it imports or writes was moved.

- **Root decluttered.** Working notes and reference material moved out of the repository root
  into `docs/`: `code_structure.md` → `docs/CODE_MAP.md`, `context_global_study.md` →
  `docs/STUDY_CONTEXT.md`, `AUDIT_REPORT.md` → `docs/AUDIT_2026-08-13.md`, the four `Li_2026_*`
  files → `docs/reference/li_kusche_2026_*`, and the two Africa pilot context files →
  `docs/history/`. The manuscript drafting record (`DECISIONS`, `FIGURE_PLAN`,
  `REWRITE_LEDGER`, `REWRITE_NOTES`) moved to `paper/notes/`, leaving `paper/` holding only
  what LaTeX compiles. All moves were `git mv`, so history follows the files; every live
  cross-reference in docs, `main.tex` comments, `figures/BUILD_NOTES.md` and
  `scripts/make_figures.py` was rewritten to the new paths. RUN_LOG entries above were left
  alone — this journal is append-only and records what was true when written.
- **Docs rewritten for a reader with no background**, at Ishaan's request — teammates joining
  cold should not need hydrology or forecasting vocabulary to follow the repo. `README.md` is
  now a full front door: what GRACE and TWSA actually are, why we deseasonalize, the three
  findings in plain terms, a repository map, setup and data acquisition, a from-scratch
  explanation of the Kalman filter and how the neural arms sit on top of it, the issue-date
  fold protocol (the convention that caused P0-2) spelled out, the placebo/surrogate logic,
  and a 19-term glossary. `docs/CODE_MAP.md` rewritten the same way, organised around
  "`src/` thinks, `scripts/` runs, `results/` remembers", with files split into "need to
  understand" vs "only if you go deeper".
- **Module docstrings rewritten in plain language** for the eight files a newcomer actually
  opens: `kalman.py` (the two-line model, the gain, why it beats damped persistence, the r
  caveat), `evaluate.py` (the issue/target-date split, spelled out with the reason), `graphs.py`
  (what the placebo graphs are for), `decompose.py` (why we remove trend and season),
  `stats.py` (what each of the three tests answers), `basins.py`, `experiment_kalman.py` and
  `experiment_lstm_combined.py` (both now open with a plain-terms paragraph, technical detail
  retained below it). Docstrings only — no code paths touched; all 22 modules import and
  15/15 tests pass after each edit.
- **Dead code removed**, after a reference sweep across `src/`, `scripts/`, `tests/` and
  `notebooks/`. Scripts: `regression_check.py` (phase-0 Africa reproduction gate, built on the
  retired 70/10/20 chronological split — it would now emit wrong-protocol numbers if anyone
  ran it) and `run_phase3_neighbors.py` (ridge-backbone phase 3, superseded by 3b; its outputs
  no longer exist anywhere in `results/`). Functions: `basins.decode_csr_time` (superseded by
  `assign_solution_months`), `decompose.restore` (never called; `run_phase6_li_comparison.py`
  hand-rolls the remove-restore inline), and `models.damped_persistence_forecast` (duplicate of
  the `mode="regression"` branch of `evaluate.damped_persistence_pred`, and orphaned by the
  `regression_check.py` deletion). Verified after: all 22 modules import, 15/15 tests pass.
  Several scripts that looked disposable are NOT — `download_indices.py`,
  `build_li_basin_series.py`, `run_phase2_baselines.py` and `run_phase6_hybrid.py` all feed
  declared chain inputs or a live manuscript todo, and were kept.
- **Known duplication left in place, deliberately:** `experiment_gnn._era5_node_tensor` vs
  `experiment_lstm._era5_state_tensor`, the Units-parsing preamble in
  `basins.assign_solution_months`, and the byte-identical `emit_rows` in the phase-5 coupled
  and fusion runners. All three are safe merges, but the chain was importing those modules;
  refactoring mid-run risked failing a multi-hour step for no scientific gain.
- **New:** `scripts/make_manifest.py` writes and verifies `results/SHA256_MANIFEST_LIVE.csv`
  over the result files git does not carry (`--check` re-verifies). Run it once the chain lands.
- **Housekeeping:** `.claude/settings.local.json` untracked (machine-local permissions with
  absolute scratchpad paths); `results/chain_*.log` untracked (transient per-run, superseded by
  this journal); `.pytest_cache/` ignored; notebook outputs cleared so pre-audit numbers in
  stored cells cannot be mistaken for current results.
- **Remote:** repository published to `https://github.com/ishaankejriwal/graceseespaper.git`.

---

## 2026-08-16 23:48 — corrected rerun chain COMPLETE (14/14 steps, exit 0)

`scripts/run_chain.py`, launched 2026-08-15 ~17:15, finished 2026-08-16 23:47 (~30.5 h).
Step times: predlag 6.1 / conditioned 2.7 / surrogates 32.1 / r0_ablation 4.4 /
phase5_nonlinear 134.6 / phase5_stats 0.9 / basin_analysis 3.5 / phase6_era5 180.2 /
phase8_h13 327.3 / phase8_h46 298.1 / phase8b_merge 1.9 / phase8_strat 3.3 /
phase7_resmlp 480.5 / phase7_lstm 541.9 min. Every output verified fresh by the runner.

**Integrity:** `results/SHA256_MANIFEST_LIVE.csv` written over the 23 gitignored result
files (1.85 GB); verify later with `scripts/make_manifest.py --check`.

**Headline outcomes of the corrected rerun** (details + exact numbers in ledger §10a–10g):
- Benchmark, crossing, ERA5-lead-1, conditioning, surrogates, jump screen: reproduced.
- Stacked neighbour correction: intact (+0.91..+1.96%, 20/20 placebos, 30/30 fold×lead
  cells positive, 60/60 strata cells positive). OOF stage-2 makes it LARGER (headline was
  conservative).
- NEW, claim-changing (hist12 arm): representation, not delivery, separates the arms →
  §results_delivery retitled, gradient starvation demoted, abstract reworded.
- NEW, claim-changing (flat12 twins): the "sequence modeling" gain was history length.
  A flat-12-month ridge beats the LSTM on identical information at all three leads and
  beats the FULL STACK at h2–3 (tie h1). §results_era5 retitled, nominal-best disclosure
  rescoped. The Nie-convergence theme is now stronger than pre-audit, not weaker.
- Per-basin FDR (stack, h1): 20 helped / 2 hurt of 234; ρ=+0.49 vs linear map.
  New file `results/phase8_perbasin_fdr_h1.csv`.
- cm-unit Li comparison: pooled 5.0–5.8 vs 7.6–7.7 cm (flatters us); per-basin median
  2.39/3.11/3.27 vs 2.70/2.78/2.93 cm — handover one lead earlier than standardized.

**Manuscript:** 39 pp, 0 undefined refs, all 7 figures wired and ledger-asserted.
ONE RERUN todo remains: ERA5 variable-ablation attribution (needs a new LOFO-by-variable
experiment; nothing in the current text depends on it).

---

## 2026-08-17 — external-audit code repairs (stats pairing, cache fingerprint, full chain)

Acting on the code-level findings of the 2026-08-17 external audit; manuscript edits
deferred to the paper pass.

- **`stats._paired_losses` row-set assertion:** the join now raises when both models
  have rows but their (name, target_date) key sets differ — an inner join there would
  silently score each model on a subset it wasn't asked about. A model wholly absent
  from a slice (not run at that horizon) still returns an empty frame: that is
  coverage, not a pairing defect. Two regression tests added (suite now 17).
  Smoke: `build_paper_ladder`, `run_phase5_stats`, `run_phase8b_merge` reran clean
  under the assertion; `make_manifest.py --check` 23/23 files unchanged.
- **Stale artifact found by the smoke:** `paper_baseline_contrasts.csv` still carried
  h4–6 bootstrap CIs computed before the 2026-08-15 `block=max(3,h)` repair. Refreshed:
  point estimates and DM p-values identical, only h4–6 CI bounds moved. Any manuscript
  quote of those CI bounds needs re-checking against the refreshed file.
- **Kalman cache fingerprint widened** (`src/gracefc/cache.py`): now also hashes
  `basin_meta.csv` bytes (the keep-cohort deciding WHICH basins get fit), `kalman.py`
  source bytes (model semantics + fit hyperparameters), and numpy/scipy versions (the
  MLE optimizer). `kalman_fold_params.pkl` was refit from scratch under current code
  before restamping: all 5 folds × 234 basins reproduce rho/q/r with max|diff| = 0.0.
- **`run_chain.py` completed to the whole pipeline:** 30 steps from the table builds
  (basin/ERA5/Li) through phase2/kalman/phase3b, jump screen, all controls,
  fusion/coupled, Li comparison, hybrid, basin analysis, resolution sensitivity,
  phases 7/8, the paper ladder, figures, and the checksum manifest. Only downloads
  (network/credentials) stay manual. This also fixed a latent ordering bug:
  `basin_analysis` ran BEFORE `phase6_era5`, whose predictions file it reads. Every
  declared input/output verified present on disk; README §7 and CODE_MAP synced.
- **Surrogate docstrings rescoped:** "the conservative choice" → disclosed transductive
  sensitivity (surrogate distribution/spectrum estimated with test-period observations);
  closely-matched-null is a design argument, not a proof of conservatism.
- **Deferred:** manuscript claim fixes (four false statements, delivery framing,
  placebo-draw text, issue dates, noise-mechanism language, Chen et al. 2025 citation);
  basin-mask provenance (user: ignore for now); ERA5-coverage disclosure; the
  flat12-vs-LSTM 85 %-train caveat (matched sensitivity already run, direction holds).

---

## 2026-08-17 — manuscript repair pass + two new experiments (attribution, flat85)

Acting on the paper-side findings of the 2026-08-17 external audit. Numbers in
paper/notes/REWRITE_LEDGER.md §10h; scripts added to the chain.

- **NEW EXPERIMENT `run_phase6_era5_attribution.py`** (leave-one-variable-out + solo
  ridge refits at lead 1, plus fold/continent/FDR decompositions of the published
  ridge_own_era5 arm): full arm reproduces +4.62%; evaporation carries the largest
  unique contribution (+1.45pp, p=.002); swvl1/swvl2 strongest solo (+2.29/+2.22%);
  t2m/swe nothing. Fold f3 (2022-05..2023-09, triple-dip La Niña) is the sole failure
  (−3.05%, ns). Continental: NA/EU/Asia +6.0..+7.1%, Africa −0.4% ns. FDR: 23/234 =
  21 helped / 2 hurt. This filled the manuscript's last RERUN todo with measured values.
- **NEW EXPERIMENT `run_flat12_train85_sensitivity.py`**: flat12 ridge refit on the
  LSTM's exact 85% training rows — vs LSTM ens +1.02 (p=.094) / +2.76 / +2.27%; vs
  full-row flat12 within ±0.2% ns. The flat12 result does not rest on the extra rows;
  h1 head-to-head weakens to ns under row matching (disclosed in §results_era5).
- **Manuscript surgery** (all four false statements from the audit fixed): abstract 2×2
  scoped with the exception cell flagged; resMLP placebo count corrected 12/12 → 9/9
  (3 seeds × 3 horizons; also fixed in phase7_corrected_analysis.md); §results_era5
  heading now "A longer window of forcing extends the gain; sequence architecture does
  not"; design-map sentence rewritten around the 2-seed ensemble (+0.81% h4 p=.0011,
  gone h5-6; architecture-vs-window untested past h3); "delivery decides" purged
  (fig05 caption, README Finding 3 → representation decides); placebo-draw text
  unified with the per-cell redraw reality; issue windows June 2019–April 2026;
  "growing rather than decaying" → "persistent, largest at lead 6"; r=0 language
  demoted to ablation-grounded attribution; IAAFT stated transductive; ERA5 coverage
  disclosed (13/234 <50%, 79 <90%, min 24.5%, verified from era5_basin_coverage.csv);
  zhang2025ipa (J. Hydrol. 661:133552 — first author ZHANG, not Chen as the external
  audit had it) cited and the benchmark novelty scoped to forecasting/benchmarking.
- **Metadata**: author/affiliation finalized (sole author, independent researcher),
  contributions/acknowledgements/funding written, \todo macro deleted, stale comment
  blocks removed. REMAINING before submission: mint the Zenodo DOI (needs the
  author's account) and swap it into the availability statement.
- **Builds**: all 7 figures rebuilt, every ledger assert passed; PDF compiled clean
  (40 pp, 0 undefined references, 0 bibtex warnings, no placeholder text in the
  rendered PDF, verified programmatically).

---

## 2026-08-17 — three-agent audit round on the repaired state, findings fixed

Per the standing audit cadence, three independent subagents audited the finished state:
a paper-claims-vs-data verifier (adversarial recompute), a cold-reader repo auditor
(fresh-clone stance), and a code auditor on the two new experiment scripts.

**Code audit (new scripts):** no blockers; the attribution script's construction was
verified to reproduce the published arm's rows to machine precision (19,422 rows, max
|pred diff| 8.9e-16). Fixed on its findings: t2m/swvl1 negative unique terms now stated
(not "ns"), evap p quoted with the 11-test family context (survives Bonferroni), fold
p bound 0.001→0.002, stats.py empty-model path now returns full schema (per_basin_dm_fdr
and bootstrap no longer crash; regression test extended), chain inputs completed.

**Paper-claims audit:** all of today's edits verified against the CSVs (2x2 cells, 9/9,
attribution, flat85, ensemble contrast, coverage, controls table — clean). Fixed on its
findings: tab:crossing h4-6 CI bounds refreshed to the block=max(3,h) values (points/p
unchanged); limitations item 10 and intro contribution 3 scoped to match the abstract
(exception cell no longer glossed); "Five properties" enumeration was actually six with
a duplicated "Fifth" — renumbered; "largest at leads 5-6" corrected (h2 +1.45 > h5
+1.42; largest is lead 6 alone); hist12 p bound 1.5e-3→5.0e-3; r0 bound 1e-15→1e-13;
precip unique 0.572pp → "as much as 0.6"; fusion p 1e-4→6e-5; Africa p .88→.87; fig8
caption "negative elsewhere" precision; conclusions' residual "stage built for it" →
"usable only in propagated-state form".

**Cold-reader audit:** found ONE fresh-clone BLOCKER the batch audits could not see —
phase8_strat reads phase7_resmlp_predictions.csv but ran before phase7_resmlp and never
declared it. Fixed: phase7_resmlp/phase7_lstm moved ahead of the phase-8 steps, the
input declared, and torch-phase outputs now include their predictions files. Also fixed:
README data table completed (indices, Li/PANGAEA 973113, CSR ancillary rows added),
"15 passed"→17 with fresh-clone skip note, five-of-six qualifier, figure-verify
mechanism described honestly (asserts are transcribed into make_figures.py, not read
from the ledger), runtime restated (~1.5-2 days; the 2026-08-16 entry's "~30.5 h" vs
its own step sum of 33.6 h was a misrecorded launch time — step times are authoritative),
stale "winner" framing in CODE_MAP/README rescoped to "best at leads 4-6", stale
STUDY_CONTEXT header rewritten, stale make_figures DISCREPANCY comment cleared,
--list now prints declared I/O, .gitignore README-section pointer fixed, stackdump
litter removed. Chain is now 32 steps / 31 default; declared inputs verified present;
17/17 tests; PDF recompiled clean (40 pp).

## 2026-08-17 — Publication-polish round: prose audit + figure house style

- Why: user judged the manuscript's writing and figures visibly below the register of
  the published papers it engages (Li & Kusche 2026 and typical HESS papers): abstract
  a ~560-word result inventory with 9 inline p-values, lab-notebook/audit-trail voice in
  rendered text, default-matplotlib figures with inconsistent color semantics and a
  print-scaling bug. Two review agents ran (cold-referee prose audit against
  Gopen & Swan / Farquhar / Lipton standards; per-figure critique against AGU/EGU norms),
  findings then applied.
- Manuscript (`paper/main.tex`): title cut 24→12 words (subtitle dropped); abstract
  rewritten ~560→~280 words, 6 numbers, 0 p-values; contribution 3 cut 45→8 lines;
  conclusions rewritten without nested em-dash chains; all ~20 audit-trail/journey-voice
  passages purged from rendered text (protocol-sensitivity content retained, confessional
  framing removed; internal dates now only in % source comments); statistics de-duplicated
  (headline numbers at full precision once; qualitative elsewhere); new ERA5 attribution
  table (tab:attribution) built digit-for-digit from phase6_era5_attribution{,_continent}.csv
  replaces the ~25-number prose dump; terminology unified (lead not horizon;
  "footprint-sharing score" with one-line note that archives name it `contamination`;
  "variant" not "arm"; GRACE-FCast with CSR member named once; "pre-specified" not
  "registered"; all five "honest X" tics removed); crossing section rewritten to interpret
  Table 3 instead of re-reading it; splice and physical-units paragraphs compressed;
  worst 25 sentences from the audit broken up. Appendix Table A2 (results/*.csv paths)
  moved out of the manuscript to docs/ARCHIVE_MANIFEST.md for the Zenodo README; Table A1
  (name→identifier) retained. Caption contradictions fixed (F3 "hatched"→muted fill +
  filled/open dots; F8 "ribbons"→intervals); includegraphics width bug fixed (17 cm builds
  now \textwidth, was 12 cm = fonts at ~4.5 pt print).
- Figures (`scripts/make_figures.py`, styling only, ledger asserts untouched, data
  identical): Okabe–Ito palette with fixed cross-figure color semantics (Kalman blue,
  stacked ensemble near-black, ridge-on-states vermillion, input-channel orange,
  placebos gray; vik retained only as the F3 map colormap); house rcParams (8/7 pt,
  no top/right spines, inward ticks, y-grid, capsize 0, fonttype 42); filled=DM p<0.05
  convention annotated on F2/F5/F8. F1 broken axis + direct labels (legend deleted);
  F3 cartopy Robinson + coastlines, non-FDR basins muted, extend="both" colorbar
  (cartopy 0.25/pyproj/shapely added to venv and requirements-lock.txt, regenerated
  88→95 pins); F4 rebuilt (null-distribution strips absorbing the floating IAAFT
  annotation + signed bars + caterpillar); F6 rebuilt (gray cloud, Africa/Europe
  highlighted, Theil–Sen line, quadrant labels, clip triangles; median table and
  on-data legend deleted); F8 de-whiskered, terciles one-hue (darkest=high),
  figure text says "footprint sharing" matching the paper. Known deviations recorded
  in the script: F5 single-seed markers dropped (seed-1 channel contrast absent from
  phase8b_h16_headline.csv); F4 IAAFT row uses summary marks (no per-surrogate file).
- State: PDF 39 pp (was 40; abstract cut), 0 undefined refs, 17/17 tests pass.
  Zenodo DOI remains the only submission blocker. Availability statement still says
  "upon acceptance" — replace with the minted DOI before submission (Copernicus wants
  it reviewable at submission).

## 2026-08-17 — HESS house-style pass: dash/bold reduction, US spelling, journal-spec check

- Verified HESS/Copernicus submission specs from the live official pages: rolling
  submission (no deadline; scheduled-special-issue page currently lists none, so a
  regular research article is the only route); no word/page limit (flat APC EUR 1800
  net since 2025, EGU members EUR 1620); Copernicus LaTeX package v7.14 (matches ours);
  figures min 8 cm wide, 300 dpi, one font family, CVD-safe, panels labeled (a)/(b) —
  all already satisfied; competing-interests null wording already exact; data/code
  policy expects FAIR-repository DOIs citable at submission (Zenodo DOI remains the
  one blocker). Copernicus English guidelines prefer spaced en dashes over em dashes
  for breaks, sentence-case headings (already true), consistent spelling variety.
- Prose pass on paper/main.tex, punctuation/formatting only: all ~95 spaced-dash
  parentheticals in rendered text rewritten (commas, parentheses, colons, or sentence
  breaks; 0 remain), the lone em dash removed; every \textbf in running prose removed
  (contribution/property/limitation lead-ins now \emph; table \mathbf values kept;
  four bolded table-cell verdicts in tab:controls now plain); a handful of stray
  \emph intensifiers dropped. Spelling unified to US (Neighbouring/unmodelled/favour x2).
- Verification: git word-diff numeral multiset identical before/after (no quantitative
  content touched); PDF recompiled clean, 39 pp, 0 undefined refs/citations.
