# Phase 6 Analysis — Where the ERA5 Gain Lives (2026-08-12)

Interpretation pass over the audited Phase 6 ERA5 results. Everything below is computed fresh
from `phase6_era5_predictions.csv`, `basin_meta.csv`, `phase2_strata.csv`,
`era5_basin_coverage.csv`, and an exact refit of the fold ridges (scripts in the session
scratchpad; the refit reproduces the archived per-fold skill to the second decimal, so the
ablations are on the engine's own construction, not an approximation). Headlines already in
the RUN_LOG (+5.17% pooled at h1; GBM/MLP tie ridge; neighbor survives conditioning at
+0.485%) are taken as given. All of this is exploratory and must be labeled so in the paper.
Derived tables written alongside: `phase6_perbasin_era5_skill_h1.csv`,
`phase6_perbasin_fdr_era5_h1.csv`, `phase6_perbasin_fdr_neighbor_{era5,noera5}_h1.csv`,
`phase6_era5_vargroup_ablation_h1.csv`.

---

## 1. The ERA5 gain is broad-based — except in Africa, where it is zero

Per-basin skill of `ridge_own_era5` vs `ridge_own` at h1 (n = 234): median +6.7%,
174/234 basins positive (74%), IQR −0.1% to +13.5%. Concentration is moderate, not extreme:
the top 25 basins carry 52% of the pooled loss reduction, the top 50 carry 80%.

The continental pattern is the mirror image of the neighbor effect:

| continent | n | pooled skill | 95% CI | DM p | per-basin win rate |
|---|---|---|---|---|---|
| north_america | 46 | **+7.78%** | +3.97..+10.48 | 2e-5 | 83% |
| europe | 30 | **+7.29%** | +4.40..+10.51 | 3e-5 | 83% |
| asia | 74 | **+6.37%** | +3.86..+8.88 | 3e-6 | 82% |
| australia | 19 | **+5.00%** | +2.86..+7.50 | 5e-4 | 79% |
| south_america | 27 | +1.69% | −7.21..+7.78 | .61 | 59% |
| **africa** | 38 | **+0.30%** | −5.22..+4.78 | .90 | 50% |

Africa — the continent where the neighbor effect is ~3x global — gets **nothing** from ERA5.
The reason is a tropical belt of large negative outliers: the five biggest pooled-loss
increases are all African (E_Lower_Lake_Chad −32.3%, R_Congo_River −65.6%,
C_Gulf_of_Guinea_Coast −29.8%, C_South_Mediterranean_Coast −13.2%, C_Nigerian_Delta −7.3%),
and the single worst basin on Earth is **R_Amazon_River at −99.1%** (ERA5 nearly doubles its
h1 MSE; its own-arm errors are tiny, so the pooled damage is modest, but per-basin it is a
catastrophe). This is consistent with the documented weakness of ERA5-Land precipitation in
the deep tropics (sparse gauge constraint over the Congo/Amazon). Meanwhile the biggest
winners are high-latitude: C_East_Barents_Sea_Coast +36.3%, R_Yenisey_River +32.3%,
R_Saskatchewan_Nelson_Rivers +32.2%, C_East_Ob_Bay_Coast +30.2%; warmer basins gain
systematically less (Spearman rho(mean t2m, skill) = −0.18, p = .007).

## 2. Strata: ERA5 gains live exactly where the neighbor gains don't

By phase-2 SNR stratum, pooled h1 skill of ERA5:

- low +6.05% (CI +3.56..+8.59, n = 111), sub_noise +5.96% (+3.76..+7.92, n = 62),
  mid +2.92% (ns, n = 48)
- **high −24.7% (CI −50.2..−5.7, n = 5)** — the stratum where the neighbor effect was
  largest (+5.16% in phase 5). Individually: C_South_Alaska_Coast_Frazer_River −85.7%,
  C_North_Brazil_South_Atlantic_Coast −40.2%, C_Ellesmere_Island −21.2%,
  C_Capim_River_Delta −8.8%; only R_Sao_Francisco_River (+14.7%) gains.
- glaciated +0.01% pooled, median −4.8%, 3/8 positive (neighbor: +1.71% in phase 5).

So the two information sources are stratified oppositely: the neighbor helps where the
basin's own signal is strong (big regional anomalies to share); ERA5 helps where the signal
is weak or noisy (exogenous forcing substitutes for a poorly observed state) and actively
hurts strong-signal and ice basins, whose storage dynamics (glacier mass, big-river routing)
ERA5-Land's land-surface water budget does not represent. Larger basins also gain more
(area-quartile pooled skill +2.70% → +6.10% from smallest to largest; Spearman rho = +0.23,
p = 5e-4), the footprint-averaging direction you'd expect.

## 3. Per-basin FDR: 27 helped / 3 hurt — a much cleaner sheet than the neighbor's 9/7

`per_basin_dm_fdr` (q = 0.10) for ridge_own_era5 vs ridge_own at h1: **30 of 234 basins
significant, 27 helped, 3 hurt** (E_Lower_Lake_Chad, R_Congo_River,
C_South_Alaska_Coast_Frazer_River). Winners by continent: Asia 9, North America 8, Europe 5,
South America 3, **Africa 2**. R_Yenisey_River and R_Yangtze_River — both significantly hurt
by the neighbor in phase 5 — are FDR-significant ERA5 *winners*. Full table:
`phase6_perbasin_fdr_era5_h1.csv`. Contrast with the neighbor's two-sided 9/7 split: ERA5 is
a conventional, mostly-one-sided predictor; the neighbor is a double-edged one.

## 4. What carries the gain: soil moisture ≈ half, precip a quarter, temperature nothing

Leave-one-group-out and one-group-only ablations of the h1 ridge, exact engine construction,
pooled over all 5 folds (own-only ridge as reference; full ERA5 = +5.17%):

| group (lags 0-2) | drop-group skill | loss vs full | own+group only |
|---|---|---|---|
| soil moisture (swvl1-4) | +2.66% | **−2.51pp** | **+3.14%** |
| evaporation | +3.83% | −1.34pp | −0.97% |
| precipitation | +4.27% | −0.90pp | +2.07% |
| runoff (ro/sro/ssro) | +4.68% | −0.48pp | +1.30% |
| snow (SWE) | +5.05% | −0.12pp | +1.03% |
| temperature (t2m) | +5.24% | **+0.07pp** | +0.27% |

- **Soil moisture is the primary carrier**: alone it delivers 61% of the full gain, and
  removing it costs half. Sensible — swvl is the ERA5 state variable closest to TWSA itself.
- **Precipitation is second but is also the destabilizer**: own+precip alone is +5.8/+5.3/+5.7%
  in folds f1/f2/f5 but **−5.6% in f3** (see finding 5). Soil moisture alone is positive in
  every fold including f3 (+2.8%).
- **Evaporation only works in combination** (alone −0.97%, but dropping it costs 1.34pp): it
  corrects the water-input variables rather than carrying signal itself (P−E logic).
- **Temperature contributes nothing** (dropping it slightly improves skill) and snow is
  negligible. The gain is water-state information, not energy-state.
- Coefficient magnitudes on the standardized features agree: sum |coef| soil 0.39 > precip
  0.34 > runoff 0.32 >> snow 0.09 > evap 0.06 > t2m 0.014; the single largest coefficient in
  every fold is precip lag-0.

## 5. Fold stability: 4 of 5 folds strongly positive; the f3 dip is tropical-precip
misinformation, not the Libya flood

Per-fold h1 skill of ridge_own_era5 vs ridge_own: f1 +8.92% (p = 8e-6), f2 +6.37%
(p = 2e-4), **f3 −0.87% (p = .69)**, f4 +5.08% (p = .014), f5 +8.32% (p = 3e-5). The h2
gain shows the same shape (f3 is the only negative, −2.66%, and is nominally significant in
the wrong direction, p = .036).

Forensics on f3 (test window 2022-04..2023-08):

- The five worst pooled months of the whole experiment are all in f3: 2023-03 (hurt driven by
  Madagascar, Amazon, Congo), 2022-10 and 2022-11 (West Africa / Gulf of Guinea belt),
  2023-08. Dropping the three worst f3 basins (Gulf of Guinea, Congo, Amazon) moves f3 from
  −0.87% to +0.29%.
- The variable ablation localizes it: **drop-precipitation turns f3 positive (+3.55%) and
  precip-alone is f3's disaster (−5.57%)**, while soil moisture alone stays positive in f3.
  The window is the tail of the 2020-23 triple-dip La Niña; the pattern reads as ERA5-Land
  precipitation feeding the ridge wrong interannual rainfall anomalies over the poorly gauged
  tropics for ~a year.
- **The Libya winsorization held.** 2023-09 (Storm Daniel) is among the worst months, but its
  hurt is carried by Siberian-coast basins (C_Olenyok_Delta, C_Banks_Island), not by anything
  Libyan — no residual detonation from the ±10σ clip.
- The gain is not event-driven in the helpful direction either: ERA5 helps in 81% of the 84
  test months, and excluding the single best month (2024-10) only moves the pooled number
  from +5.17% to +5.01%.

Honest summary for the paper: "+5.17%" is really "+5 to +9% in four folds, ~0 in the fold
dominated by La Niña tropical rainfall errors" — worth a sentence, since it marks the
failure mode of exogenous forcing (reanalysis quality, not model capacity).

## 6. ERA5 does not change WHERE the neighbor helps — and Africa's neighbor effect
actually strengthens under conditioning

- Per-basin neighbor DM statistics (corr_top1 vs own) with and without ERA5 conditioning
  correlate at **Spearman rho = 0.857** (Pearson 0.864, n = 234) — far above the 0.68-0.82
  cross-architecture concordance from phase 5. Conditioning on local weather barely moves the
  skill geography, which is what the "TWSA-specific spatial information" reading predicts.
- Sanity note worth recording: the no-ERA5 per-basin DM stats recomputed on phase-6 rows are
  *bit-identical* to the phase-5 file — the ERA5 merge dropped no rows, so all phase 5/6
  comparisons share one row set.
- The phase-5 16-basin FDR set: **all 16 keep their sign under ERA5 conditioning; 11/16
  remain individually FDR-significant** (dropouts: E_Tarim_He_Lop_Nur,
  C_East_Barents_Sea_Coast, R_Yangtze_River, R_Amur_River, C_Southwest_Mediterranean_Coast —
  all still same-signed). The conditioned FDR set *grows* to 24 (14 helped / 10 hurt): once
  ERA5 absorbs weather variance, more basins individually resolve. New winners include
  R_Tocantins_River, C_Novaya_Zemlya, C_Svalbard, C_Ireland, C_Indian_Ocean_Africa_Coast;
  new significant losers include R_Yukon_River, E_Plateau_of_Tibet, E_Lower_Lake_Chad,
  C_Angola_Coast (`phase6_perbasin_fdr_neighbor_era5_h1.csv`).
- **Yenisey is still significantly hurt by the neighbor** (DM +3.43, p = .001) — its fourth
  straight architecture/conditioning loss. But ERA5 gives Yenisey +32.3%, its second-largest
  single-basin ERA5 gain in the sample. The moral is quotable: the information a correlated
  neighbor imports into Yenisey as noise, local forcing delivers cleanly.
- **Africa**: the neighbor effect *conditioned on ERA5* is **+1.91% (CI +0.96..+2.82,
  DM p = .0002)** vs +1.32% unconditioned — conditioning sharpened it, presumably because
  ERA5 soaks up weather variance without touching the shared TWSA signal. The Africa
  highlight survives its hardest control with a better p-value than it started with.

## 7. ERA5 and the neighbor are (mildly) substitutes at the basin level — and complements
in the portfolio sense

Per-basin ERA5 gain vs per-basin neighbor gain: Spearman rho = −0.121 (p = .065);
vs the ERA5-conditioned neighbor gain, rho = **−0.220 (p = .0007)**. The neighbor tends to
help most exactly where ERA5 helps least (Africa is the extreme case). So the two sources
are not redundant — they cover each other's dead zones — which is the strongest one-line
justification for keeping both in a forecasting system, and it explains mechanically why the
pooled neighbor increment is unchanged by conditioning (+0.488% → +0.485%).

## 8. The island-coverage screening: conclusions unchanged, but the count in the audit is 13,
not 5

Correction to the audit/RUN_LOG advisory: `era5_basin_coverage.csv` has **13 keep-basins
below 0.5 coverage**, not 5 — the audit named only the worst five and missed e.g.
C_East_Timor_Islands (0.324, below Jamaica's 0.333), C_Halmahera_Islands (0.378),
C_Sardinia_and_Corsica (0.418), plus three Canadian-Arctic archipelago basins (0.438-0.462)
and C_Sulawesi (0.494). Substantively it doesn't matter: ERA5 mostly *helps* the low-coverage
islands (Sulawesi +16.1%, Halmahera +9.0%, Nusa_Tenggara +8.6%, East_Timor +7.8%; only
Solomon_Islands −11.3% and two Arctic archipelagos are negative). Screening all 13 moves the
pooled h1 ERA5 gain to +5.42% (CI +3.06..+7.56) and the conditioned neighbor increment to
+0.506% (CI +0.16..+0.93, p = .012). Report as a robustness line; no screening needed in the
headline.

## 9. MLP seed spread: unchanged on own arms, tripled on the neighbor+ERA5 arm — the
"three-way tie" is two seeds deep

Across-seed h1 skill (vs ridge_own) spreads: mlp_own 0.15pp, mlp_own_era5 0.13pp,
mlp_corr_top1 0.24pp, **mlp_corr_top1_era5 1.84pp** (+5.54 / +5.70 / +3.86% for s0/s1/s2).
Richer features did not stabilize the MLP; on the fullest feature set one seed in three
loses ~1.8pp of skill. At h2-h3, all MLP seeds on all arms are worse than their ridge twin
(9/9 arm-horizon cells). So the summary's "gbm/ridge/mlp three-way tie at RMSE ≈ 1.015"
holds for 2 of 3 MLP seeds; the honest phrasing is "GBM ties ridge; MLP ties ridge for most
seeds and never beats it." GBM itself is remarkably close to deterministic parity
(gbm_own_era5 +5.12% vs ridge_own_era5 +5.17%).

## 10. Smaller observations

- ridge_own_era5 beats the raw Kalman bar in 74% of basins at h1 (median per-basin +6.5%) —
  ERA5 is the first addition in the project that lifts the *majority* of basins over the
  phase-2 bar.
- The best pooled h1 month for ERA5 (2024-10) is carried by arid-belt basins (Central
  Mongolia, Northern Kordofan, Gobi, Badain Jaran, Upper Lake Chad) — post-monsoon soil
  moisture information in dry regions, consistent with finding 4.
- Congo is hurt in only 48/84 months but the hurt months are huge (2023-03 alone −3.5 in
  monthly mean-loss units) — episodic reanalysis error, not a constant bias.

---

## Ranked implications for the paper

1. **The mechanism table gets a new column and a cleaner story.** ERA5 (+5.17%) and the
   neighbor (+0.49%) are anti-correlated across basins (rho ≈ −0.22), stratified oppositely
   (ERA5: weak-signal, high-latitude, large basins; neighbor: strong-signal basins, Africa),
   and the neighbor's geography is invariant to conditioning (rho = 0.857). One
   figure — per-basin ERA5 gain vs neighbor gain scatter, colored by continent — carries
   findings 1, 2, 6, 7 at once and is the single best addition from this phase.
2. **Africa's dual role is now the paper's sharpest paragraph**: zero pooled ERA5 gain
   (Congo-belt reanalysis errors) yet the neighbor effect strengthens to +1.91% (p = .0002)
   under ERA5 conditioning. Spatial TWSA information is most valuable precisely where
   reanalysis forcing is least trustworthy — a policy-relevant sentence for data-sparse
   regions. (Exploratory subgroup; label as such.)
3. **Report the ERA5 headline with its fold caveat and FDR sheet**: "+5.17% (27 basins
   FDR-helped / 3 hurt), driven by soil moisture ≈ half and precipitation ≈ quarter;
   ~0 in the one fold where La Niña-era tropical precipitation errors dominate; temperature
   and snow contribute nothing." The variable ablation
   (`phase6_era5_vargroup_ablation_h1.csv`) is cheap, engine-exact, and referee-friendly.
4. **Yenisey closes its arc**: hurt by the neighbor under four architectures/conditionings,
   but ERA5's #2 winner (+32.3%). Use it as the closing line of the "when spatial context
   hurts" subsection.
5. **Amend the two-sided caveats**: the Amazon (−99.1%) and the high-SNR stratum (−24.7%)
   are ERA5's own hurt list — exogenous forcing is not a free lunch either, and the paper's
   "know when to use which source" discussion should cite both directions symmetrically.
6. **Housekeeping**: correct the audit advisory to 13 low-coverage basins (not 5); add the
   screened numbers (+5.42% / +0.506%) as a robustness line; phrase the MLP result as
   "2/3 seeds tie" (finding 9); note the bit-identical row-set re-anchor (finding 6) in the
   RUN_LOG as the cross-phase comparability guarantee.
