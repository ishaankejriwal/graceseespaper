# Phase 6 basin analysis — which basins benefit from neighbors, and can we predict it?

**Script:** `scripts/run_phase6_basin_analysis.py` · **Data:** phase3b / phase4 / phase6_era5 prediction
files, `phase6_li_comparison_perbasin.csv`, `phase2_strata.csv`, `basin_meta.csv`, ERA5 coverage,
Kalman fold params, phase5 coupling. All per-basin numbers are h1, 234 basins × 84 test months
(2019-06..2026-05, folds f1–f5). Per-basin benefit = 1 − MSE(corr_top1)/MSE(own_ridge); DM stat on
the monthly loss differential (negative = neighbor helps). Neighbor identity/correlation/distance
reconstructed with the exact fold-specific `corr_topk` graphs the experiments used.

**One-line verdict:** the neighbor effect lives in small-to-mid, tropical/African, stable-neighbor
basins and is actively harmful in large rivers; honest fold-wise selection (use the neighbor only
where it helped in the other folds) more than doubles pooled skill from +0.49% to +1.13% (DM
p<1e-4); ERA5 conditioning barely changes who benefits (rank ρ=0.86), confirming the effect is not
shared weather.

---

## Q1 — What predicts whether a basin benefits?

Spearman correlation of each covariate with the per-basin DM stat (negative DM = helped, so a
**negative r means the feature predicts benefit**):

| feature | r vs DM | p | reading |
|---|---|---|---|
| nbr_stable (same neighbor all 5 folds) | −0.27 | <1e-4 | stable neighbor → benefit |
| log10_area | +0.28 | <1e-4 | big basins → hurt |
| era5_coverage | +0.17 | .008 | area proxy, not independent (see below) |
| nbr_corr (train corr with neighbor) | −0.17 | .010 | stronger correlation → benefit |
| nbr_dist_km | +0.15 | .027 | nearer neighbor → benefit |
| coupling (phase5 c) | −0.09 | .16 | ns, matches phase5 |
| glaciated / rho / log(q/r) / latitude | \|r\|≤0.08 | ns | no signal |

- **Neighbor-graph quality is the top predictor.** Basins whose top-correlated neighbor is the same
  in every training window: median skill **+1.18%** (66% helped, n=142). Unstable-neighbor basins:
  **−1.33%** (41% helped, n=92). Instability means the "neighbor" is a noise artifact of the
  correlation ranking.
- **Basin area is the top hazard.** Largest tercile: median **−1.37%**, only 41% helped; the
  most-hurt list is exactly the great rivers (Yenisey, Paraná, Yangtze, Amur, Amazon-scale systems).
  Two mechanisms, both visible in the data: (i) large basins already spatially average out mascon
  noise, so denoising has nothing to add; (ii) their top-correlated "neighbor" is often absurdly far
  (42% of large basins pick one >1000 km away, vs 14% of small ones — Paraná's modal pick is Junggar
  Pendi at 9,547 km, a pure spurious correlation). The area effect survives within both stability
  classes (r=+0.19 stable, +0.37 unstable).
- **era5_coverage is not a real predictor**: it rank-correlates 0.62 with area, and among
  below-median-area basins its association with benefit vanishes (r=−0.02).
- **SNR stratum:** high-SNR basins are the best stratum (median **+4.85%**, 5/5 helped), then
  glaciated (+1.21%, 7/8), sub_noise (+0.76%), mid (+0.62%), low (−0.03%). Matches the phase5
  stratum result.
- **But overall predictability is weak.** A random forest on all covariates (continent + stratum
  one-hots included) reaches cross-validated R² of only **0.02–0.04**. Permutation importance ranks
  log10_area ≫ nbr_stable > log(q/r) > nbr_corr, consistent with the Spearman table. Most cross-basin
  variance in the per-basin skill estimate is sampling noise on 84 months — which is why selection on
  covariates underperforms selection on past performance (Q2).

## Q2 — Should neighbor info be applied selectively? Yes.

Pooled h1 skill vs own_ridge when the neighbor arm is applied only to selected basins (reference arm
fills in elsewhere; identical row sets across strategies):

| strategy | selection is honest? | Kalman backbone | ERA5-conditioned ridge backbone |
|---|---|---|---|
| apply to all 234 | — | +0.49% (DM p=.022) | +0.48% (p=.010) |
| oracle (in-sample sign) | no — upper bound | +1.77% (p<1e-4) | +1.75% (p<1e-4) |
| **leave-one-fold-out empirical** | **yes** | **+1.13% (p<1e-4), ~132 basins** | **+1.09% (p<1e-4), ~127 basins** |
| RF on static covariates (double-CV) | yes | +0.60% (p=.002) | +0.46% (p=.002) |

- The honest, implementable rule — *use the neighbor only where it beat own_ridge in the other four
  folds* — **more than doubles** the pooled effect (+0.49% → +1.13%) and captures ~64% of the oracle
  ceiling. Per-basin benefit persists across time folds; that persistence, not any static covariate,
  is the usable selection signal (cross-architecture Spearman 0.68–0.82 in phase5 said the same).
- Covariate-based selection helps only marginally (+0.60%) — consistent with the weak RF R².
- Same picture on the ERA5-conditioned backbone, so the selective gain is not a weather artifact.

## Q3 — Geographic patterns

| continent | n | median skill | % helped | median skill (ERA5-cond.) |
|---|---|---|---|---|
| africa | 38 | **+1.43%** | 58% | **+2.94%** |
| north_america | 46 | +1.16% | 65% | +0.07% |
| south_america | 27 | +0.98% | 59% | +0.72% |
| asia | 74 | +0.62% | 58% | +0.19% |
| europe | 30 | −0.54% | 47% | −1.20% |
| australia | 19 | −0.57% | 37% | −0.31% |

By latitude band: tropics best (median +0.98%, 79 basins), then N high-latitude (+0.58%), southern
extratropics (+0.46%), northern mid-latitude weakest (+0.28%). Distance band: benefit peaks at
300–600 km (median +0.99%, 19% significantly helped) and dies >1000 km (−1.01%) — the same
300–1000 km regional scale the phase3b/4 controls established. <300 km is a small noisy group
(n=15) with no advantage, consistent with leakage being ruled out rather than driving the effect.

Africa is the standout and **strengthens** under ERA5 conditioning (+1.43% → +2.94% median), while
North America's median drops toward zero — Africa's neighbor signal is TWSA-specific spatial
information, not shared rainfall; part of North America's apparent benefit at h1 was shared
meteorology that ERA5 features absorb.

## Q4 — Does ERA5 conditioning change which basins benefit? Barely.

- Replication check passed: the no-ERA5 ridge arms in phase6 reproduce phase3b per-basin DM stats
  **exactly** (max |diff| = 0.0), so the comparison is clean within one file.
- Per-basin DM ranks, unconditioned vs ERA5-conditioned: **Spearman 0.857** (skill: 0.83). Index
  conditioning (phase4, ENSO/IOD) changes essentially nothing: **0.997**.
- 32/234 basins (14%) flip benefit sign under ERA5 conditioning, but flips concentrate where the
  effect was already ~zero (median |DM| of flippers 0.45 vs 1.07 overall). Significantly-helped
  basins: 30 unconditioned → 34 conditioned, overlap 22.
- Li & Kusche long-lead edge (per-basin DM at h4) is essentially unrelated to neighbor benefit
  (ρ=−0.14): the exogenous-forcing basins and the neighbor-denoising basins are different, weakly
  overlapping sets — supports framing the two mechanisms as complementary.

## Caveats

- Per-basin skill on 84 months is noisy; the DM stat is the more stable measure and all headline
  patterns are stated on it or on medians, not means (a few basins have skill tails to −30%).
- The LOFO selective gain is honest in time (selection never sees the evaluated fold) but folds share
  the post-2019 climate era; a fully out-of-era test isn't possible with this record.
- The RF-selection covariates (nbr_stable, fold-averaged rho/log_qr, coupling) aggregate over all
  five folds, so they are not strictly fold-causal for early folds — structural, not target leakage,
  and RF is the weakest strategy anyway (audited).
- `era5_coverage` and continent/stratum tables are descriptive; only area, neighbor stability, and
  neighbor correlation survive as (correlated) independent-ish predictors, and none predicts well
  enough to replace performance-based selection.
- Nile Delta pair illustrates asymmetry: West Nile Delta is a top-10 winner using East as neighbor;
  East is a top-10 loser using West — benefit is a property of the (target, neighbor) pair, not the
  region.

## Outputs

- `results/phase6_basin_analysis_summary.csv` — 234-basin table: all covariates + per-basin
  skill/DM/p for the four settings (p3b, p4cond, era5r, era5c) + Li h4 DM + bands.
- `results/phase6_basin_analysis_spearman.csv`, `_rf_importance.csv`, `_selective.csv`, `_q4.csv`.

## Suggested next steps

1. Promote the LOFO-selective result to the paper as the "practical deployment" number (+1.13%,
   DM p<1e-4) alongside the +0.49% apply-everywhere headline.
2. Add a neighbor-admission rule to the method: require neighbor stability across training windows
   (or corr threshold) — it is the one *a priori* covariate rule with teeth, and it would have
   excluded most of the harmed large-river basins.
3. Map figure for the paper: per-basin DM stat choropleth with the Africa/tropics cluster and the
   hurt great rivers annotated.
