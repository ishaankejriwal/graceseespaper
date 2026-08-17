# Project Context — Global TWSA Forecasting Study

Last updated: 2026-08-17. Status: **corrected rerun COMPLETE for every phase (14/14 chain
steps, 2026-08-16); all 7 figures built and assert-checked; manuscript repaired against the
2026-08-17 external audit (four false statements fixed, last RERUN todo filled by a new
ERA5 variable-attribution experiment, metadata complete) and compiles clean at 40 pp with
zero undefined references. Remaining before submission: mint the Zenodo DOI. The "small
basins" framing is RETIRED -- footprint contamination is the real stratifier, with one
flagged exception cell (see the corrected findings below). "Delivery decides" is also
RETIRED -- representation (the propagated state) is what separates the arms.**

> Naming note: `docs/history/africa_pilot_jpl.md` and `docs/history/africa_pilot_csr.md` are the older Africa-only
> work (JPL and CSR respectively). This file covers the current global study. For code layout
> see `docs/CODE_MAP.md`; for per-phase detail see `results/*_analysis.md` and `results/RUN_LOG.md`;
> for the paper see `paper/`; for the audit see `docs/AUDIT_2026-08-13.md`.

## The 2026-08-13 repository audit — what it changed

An end-to-end audit (`docs/AUDIT_2026-08-13.md`) was independently verified and triaged. Confirmed
real and now **fixed in code**:

1. **CSR month assignment (P0-1).** Midpoint binning collapsed the Nov 2011 and May 2015
   solutions into Oct 2011 / Apr 2015 averages. The file's own `months_missing` attribute
   proves the mapping (290 span months − 33 documented missing = 257 solutions exactly).
   Fixed in `src/gracefc/basins.py` (`assign_solution_months`, with span-overlap assertions);
   basin table rebuilt: 257 months, was 255.
2. **Issue-date leakage at leads 2–6 (P0-2).** Folds split on *target* dates while every
   transform was fit on all data before `test_start`, so a lead-h forecast used transforms
   fit up to h−1 months after its issue date. Fixed: fold membership is now defined on
   **issue dates** (`evaluate.split_fold`, shared by all five engines) — the model is frozen
   at fold start and used forward. All horizons now share the same issue window.
3. **Neighbor effect concentrates in sub-resolution basins (P0-3).** Verified as stated --
   but SUPERSEDED on 2026-08-15: basin size was a proxy. The real variable is
   **contamination** (share of a basin's mascon-weighted signal from land outside the
   basin). The 2x2 shows the linear lead-1 effect at full size *inside* resolved basins
   when contamination is high (+1.16%, p=.0045) and absent elsewhere (-0.50%). CSR tiles
   were recovered empirically (~12,000 km2 equal-area, so 90,000 km2 is ~7 mascons, and
   basin areas are our own mask integration -- no external cross-check exists).
4. **Phase-8 stack is post-selection (P0-4).** Architectures were chosen and evaluated on the
   same five folds. The stack is now labeled **exploratory** in the manuscript; the
   within-backbone placebo-verified increment is the protected quantity.
5. **Kalman `r` is not identified as measurement noise (P0-5).** Verified: 273/1170 fits at
   the r<1e-6 boundary, 32 basins in all folds, q/r spans 10 orders. Manuscript language
   changed to "AR(1) state-space filtering"; identification caveat added; r=0 ablation queued.
6. **"Dead or harmful" LSTM input-channel claim was seed-selective (P0-6).** Verified: ERA5-arm
   ensemble contrast is nonsignificant at all leads (+0.43/−0.12/−0.31%); seed 1 is
   significantly *positive* at leads 1–2. Manuscript now says "seed-unstable, no robust
   ensemble gain".

Pre-audit result artifacts are frozen in `archive/pre_audit_2026-08-13/` with a SHA256
manifest. New results land in a clean `results/`.

## The question, and how it evolved

We started by asking: does knowing what's happening in *other* river basins help forecast a
basin's own water storage? Answering that honestly forced two prior questions — what's the right
baseline, and what's actually left to predict once trend and seasons are removed — and the answers
to those turned out to be the bigger findings.

Setup: 234 global HydroSHEDS basins (CSR RL0603M mascons, deseasonalized + per-basin standardized
target), leads 1–6 months, 5 expanding-window folds with issue windows covering 2019-06 to
2026-05, Diebold–Mariano tests + FDR throughout.

## The three findings (= the paper's three contributions), CORRECTED FINAL numbers (2026-08-15)

**1. The field's benchmark leaves its own headroom unexploited -- state-space filtering claims
it.** The three-parameter Kalman filter beats damped persistence by +5.0/+8.8% at leads 1/2
(+2.5..+5.6% at 3-6) and per-basin ridge at **five of six** leads (h4 ns, p=0.078). The
margin did NOT grow under the corrected protocol -- framed as robustness, never growth. The
identification caveat on r stands (259/1170 boundary fits).

**2. The crossing, now measured against our best system.** vs Li & Kusche's full product
(227 matched basins x 60 months): **+20.0% lead 1 / -3.5% ns lead 2 / -12.9..-30.3% leads
3-6** -- crossover between leads 2 and 3. Their lead 1 is 11% *worse than damped persistence*
(p=.057): the cleanest filtering-vs-forcing statement in the study. Concept old (Wood &
Lettenmaier 2008); locating it on observed TWSA with two real systems is the contribution.

**3. Neighbours: two-sided.** LINEAR tier: a controlled null (+0.31% lead 1, p=0.199;
negative at 2-6; correlation-selection still beats 50/50 random graphs at lead 1 -- the two
questions are kept separate). What pooled effect exists sits in high-contamination
(footprint-sharing) basins regardless of size -- a leakage signature. NONLINEAR tier: the
same information delivered as a residual-correction stage on the LSTM adds
**+0.91/+1.45/+1.33/+1.32/+1.42/+1.96%** at leads 1-6 (2-seed ensemble, all p <= 3.2e-8,
20/20 seed-matched placebos in all 12 cells), and the effect is positive in every
contamination/size stratum with NO concentration in shared-footprint basins (interaction ns
at every lead) -- the leakage artifact is ruled out for the stacked correction; small basins
gain least (the opposite of leakage's prediction). Mechanism deliberately unclaimed.
Delivery matters: the same neighbor state as an LSTM *input channel* is ns-to-negative.

Supporting results: ERA5 own-basin forcing +4.6% at lead 1 (largest single source, lead-1
only); ERA5 sequence modeling +1.4-2.4% at leads 1-3; the one nonlinear GRACE-only neighbor
survivor is the residual-MLP (20/20 placebos); GNN dead; corrected IAAFT surrogates: real
beats 99/99 at lead 1 only (time-alignment confirmed), beaten at 2-6, consistent with the
null.

## Novelty (audited twice, independently — see memory + `results/phase8_related_benchmarks.md`)

- The noise/benchmark argument has no precedent found in any satellite time-series field. Cite
  Niraula & Goessling 2021 (sea-ice analogue), the GRACE-DA literature (different thing), and
  Nie et al. 2025 (linear-beats-DL, but with autoregressive inputs deliberately excluded — our setup).
- Neighbour claim is now worded as first *characterization with null-model controls* on
  observed GRACE — not "isolation and validation" — because of the sub-resolution concentration
  (Steidl & Zhu 2025 built a basin-graph forecaster, but on reconstructed data, no nulls).
- Crossing claimed as a measurement, never as a concept.
- Must acknowledge the public FLDAS-Forecast hindcast and justify non-comparison.

## Where things stand

**Done (as of 2026-08-15):** corrected rerun of every headline phase (2, kalman, 3b, 6-ERA5,
7, 8, 8b) with audits; resolution/contamination audit (tile recovery, contamination metric,
2x2) + stacked-correction stratification with formal interaction tests (audited, all spot
checks exact); manuscript fully rewritten from `paper/notes/REWRITE_LEDGER.md` (the single source
of truth for final numbers + framing rules), adversarially audited (GO; two p-bound slips
fixed, two universal claims scoped), then voice-passed to zero internal jargon (model names
defined in a Methods table; code identifiers only in the reproducibility appendix; rules in
ledger sect. 8b + the ml-paper-writing skill at `.claude/skills/`); MiKTeX installed, compile
clean (33 pp, 0 undefined refs); figures F1/F2/F5/F8 built + verified
(`scripts/make_figures.py`, `figures/BUILD_NOTES.md`); corrected jump screen + IAAFT
surrogates landed.

**In flight:** supporting-phase rerun chain (phase-5 nonlinear/coupled/fusion, hybrid
splice); basin analysis rerun after phase5_coupled (dependency).

**Next (in order):** chain lands -> rerun basin analysis -> RUN_LOG batch entry + batch
audit -> fill the five remaining todo-RERUN markers -> build F3/F4/F6/F7 -> uncomment figure
includes -> final compile; then authors/affiliation, Zenodo DOI, fetch two paywalled papers
(Niraula & Goessling 2021, Kankanige 2026). Revision-stage (non-blocking): out-of-fold
stage-2 residuals; r=0 Kalman ablation.
