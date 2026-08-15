# Phase 5 Analysis — What the Complete Result Set Says (2026-08-12)

Interpretation pass over the audited Phase 0-5 results. All numbers below are either taken
directly from audited CSVs or computed fresh from `phase5_coupled_coupling.csv`,
`phase5_perbasin_fdr_h1.csv`, `phase3b_predictions.csv`, `phase2_strata.csv`, and
`data/processed/basin_meta.csv` (scripts in the session scratchpad; trivially rerunnable pandas).
Everything new here is exploratory and must be labeled as such in the paper.

---

## (i) The paper's argument in one arc

The study asks whether spatial context helps forecast a basin's water storage once everything
easy is removed. Phases 0-2 build the honest test bed: deseasonalize fold-safely, then show that
a three-parameter per-basin Kalman filter beats every baseline — including ridge and ML heads —
at all six horizons, meaning most published "skill above persistence" is noise filtering that a
tiny model gets for free. Against that bar, Phase 3b's registered comparison finds that one
correlation-selected neighboring basin still adds +0.4879% pooled skill at one month
(CI +0.14 to +0.90, DM p = .022), and Phase 4's five controls pin down what it is: not chance
(99/99 placebo graphs), not shared ENSO (index conditioning), not satellite blur (unchanged at
>=300 km), but a regional shared hydroclimatic signal living at 300-1000 km that dies past one
month. Phase 5 then attacks the effect with every architecture that "should" enlarge it —
predictive-lag selection (worse than random), a fusion filter treating the neighbor as a second
sensor of the same latent (fails informatively: the neighbor carries persistent local signal),
a bivariate coupled filter with correlated process noise (correct structure, matches but never
beats the ridge bolt-on), GBM/MLP heads (significantly worse than ridge on identical features),
and 2-hop chains (worse than 1-hop). The conclusion is a bounded fact, established twice over:
the spatial information in TWSA fields is real, small, linear, one-hop, and one-month — and it
is heterogeneous, helping significantly in 9 basins (4 in Africa; Africa pooled +1.3%) while
significantly hurting in 7 (mostly large high-latitude rivers). Three recent papers assume
spatial context helps and never ran a control; this study is the controlled measurement of that
assumption, with the Kalman bar as a second free-standing contribution.

---

## (ii) Findings from today's three analyses

### 1. Coupling coefficients c (phase5_coupled_coupling.csv — first look)

**c is large, positive, and a stable basin property — but it is a weak predictor of where the
neighbor helps.**

- Distribution (n = 1170 = 234 basins x 5 folds): median 0.78, mean 0.74, only 6 fits (0.5%)
  negative, 16 (1.4%) near zero (|c| < 0.1), 236 (20.2%) at the 0.94+ upper-bound region —
  matching the audit's saturation note. Fold-to-fold distributions are essentially identical
  (fold means 0.735-0.759).
- Stability: 86.4% of total variance in c is between-basin (within-basin across-fold variance
  0.0054 vs between-basin 0.0345); median per-basin across-fold std is 0.035; 230/234 basins
  keep the same sign in all five folds. The MLE is estimating a real, reproducible basin
  property — near-universal positive regional co-movement of process noise.
- **But c only weakly maps skill.** Against the headline per-basin DM statistic
  (corr_top1 vs own_ridge, sign flipped so positive = neighbor helps): Pearson r = +0.130
  (p = .048), Spearman rho = +0.093 (p = .16). Headline win rate rises from 47.5% in the lowest
  c-quartile to 61% in the highest — a real but shallow gradient. Against the coupled filter's
  own per-basin wins: nothing (r = -0.065, ns). Mann-Whitney on mean c for headline
  winners vs losers: p = .097.
- Verdict for the paper: the planned figure "fitted coupling maps where regional signal lives"
  is **half right**. A world map of c is a legitimate descriptive mechanism figure (regional
  coupling is ubiquitous and stable), but it cannot be sold as predicting where skill appears —
  the honest caption is "coupling is nearly everywhere; usable forecast skill is not, because
  skill depends on the neighbor's noise level and the strength of the basin's own signal, not
  on coupling alone." Report the quartile gradient (47.5% to 61%) as the supporting statistic.

### 2. The headline FDR basins (phase5_perbasin_fdr_h1.csv + basin_meta)

**The "16 FDR basins" split 9 helped / 7 hurt — the paper must report two-sided heterogeneity,
and the RUN_LOG phrasing ("132/234 favor it, 16 pass FDR") must not survive into the write-up
as 16 supportive basins.**

- Winners (9, FDR q = 0.10, treatment better): E_Tarim_He_Lop_Nur,
  C_East_Brazil_South_Atlantic_Coast, E_Lake_Rukwa_Basin, E_Farahrud_Helmand_Hamun_i_Mashkel,
  C_West_Nile_Delta, R_Sao_Francisco_River, R_Niger_River,
  C_South_Hudson_Strait_Ungava_Bay_Coast, C_Southwest_Mediterranean_Coast.
- Significantly hurt (7): R_Yenisey_River, R_Amur_River, R_Yangtze_River, R_Parana_River,
  C_East_Barents_Sea_Coast, C_Aleutians_South_Alaska_Coast (the one glaciated entry),
  C_South_Gulf_of_Oman_West_Arabian_Sea_Coast. Pattern: four of Earth's largest rivers plus
  high-latitude coasts — basins with strong, well-filtered own signal where a top-1 correlated
  neighbor imports noise. R_Yenisey_River is significantly hurt under the headline, fusion, AND
  coupled comparisons — the cautionary poster child, consistent with the discarded Arctic
  long-range artifact from Phase 4.
- **Africa: 4 of the 9 winners (Niger, Lake Rukwa, West Nile Delta, SW Mediterranean Coast) vs
  16% of the sample; 22/38 African basins favor the treatment; pooled Africa-only headline
  skill at h1 is +1.32% (monthly block-bootstrap CI +0.38 to +2.19) — nearly 3x the global
  +0.49% and CI excludes zero.** North America is the other strong region (+1.12%, CI +0.33 to
  +1.91; 30/46 basins favor). Europe (-0.75%) and Australia (-0.86%) are negative, ns. This is
  the Africa-highlight section's quantitative core (exploratory subgroup, label it so).
- SNR strata (phase2_strata.csv): the h1 headline is +5.16% in the 5 high-signal basins
  (CI +2.11 to +7.78), +1.71% in the 8 glaciated (CI +0.18 to +3.08), +0.59% in the 62
  sub-noise, +0.30%/+0.19% in low/mid. Gains concentrate where signal is strong (regional
  anomalies big enough to share) — a useful mechanism note, though the two strong strata are
  tiny and glaciated carries the usual ice-variability caveat.
- Dropping just the 7 significantly-hurt basins raises pooled skill from +0.49% to +0.73% —
  do NOT do this in the headline (post-hoc), but it quantifies the heterogeneity cost and
  motivates "when to use a neighbor" as a discussion point.

### 3. Cross-architecture overlap

**The per-basin skill geography is architecture-invariant — that concordance, not FDR-set
overlap, is the strongest new robustness evidence.**

- FDR sets are asymmetric by construction: headline 16 (9 win/7 lose), fusion 11
  (3 win/8 lose — fusion is pooled-negative), coupled 1 (0 win/1 lose: Yenisey; coupled's tiny
  pooled gain is diffuse, no basin individually significant).
- Set overlap where it can exist: 2 of fusion's 3 winners (E_Tarim_He_Lop_Nur,
  C_East_Brazil_South_Atlantic_Coast) are also headline winners — hypergeometric
  P(overlap >= 2) = .0039. Even an architecture that loses on average wins in the same places
  the ridge bolt-on wins.
- The continuous version is much stronger: per-basin DM statistics correlate across
  architectures at Spearman rho = 0.68 (headline vs coupled), 0.74 (headline vs fusion), 0.82
  (coupled vs fusion), n = 233-234. Three different neighbor-using architectures agree on WHERE
  the neighbor helps and where it hurts. This is a paper figure: scatter of per-basin DM,
  headline vs coupled (or a 3-panel), rho annotated — it converts "the effect survives
  architecture changes" from a table into a mechanism-level statement that the signal is a
  property of the basins, not of the model.

---

## (iii) Ranked next steps toward submission

The experiments phase stays closed — nothing found today argues for new model runs. Everything
below is evaluation, figures, or text. Target: HESS (best fit: methods-critical, open review) or
WRR; J. Hydrology as fallback.

**1. Paper figures + notebook 03 (2-3 days). Do first.**
The figure set is now fully determined, so nothing upstream blocks it, and every later step
(skeleton, comparisons) consumes these figures. Core set: (a) distance-decay of skill —
centerpiece; (b) baseline ladder / Kalman-bar figure; (c) placebo + surrogate null
distributions with the real effect marked; (d) per-basin h1 DM map with the 9/7 FDR basins
highlighted; (e) cross-architecture concordance scatter (rho = 0.68-0.82) — new today;
(f) world map of coupling c (stable, near-universal; descriptive) — new today; (g) horizon
profile h1-h6 showing death after h1; (h) Africa/continent breakout bar with CIs. Notebook 03
rebuilds all from results CSVs alone (repro requirement already registered).

**2. SNR-stratified + continent breakout table of the headline (0.5 day).**
Registered in the plan ("every headline result additionally reported split by basin TWSA
variance") and essentially computed today at h1; extend to h2-h6 and to the placebo controls,
CI via the existing monthly block bootstrap. Buys: fulfills a pre-registered reporting promise,
supplies the Africa section's numbers, and heads off the obvious referee question about where
the half-percent lives.

**3. Paper skeleton draft (2-3 days).**
The argument arc (section i above) is stable and all write-up constraints are collected
(phase5_audit.md recommendations 1-4, plan-file constraints i-iii, the 9/7 two-sided FDR
phrasing, exploratory-vs-registered disclosure). Locking prose structure now prevents the
constraint list from decaying. Include the "when spatial context hurts" subsection (7 basins,
Yenisey case) — it converts heterogeneity from a weakness into a contribution.

**4. Full-TWSA remove-restore evaluation (1-2 days).**
Code exists in src/gracefc/decompose.py, never evaluated. Buys two things: (a) RMSE in cm on
full TWSA — the unit practitioners and referees actually read, and the number that makes the
Kalman-bar contribution legible outside this study's skill metric; (b) the prerequisite for
step 5, since Li & Kusche forecast full TWSA. Low risk: purely an evaluation transform of
existing predictions.

**5. Li & Kusche PANGAEA head-to-head (3-4 days).**
The single strongest addition for referees: the field's only comparable published global
forecast, free (CC-BY, no login, doi.pangaea.de/10.1594/PANGAEA.973113), and the paper
currently criticizes their uncontrolled spatial claim without ever meeting their numbers.
Aggregate their hindcast to our 234 basins, score on overlapping months with our full-TWSA
(step 4) and deseasonalized metrics. Either outcome helps: if the Kalman bar matches or beats
them, the "field's baselines are too weak" claim gets teeth; if not, the controls story still
stands and the comparison shows good faith. Keep it one figure + one table (Phase 6 was always
droppable — cap the effort).

**6. ERA5 ingest + covariate robustness tier (4-6 days) — only if the timeline allows;
otherwise hold as revision ammunition.**
Data is downloaded (590 MB, 2001-2026) but nothing is ingested, and the unit traps
(rate-vs-total, negative evaporation) plus basin aggregation make this the most expensive
remaining item. What it buys: the strongest possible version of the shared-weather mechanism
test (does conditioning on ERA5 precipitation absorb the neighbor effect?) and parity with
Li & Kusche's input tier. But it reopens the experiments phase, the climate-index control
already addresses shared forcing at the teleconnection scale, and either outcome complicates
rather than blocks the current story. Recommendation: submit without it, offer it in response
to reviews — HESS reviewers are likely to ask, and having the data staged makes the revision
cheap.

**Killed or absorbed by today's results:**
- Per-basin neighbor-coefficient map as a separate experiment — superseded: the coupling-c map
  (figure f) plus the DM map (figure d) carry the same content with parameters that already exist.
- Any further architecture work (GNN, deeper fusion variants, k > 1 revisits) — the coupled
  filter is the theoretically correct architecture and it caps at the ridge bolt-on's number;
  the nonlinear null is now general, 2-hop is negative, pred-lag is worse than random.
- "Fitted coupling predicts where the neighbor helps" as a headline mechanism claim — measured
  today at r = 0.13: report the quartile gradient honestly, do not build a figure's caption on it.

**Housekeeping (1 hour, fold into step 1):** the two audit code guards
(`two_hop_map` empty-list fix in experiment_nonlinear.py:28; per-month count assertion in
run_phase5_stats.py), the stale MLP-averaging comment (run_phase5_nonlinear.py:39), and a
RUN_LOG addendum noting the 9/7 direction split behind "16 pass FDR."
