# Phase 7 Analysis — Nonlinear Architectures on the Kalman Backbone (2026-08-13)

Interpretation pass over the three Phase 7 experiments (resMLP two-stage, shared-encoder LSTM,
1-layer GAT), all on the global Kalman backbone: 234 basins, 5 folds, horizons 1-3,
deseasonalized standardized target, heads learning residual target−kalman, n = 19,656 rows per
arm per horizon, 20 degree-matched random-graph placebos per neighbor family. All numbers below
are from `phase7_{resmlp,lstm,gnn}_{summary,headline}.csv` or computed fresh from the
predictions/placebo CSVs (scripts in the session scratchpad; per-fold tables saved as
`p7_{resmlp,lstm,gnn}_folds.csv`, per-basin FDR details as `p7_fdr_*.csv` in scratchpad).

Verification performed here: zero NaN predictions in all three files; every arm has exactly
19,656 rows per horizon (234 basins × 84 months; folds span 17/17/17/17/16 months,
2019-06 to 2026-05); the five shared reference arms (`kalman_ar1`, `ridge_own`,
`ridge_corr_top1`, `ridge_own_era5`, `ridge_corr_top1_era5`) are **bit-identical across all
three prediction files** (md5 on sorted predictions), so every cross-experiment and
cross-phase comparison in this document is exact.

Audit caveats carried throughout: (a) placebo draws are shared across arm families within an
experiment (family-level placebo wins are not independent); (b) LSTM/GNN train on ~85% of train
months (early-stop holdout, never refit) — a conservative bias vs ridge; (c) resMLP's MLP stage
uses sklearn random-10% early stopping (matches Phase 5/6 MLPs), not the torch models'
time-ordered split; (d) the GNN `_era5` twin comparison conflates architecture with a
neighbor-ERA5 information widening; (e) seed spread is reported before any win is claimed.

---

## (i) Experiment-by-experiment breakdown

### 1. resMLP two-stage (ridge on own state + MLP correcting the residual from neighbor-only features) — the surprise of the phase

**Pooled skill (%) vs the ridge twin on identical information (`ridge_corr_top1`), by seed:**

| comparison | h | s0 | s1 | s2 | DM p (worst seed) | placebo (all seeds) |
|---|---|---|---|---|---|---|
| resmlp_corr_top1 vs ridge_corr_top1 | 1 | +0.50 | +0.80 | +0.73 | 1.5e-2 | 20/20 beaten, p_rank .048 |
| resmlp_corr_top1 vs ridge_corr_top1 | 2 | +1.45 | +1.78 | +1.71 | 4.8e-11 | 20/20 beaten, p_rank .048 |
| resmlp_corr_top1 vs ridge_corr_top1 | 3 | +1.74 | +1.65 | +1.53 | 1.4e-9 | 20/20 beaten, p_rank .048 |

Versus `ridge_own` the same arms score +0.99/+1.29/+1.21 (h1), +1.61/+1.94/+1.87 (h2),
+1.85/+1.76/+1.64 (h3). Seed spread is small relative to the effect (≤0.3 pp against effects of
0.7-1.9 pp) — this is NOT the MLP-seed-noise regime of Phases 5/6.

**Fold stability: complete.** Per-fold skill vs the twin is positive in **5/5 folds at every
horizon for every seed** (15/15 fold-horizon cells per seed; weakest cell +0.07, f4 h1 s0). The
win is not driven by any fold.

**Placebo control: at the floor everywhere.** Every no-ERA5 resMLP arm (top1 and top2, all
three seeds, all three horizons) beats all 20 random-neighbor placebos (p_rank = 1/21). And the
placebo family itself is informative: random-neighbor resMLP pools to RMSE 1.0456/1.1881/1.2481
(h1/h2/h3 family means) — i.e. **at the `ridge_own` level (1.0452/1.1877/1.2477)**. The
two-stage architecture with a junk neighbor adds nothing; all of the gain over `ridge_own`
requires the real neighbor.

**Per-basin FDR (q=.10), resmlp_corr_top1 vs ridge twin (win/lose):** h1 9/4, 4/2, 14/6
(s0/s1/s2); h2 **16/8, 25/8, 28/9**; h3 10/6, 10/2, 10/3. Winners dominate at every horizon and
seed; per-basin DM geography is seed-stable (s0 vs s1 at h2: Spearman rho = 0.92).

**The mirror geography — key mechanism finding.** The per-basin resMLP-vs-twin DM statistic is
strongly ANTI-correlated with the Phase 5 linear headline DM (corr_top1 vs own_ridge, h1):
**rho = −0.90 at h1, −0.73 at h2** (n = 234, both p < 1e-40). Concretely: 3 of Phase 5's 7
significantly-hurt basins (R_Parana_River, C_Aleutians_South_Alaska_Coast,
C_South_Gulf_of_Oman) are resMLP h2 FDR **winners**, while 4 of the 9 linear winners
(E_Tarim_He_Lop_Nur, R_Niger_River, C_East_Brazil_South_Atlantic_Coast,
C_Southwest_Mediterranean_Coast) are resMLP h2 FDR **losers** vs the twin. The two-stage MLP
gains where the linear neighbor coefficient was harmful and cedes ground where the linear
coefficient was already right. But note this cannot be pure damage-gating: gating a harmful
neighbor can only recover the `ridge_own` level (skill 0), whereas the real arms sit +1.6 to
+1.9% ABOVE `ridge_own` at h2 with placebos at the floor — the MLP is extracting genuine
neighbor information at h2-h3, concentrated in exactly the basins where the linear map failed.

**ERA5 tier.** `resmlp_own_era5` (ERA5-only control, no neighbor) beats `ridge_own_era5` at all
horizons in all seeds: +1.75/+1.80/+1.52 (h1), +1.86/+1.49/+1.58 (h2), +1.09/+1.35/+1.61 (h3),
all DM p ≤ .009 — the residual-MLP architecture also improves the linear ERA5 map, with no
neighbor involved. The neighbor arms with ERA5 aboard add nothing beyond that control
(vs `resmlp_own_era5`: −0.8 to +0.1 at h1, +0.2 to +0.8 at h2, −0.4 to +0.2 at h3, all ns or
seed-sign-flipping) and their placebo ranks collapse from the .048 floor to .19-.90. F5 is
negative at h1 for `resmlp_own_era5` in all three seeds (−0.7/−1.2/−1.1).

### 2. Shared-encoder LSTM — the ERA5 integration win, seed-halved but real

**No-ERA5 tier: consistent with the locked story.** `lstm_own` vs `ridge_own`: +0.07 ns /
+0.59 (p=.002) at h1 (s0/s1), +0.82/+0.91 both p<2e-4 at h2, −0.62/−0.10 at h3.
`lstm_corr_top1` vs `ridge_corr_top1`: ns at h1-h2, significantly WORSE at h3 (−0.81 p=.03,
−1.11 p=.008). Placebos: real neighbor beats 20/20 at h1 (both seeds) but collapses to
6/20-0/20 at h2-h3. A curious both-seed h2 bump aside (see anomalies), sequence modeling of the
filtered state alone adds nothing durable — the Kalman filter has already extracted the
temporal structure.

**ERA5 tier: the headline.** `lstm_own_era5` vs `ridge_own_era5` (identical information — own
state + 11 ERA5 anomaly channels):

| h | s0 | s1 | 2-seed ensemble | DM p (s0 / s1) | folds positive (s0 / s1) |
|---|---|---|---|---|---|
| 1 | +2.05 | +1.13 | +1.95 | 1.5e-7 / 8.2e-3 | 5/5 / 4/5 |
| 2 | +2.24 | +1.19 | +1.99 | 8.4e-8 / 3.8e-3 | 5/5 / 4/5 |
| 3 | +1.60 | +0.94 | +1.43 | 2.2e-7 / 5.3e-3 | 5/5 / 4/5 |

The prompt's quoted +2.05/+2.24/+1.60 is the BEST seed; s1 is roughly half the size. But the
win is sign-stable, significant in both seeds at all horizons, fold-stable (s1's only negative
fold is f4, mildly, at all horizons; gains concentrate in f2/f3), and the 2-seed ensemble holds
+1.4 to +2.0%. Per-basin FDR: s0 gives 12/0, 20/1, 14/2 (win/lose, h1/h2/h3) but s1 gives
**0/0** at h1 (12/4, 11/2 at h2/h3) — per-basin significance at h1 is seed-fragile even though
the pooled effect is not; per-basin DM geography correlates across seeds at rho = 0.70 (h1) /
0.82 (h2). With caveat (b) (LSTM sees only ~85% of train months), these numbers are if
anything conservative.

**Where the LSTM-ERA5 gain lives:** its h1 FDR winners (Saharan Atlas, Parnaiba, Dniester,
East Brazil, East Australia, Japan, Niger, Svalbard, Sumatra...) overlap the Phase 6 linear
ERA5 FDR winner set (27 basins) in only **1/12** — the LSTM is not amplifying ridge's ERA5
gains, it is finding skill in different basins (including Niger and East Brazil, neighbor-story
basins where linear ERA5 did little).

**Neighbor channel on top of ERA5: redundant and seed-noisy.** `lstm_corr_top1_era5` vs
`lstm_own_era5`: s0 +0.04/−0.78/−0.73, s1 +0.85/+0.68/+0.10 — the sign flips across seeds at
every horizon; the within-architecture neighbor question is below the seed-noise floor.
Placebo ranks: still near the floor at h1 (18/20, 17/20 beaten) but collapsed at h2
(0/20 for s0 — see anomalies — 7/20 for s1) and h3 (1/20, 4/20).

### 3. GAT — clean negative, question closed

**The GNN never significantly beats its ridge twin in any of the 36 arm-horizon cells**
(2 graphs × 2 tiers × 3 seeds × 3 horizons). Best cell: `gnn_corr_top1_era5_s2` h2 +0.52
(p=.11 ns). Breakdown:

- **No-ERA5:** all seeds/graphs pooled-negative vs twin at every horizon (top1: −0.24 to −1.27),
  significantly worse at h1 for all three seeds (p ≤ 4.2e-4). Per-fold: 0-2 folds positive.
  Yet the real graph still beats 20/20 random graphs in most no-ERA5 cells — the neighbor
  signal is present in the GNN's inputs; the architecture wastes it relative to plain ridge.
- **ERA5 (with the neighbor-ERA5 message widening — MORE information than the twin):** s0/s1
  significantly worse than the twin at h1 (DM +3.4/+3.6); only s2 is competitive (ns both
  directions). Losing while holding an information advantage is the strongest possible closure
  (caveat (d) works against the twin here, and the GNN still loses).
- **Seed spread is the largest of the phase:** era5-h1 skill vs `ridge_own` spans 3.9% to 5.4%
  (top1) and 2.8% to 5.5% (top2) across seeds — 1.5-2.7 pp, larger than most effects of
  interest, consistent with the memory note that seed spread can exceed effect sizes.
- **top2 ≤ top1 everywhere** it is testable: ns to significantly worse (h2 s0 −0.92 p=9e-4;
  era5 h1 s0 −1.22 p=.027, era5 h3 s0 −0.80 p=.006). The one-hop result stands in the message-
  passing family too.

---

## (ii) Cross-architecture synthesis — what Phase 7 adds to the locked story

The locked story was: "the spatial effect is real/small/linear/one-hop/one-month; ridge is
enough; ERA5 is the largest skill source." Phase 7 forces two amendments and two confirmations.

**1. The resMLP two-stage win is robust — and it breaks the "linear, one-month" clause.**
Robust by every registered criterion: 3 seeds (spread ≤0.3 pp), 5/5 folds at all horizons,
20/20 placebos at all horizons/seeds, DM p down to 1e-11, per-basin FDR winner-dominated with
seed-stable geography (rho=.92). Its content: at h1 a nonlinear neighbor→residual map roughly
doubles the linear neighbor effect (+1.0 to +1.3% vs `ridge_own` against +0.49% linear); at
h2-h3, where the linear neighbor effect was dead (+0.16%/+0.11%), the nonlinear map extracts
+1.6 to +1.9% that requires the real neighbor (placebo floor, placebo family = `ridge_own`
level). The mirror geography (rho = −0.90/−0.73 vs the Phase 5 linear DM) says why Phase 5
missed it: the h2-h3 neighbor signal lives disproportionately in basins where the linear
coefficient was harmful, so a single linear term averaged it away. "One-month" survives only
as a statement about the LINEAR effect. Remaining holes before paper-grade: caveat (c)
(sklearn random-10% early stopping — the one design element not shared with the ridge twin;
note the same design LOST in Phases 5/6, so it is an unlikely savior, but a one-seed
time-ordered-early-stop rerun kills it cheaply) and the unknown death horizon (h4-h6 not run).

**2. The LSTM-ERA5 win is the first genuine sequence-architecture gain, and it sharpens — not
overturns — the Li 2026 framing.** On identical information, both seeds, all horizons,
fold-stable, ensemble +1.95/+1.99/+1.43%. Report it as the ensemble or seed range, never the
s0 numbers alone (the effect halves from s0 to s1). Its horizon profile is the interesting
part: linear ERA5 skill vs `ridge_own` dies by h3 (+5.2/+1.8/+0.65 ns), while LSTM-ERA5 holds
+7.1/+4.0/+2.2 (s0; ensemble similar shape). The "short-lead = filtering, long-lead = exogenous
forcing" framing gains a clause: **at leads beyond ~1 month, the forcing channel is usable only
through nonlinear temporal integration of the forcing history** — a ridge on lag features
cannot carry ERA5 skill to h3, a 12-month LSTM window can. This is directly relevant to Li's
h4+ edge (their NN ingests met forcing sequences) and gives our paper a mechanism for it,
while the Kalman-bar point (their short-lead skill is mostly free filtering) is untouched.
Also new: the LSTM's gains sit in different basins from ridge-ERA5's (1/12 overlap), so
nonlinearity widens ERA5's geographic reach rather than deepening it where it already worked.

**3. The GNN failure closes the graph-architecture question.** With identical node information
the GAT loses to feature concatenation + ridge at every seed/horizon; with strictly MORE
information (neighbor ERA5 via messages) it still never wins; top2 ≤ top1; seed spread is
huge. Combined with Phase 5 (fusion, coupled filter, 2-hop, pred-lag all ≤ ridge bolt-on),
message passing at n=234 basins with ~65 train months per fold is now a measured dead end,
softened only by caveat (b). One table row and two sentences in the paper; no more runs.

**4. Neighbor-channel redundancy under ERA5 is now a three-architecture result — the neighbor
headline needs reframing as mechanism, not as deployable skill.** In every architecture that
carries both channels, the neighbor's placebo ranks collapse at h2-h3 once ERA5 is aboard
(resMLP .19-.90; LSTM 0-7/20 beaten; GNN era5 mixed), and within-architecture neighbor-vs-own
deltas are ns or seed-sign-flipping. At h1 a residue survives (LSTM era5 arms still beat
17-18/20 placebos; ridge_corr_top1_era5 vs ridge_own_era5 was +0.48%, p=.0096 in Phase 6).
Reading: the neighbor is a PROXY for shared regional weather forcing; hand the model the
forcing directly and the proxy's information is mostly contained — at one month partially, at
two-three months (including the new nonlinear channel, which the resMLP era5 arms show being
absorbed) almost entirely. The paper's neighbor claim should therefore be stated as: a
controlled, placebo-verified measurement of how much regional signal TWSA fields themselves
carry (+0.5% linear h1, ~+1.6-1.9% nonlinear h2-h3), together with the finding that ERA5
subsumes most of it — which is itself the cleanest evidence for the shared-weather mechanism,
stronger than Phase 4's index conditioning. Phase 6's geographic complementarity (Africa)
survives as the where-you-lack-forcing-data story.

---

## (iii) Sanity checks and anomalies

1. **No data problems.** Zero NaNs; exact row counts everywhere; reference arms bit-identical
   across the three files (and, per the run audits, to Phase 6/Phase 5) — cross-phase deltas
   are real model differences, not pipeline drift.
2. **A placebo family beats a real arm outright:** `lstm_corr_top1_era5_s0` at h2 is beaten by
   ALL 20 random-neighbor placebos (p_rank = 1.0; real 1.1681 vs family mean 1.1644, best
   1.1605); `lstm_corr_top1_s1` h2/h3 and `gnn_corr_top2_era5_s0` h1 also hit 0/20. Once the
   shared signal arrives via ERA5 (or the encoder), the real top-correlated neighbor imports
   correlated noise that a random neighbor does not — the same mechanism as Phase 5's
   hurt-basin story, now visible at the pooled level. Caveat (a) applies (shared draws), but
   the direction is consistent across families.
3. **Non-monotonic horizon pattern, LSTM no-ERA5:** `lstm_own` beats `ridge_own` at h2 only
   (+0.82/+0.91, both seeds p<2e-4, 10/10 folds) with h1 flat and h3 negative. Real but odd;
   plausibly the 12-month window captures a slow component the scalar filtered state misses,
   which matters at h2 and drowns by h3. Unexplained; flag, do not build on it.
4. **GNN seed pathologies:** `gnn_corr_top1_s1` h1 beats only 8/20 placebos where its sibling
   seeds beat 20/20; era5-tier seed spread 1.5-2.7 pp. Any GNN statement must be seed-averaged.
5. **Fold notes:** no single fold drives any headline (see per-fold tables). Echoes of Phase
   6's f3 ERA5-precip issue appear in isolated era5 cells (gnn_s0 f3 h1 −4.2;
   resmlp_corr_top1_era5_s2 f3 h3 −2.6) and resMLP-era5's f5 h1 is negative in all seeds —
   consistent with the known La Niña-window fragility of ERA5 features, not a new problem.
6. **E_Tarim_He_Lop_Nur** — Phase 5's strongest linear-neighbor winner — is an FDR LOSER
   against every nonlinear treatment tested here (resMLP h2/h3, LSTM-era5 h2). Where the
   linear map is already right, nonlinear heads subtract. Good cautionary sentence for the
   discussion, same family as the Yenisey story in reverse.

---

## (iv) Ranked next steps for the paper

**1. Extend `resmlp_corr_top1` (no-ERA5) and `lstm_own_era5` to h4-h6 (one seed each
suffices).** The two new claims both live at the horizon frontier: the paper currently says
the effect dies after h1, resMLP says a nonlinear channel is alive and GROWING through h3, and
LSTM-ERA5 at h4+ meets Li's edge exactly where their model wins. This is the single highest-
information run remaining, and it feeds the phase's centerpiece figure: skill-vs-horizon
ladder (kalman / ridge / +neighbor / +ERA5 / resMLP / LSTM).

**2. resMLP time-ordered early-stopping control (one seed, corr_top1, h1-h3).** Kills caveat
(c), the only design asymmetry left under the headline nonlinear-neighbor claim. Cheap (hours).
If it holds, the claim is paper-grade: 3 seeds × 5/5 folds × 20/20 placebos × DM 1e-11.

**3. A third LSTM seed (s2) for `own_era5` and headline-by-ensemble.** With a 2x spread
between two seeds, n=2 is thin for a headline number; s2 decides whether ~+2% (s0-like) or
~+1.5% (ensemble) is the reportable effect. Report "range across seeds, ensemble headline."

**4. Figures/tables now determined:** (a) Phase 7 architecture ladder table (arm × horizon,
skill vs ridge_own, DM vs twin, placebo rank — one row per architecture family, seed range in
parentheses); (b) the mirror-geography scatter — per-basin resMLP-vs-twin DM against Phase 5
linear DM, rho = −0.90 annotated, Parana/Tarim called out — this is a new mechanism figure of
the same rank as the Phase 5 concordance scatter (and its complement: concordance said all
LINEAR architectures agree where the neighbor helps; this says the NONLINEAR channel lives
precisely where they fail); (c) placebo-rank collapse panel (neighbor placebo ranks with vs
without ERA5, all three architectures) — the redundancy result as one picture; (d) LSTM-ERA5
horizon profile vs ridge-ERA5 (the "forcing needs nonlinear integration" figure, pending
step 1's h4-h6 points).

**5. Prose changes to the locked arc:** the summary sentence becomes "linear, one-hop,
one-month at the linear level; a nonlinear residual channel persists to at least three months,
is architecture-cheap (a two-stage MLP), lives where the linear map fails, and — like the
linear effect — is largely absorbed once ERA5 forcing is supplied." The Kalman-bar and
controls contributions are untouched; "ridge is enough" must be retired or scoped to the
no-ERA5 linear tier.

**Killed:** further GNN work; top2 graphs in any architecture; within-LSTM neighbor arms
(seed-sign-flipping, redundant under ERA5); treating s0 LSTM numbers as the headline.
