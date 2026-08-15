# Manuscript rewrite notes — `main.tex` against `paper/notes/REWRITE_LEDGER.md` (2026-08-15)

Companion to the corrected-pipeline rewrite of `paper/main.tex`. Three parts:
(a) claims whose direction or significance changed, (b) every `\todo{RERUN}` inserted and
what fills it, (c) ledger items not implemented and why.

> **Two mid-flight spec changes, both applied.** The ledger was amended twice while this
> rewrite was in progress (resolution-audit + phase-8 stratification entries, `RUN_LOG.md`
> 2026-08-15). (1) The stratification passage, contribution 3, the discussion, the conclusions,
> the controls table and the limitations had been written under the superseded
> "sub-resolution" framing and were revised to the footprint-sharing framing. (2) The word
> "uniform" was then banned for the stratification result and replaced with concentration
> language plus the size gradient. See the two dedicated sections at the end of part (a).

A `\newcommand{\todo}[1]{\textbf{[TODO: #1]}}` was added to the preamble so the file compiles
as-is. Delete the macro and all `\todo{}` calls before submission. Structural validation run
after the rewrite: environments balanced, brace depth 0, 0 undefined `\ref`, 0 missing
bibliography keys, 0 invented citation keys, all tabular row widths match their column specs,
the 6 `\includegraphics` lines remain commented out, and every file named in a `% source:`
comment exists on disk.

---

## (a) Claims whose direction or significance changed

### Reversals — a claim's truth value flipped

| Claim | Old | New | Where |
|---|---|---|---|
| Kalman beats per-basin ridge | "at **all six** leads, DM p ≤ 0.042" | **five of six**; lead 4 +1.04%, p = 0.078, not significant | abstract, contribution 1, Sect. 4.1, conclusions |
| Kalman vs **pooled** ridge | "significant at leads 1–3 and 6, nominal at 4–5" | significant at **h1–h2 only** (+3.86, +4.21%); h3–h6 not separable; h4/h5 **nominally negative** (−0.59, −0.64%) | Sect. 4.1 |
| Linear neighbor effect | +0.49%, p = 0.022 — **significant** | **+0.31%, p = 0.199 — NULL.** h2–h6 all negative, all ns | abstract, contribution 3, Sect. 4.4, Sect. 5.3, conclusions |
| Where the linear pocket lives | "concentrates in basins below the nominal GRACE resolution element"; resolved basins show nothing | **footprint sharing, not size.** High-sharing basins carry +1.16% *whether or not GRACE resolves them* (resolved×high n=57 +1.161%, p=0.0045; small×high n=21 +1.166%, p=0.0071); resolved low/mid-sharing basins (n=142) run **−0.500%** | abstract, contribution 3, Sect. 4.4, Table 2, Sect. 5.3, conclusions, Limitations |
| Linear pocket across leads | not previously stated | contamination-confined **and** short-horizon: high tercile +1.11% (p=2.3e-4) at h1, +0.59% (p=0.037) at h2, dead h3–h6; mid and low terciles negative at every lead | Sect. 4.4 |
| Is the stacked correction a leakage artifact? | untested | **No.** Positive in all 60 stratum×lead cells with no concentration in shared-footprint or small basins; +0.89 to +2.00% (all p ≤ 1.3e-7) in the resolved low/mid-sharing stratum where the linear effect is −0.50%. Per-seed worst cell +0.22%, still positive. Small basins gain *least* — a real gradient, and the reverse of leakage's prediction | abstract, contribution 3, Sect. 4.5 (new property 5), Sect. 5.3, conclusions |
| Naive resMLP neighbor concatenation | not stratified | negative overall (−0.67/−0.62/−0.55%), harm concentrated in **low/mid**-sharing basins (resolved×low/mid −0.89%, p=0.011) and flat in high-sharing — noise import except where footprints overlap | Sect. 4.5 |
| Neighbor placebo rank beyond lead 1 | not previously stated per-lead | real graph beaten by **0/50** placebos at h2–h6 (min300 arm 2/50 at h2) — the selected neighbor is *worse* than a random basin | Sect. 4.4 |
| ERA5 forcing gain at lead 2 | +1.81%, p = 0.0099 — significant | **+1.39%, p = 0.092 — lost significance.** ERA5 is now a lead-1-only result (+4.62%, p = 1.0e-4) | abstract, Sect. 4.2, Sect. 5.2, conclusions |
| GBM on ERA5 features at lead 3 | "statistically tied" | **significantly worse** (−0.71%, p = 0.037) | Sect. 4.2 |
| GRACE-only LSTM at lead 2 | "isolated exception: +0.8 to +0.9%, p < 2e-4" | +0.93%, p = 1.5e-4 (holds), but h3 now **significantly worse** (−1.03%, p = 0.008) | Sect. 4.2 |
| resMLP neighbor increment under ERA5 | "mixed, ≈ 0" | **significantly negative**: −0.67 / −0.62 / −0.55%, all p < 0.05 | Sect. 4.4 |
| Stack placebo integrity | "20/20 except s1 at h2 (16/20)" | **20/20 in all 12 lead×seed cells.** The 16/20 was a scoring artifact (seed-1 arm vs seed-0 placebos), fixed by per-seed placebo emission | Sect. 4.5, Table 4 |
| Neighbor as LSTM input channel | "dead or harmful" (single seed) | **"no robust ensemble gain"**: +0.36% (p = 0.090) at h1, negative at h2–h6. Phrase "dead or harmful" removed and the negation avoided too | contribution 3, Sect. 4.5, conclusions |
| Li h2 comparison | "statistically indistinguishable" | kept as **"not detectably different"**, now with the interval quoted (−16.5 to +7.6%) to make clear it is absence of evidence | Sect. 4.3 |

### Magnitude and framing changes — same direction, restated

| Claim | Old | New |
|---|---|---|
| Kalman vs damped persistence | +4.8/+8.1/+5.3/+3.2/+2.5/+3.5% | **+4.98/+8.79/+5.62/+3.07/+2.55/+3.63%**, DM p ≤ 8e-7. Explicitly framed as **stability, not growth** — a new "Robustness rather than growth" paragraph states both vectors side by side and says the margin did not grow |
| Ridge on filtered states vs per-basin ridge (h4–h6) | +3.1 to +4.4%, p ≤ 3.9e-6 | +3.4 to +5.4%, p ≤ 2.1e-6 |
| Persistence penalty | 21–24% | 16–23% |
| Climatology row | −122.3 … −28.6 | −125.6 … −33.6 |
| LSTM sequence gain | ens +1.95/+1.99/+1.43 | ens **+2.17/+2.36/+1.43**, p ≤ 4.6e-5 |
| resMLP vs ridge twin (GRACE-only) | s0 +0.50/+1.45/+1.74 | ens **+0.81/+2.15/+2.17**, p ≤ 6.7e-5, all seeds significant |
| Crossing (ours vs GRACE-FCast full) | +20.2/−3.8/−12.4/−16.5/−21.6/−26.7 | **+20.0/−3.5/−12.9/−17.4/−23.6/−30.3**; crossing point unchanged (between h2 and h3); framed as **robust to both fixes** |
| Matched sample | 227 basins × 61 months | **227 × 60 = 13,620 rows per lead** |
| Per-basin wins vs GRACE-FCast | 133/81/69/65/57/57 | **137/82/71/65/59/59** |
| Bare Kalman vs their full product at h1 | +14.4%, p = 0.0041 | **+14.63%, p = 0.0039** |
| vs their non-seasonal variant | wins h1–h2, loses h4–h6 by −10.3 to −23.2% | wins h1 **+30.9%**, h2 **+9.0%**; h3 a wash (−1.7% ns); loses h4–h6 by −8.9 to −23.4% — crossing against *that* variant moves to h3–h4 |
| Stage-2 neighbor correction | ~+1% flat, per-seed, h4–6 ensemble missing | **ensemble at all six leads: +0.91/+1.45/+1.33/+1.32/+1.42/+1.96%**, p ≤ 1.2e-10, CI lower bounds +0.64…+1.67. Grows rather than decays with lead |
| Stack vs strongest linear system with identical information | +2.40/+2.95/+2.62% (h1–3) | **+2.75/+3.86/+2.92/+2.52/+1.78/+2.23%** (h1–6), p ≤ 1.5e-9 |
| n per lead | 19,656 constant (pre-audit) | 19,422 / 19,188 / 18,954 / 18,720 / 18,486 / 18,252 |
| Per-basin test months | 84 | 83 at lead 1, fewer at longer leads |
| Fold issue windows | 17/17/17/17/16 | 17/17/17/17/**15** |
| Kalman boundary fits | "a substantial minority" (unquantified) | 1170/1170 converged, **259 at r < 1e-6, 31 basins at the boundary in all five folds** |
| GRACE-FCast seasonal/trend edge | ~0.02–0.08 skill units | **~0.07–0.09** at leads 4–6 |

### Structural rewrites executed (ledger §9)

1. **Contribution 3 is now two-sided.** Both the intro contribution and Sect. 4.4/4.5 state
   the linear null (with the footprint-sharing pocket as the primary sensitivity) and the
   nonlinear, placebo-validated, stratification-surviving correction as separate results. The
   section title changed from "Cross-basin information: isolation and validation" to
   "Cross-basin information at the linear tier: a null result with a footprint-sharing pocket",
   and the phrase "isolation and validation" is replaced throughout by "characterization with
   null-model controls". The full claim now reads: cross-basin information exists but is
   invisible to standard linear methods and to naive deep-model inputs; here is the delivery
   that finds it, here is the control proving it is the information not the architecture, and
   here is the stratification showing it is not the leakage artifact.
2. **A new paragraph "Two questions that must be kept apart"** separates *does
   correlation-selection beat a random neighbor* (yes at h1) from *does adding a neighbor beat
   not adding one* (no, p = 0.199).
3. **The stratification is now two displayed tables** — the area × footprint-sharing 2×2 in
   Sect. 4.4 and the 5-stratum × 6-lead stacked-correction table in Sect. 4.5 — with the bolded
   conclusion "the linear neighbor effect is confined to basins whose mascon footprint is shared
   with land outside the basin, regardless of basin size". (See the mid-flight revision note
   below; the first pass had this as a size claim.)
4. **Random-graph placebo promoted to primary control; `lstmres_own` demoted to diagnostic.**
   The four "evidential weight" properties were rewritten: property 1 is now the placebo
   (with placebo cost −0.27 to −0.05%, z = −6.8 to −18.6 sd, and the 1/21 rank-floor caveat),
   property 2 is the own-state diagnostic *explicitly labelled as not the primary control*,
   with the anti-correlation decomposition (corr(c,e) = −0.08 to −0.16; only 0.16–0.54 pp of
   the harm is variance cost, so 79–85% is wrong-direction correction), property 3 is the
   agreement between the two defensible references (+0.91…+1.96 vs stage-1, +0.92…+1.96 vs
   placebo median), property 4 is horizon stability.
   **The +2.05…+3.70% neighbor-vs-own-state contrast is quoted nowhere in the manuscript**,
   neither as skill nor as an architecture term.
5. **The zero-fill concern is closed in writing** (0 of 234 basins lack a top-1 neighbor;
   max|self-feature − own propagated state| = 0.000e+00).
6. **Ensemble reporting everywhere.** Table 4 now carries an ensemble column at all six leads
   (previously h4–6 were per-seed only, with the ensemble marked "not emitted"). Every headline
   in Sects. 4.2, 4.4 and 4.5 is a seed ensemble; per-seed values appear only as robustness.
   The Limitations seed-budget item now names the three specific seed instabilities
   (LSTM gain halving, resMLP-ERA5 s2 outlier, LSTM h2 neighbor sign flip).
7. **The stack stays labelled exploratory** — the post-selection paragraph and Limitations item
   are retained verbatim in force.
8. **New in the "Required disclosure" paragraph:** the largest architectural number in the
   study is neighbor-free (resmlp_own_era5 vs ridge_own_era5, +3.35/+3.58/+2.84%), stated so
   that no reader mistakes the leaderboard for the neighbor evidence.
9. **Placebo methods paragraph** now carries the p_rank floor caveat and the fold/horizon draw
   reuse; a dedicated Limitations item covers the dependence structure.
10. **The delivery/conditioning language was corrected** from "the effect survives
    conditioning" to "conditioning does not explain the effect away", per ledger §3.

### Mid-flight revision: what was rewritten a second time

The first pass wrote the stratification under the superseded framing, including the sentence
the amended ledger bans. Every affected passage was revised. Specifically:

**Removed everywhere (banned):** "There is no detectable linear cross-basin effect in basins
GRACE can resolve" (was set in bold in Sect. 4.4), plus its restatements in the abstract
("comes entirely from basins below the nominal GRACE resolution element"), contribution 3,
Sect. 5.3 ("nothing at all in the 199 basins GRACE can actually resolve") and the conclusions
("nothing whatever in the basins GRACE resolves"). A grep confirms no surviving instance; the
remaining occurrences of "basins GRACE resolves" now all appear in the *opposite* claim — that
the effect **is** present in resolved basins when their footprints are shared.

**Added:**
- Sect. 2.1, new paragraph **"Basin areas and native mascon geometry"**: areas are our own
  cos-latitude mask integration, the partition file and the CSR grid carry no area or tile
  geometry so there is nothing external to validate against (stated, not implied); native tiles
  recovered empirically by series-equality fingerprinting give 42,107 global tiles of median
  12,123 km² (IQR 11,518–13,032), so 90,000 km² is ~7 mascons and only 3 of 234 basins span
  fewer than two. The text explicitly declines to call the threshold "one resolution element".
- Sect. 3.4, new methods paragraph **"Footprint-sharing stratification"**: defines
  contamination = Σₜ (wₜ/W)(1 − fₜ), notes Spearman(area, contamination) = −0.325 so the two
  axes are near-independent, and states the union-of-284-masks lower-bound caveat up front.
- Sect. 4.4, the stratification paragraph rebuilt: the area split is presented first *and then
  demoted* as a proxy (with the 60k–200k sweep showing it is not a knife edge, +0.9 to +1.3%),
  followed by a new 2×2 table (area × sharing) and the contamination-tercile result
  (low −0.27 / mid −0.18 / high +1.16%, p = 0.00051), then the short-horizon extension. The
  subsection title changed from "…a null result with a sub-resolution exception" to
  "…a null result with a footprint-sharing pocket".
- Table 2 (controls): the resolution row was split into an **area** row and a new
  **footprint-sharing 2×2** row, the latter reading "not passed, and it is sharing not size".
- Sect. 4.5, new **fifth evidential property** with its own 5-stratum × 6-lead table, ending on
  "the leading artifact explanation for the correction is ruled out by stratification" and an
  explicit refusal to claim a teleconnection, plus the resMLP-concatenation diagnostic.
- Limitations: the sub-resolution item was rewritten as a footprint-sharing item, and a **new
  item** covers the contamination metric's lower-bound character, the unverifiable area
  provenance, and the fact that the stacked stratification reuses the headline's test months
  (a decomposition, not an independent replication).

**Mechanism discipline.** Per the amended ledger the manuscript stops at "artifact ruled out".
The phrase "genuine teleconnection" appears nowhere; the two places the word "teleconnection"
occurs in this context are both explicit refusals ("we make no claim that it is a hydroclimatic
teleconnection"; "calling it a teleconnection would be an overclaim"). The earlier draft's
"spatial denoising of a shared regional state" reading was also removed from the discussion,
since that is itself a mechanism claim.

Two ledger roundings were tightened against the CSVs during this pass: the sub-resolution
stacked row is `p ≤ 0.044` (max 0.04319, the ledger says 0.043), and the mid-sharing tercile is
`p ≤ 1.3e-4` (max 1.213e-4, not "< 1e-4").

### Second mid-flight revision: "uniform" retired, size gradient stated

The stacked-correction stratification was first written as "positive in all 60 cells, and the
pattern is flat rather than concentrated", with "uniform across sharing and size" in Sect. 4.5,
Sect. 5.3 and the conclusions. The amended ledger bans "uniform" for this result, because the
**size** axis carries a real, significant gradient. All three sites were rewritten, and the
gradient is now stated rather than smoothed over:

| Site | Was | Now |
|---|---|---|
| Sect. 4.5 lead-in | "the pattern is flat rather than concentrated" | "it does not concentrate in the strata a leakage artifact would favour" |
| Sect. 4.5 body | "the other is uniform across sharing and size at every lead" | new paragraph: *"Positive everywhere" is not the same as "the same everywhere"* — small basins gain **least**, size interaction lead 1 p = 0.026 / lead 6 p < 0.001; contamination interaction not distinguishable at any lead (h1 p = 0.099 trending toward *more* gain in high-sharing, h3 p = 1.0, h6 p = 0.66) |
| Sect. 4.5 resMLP diagnostic | "gains that are uniform across strata" | "gains that are positive across every stratum" |
| Sect. 5.3 | "uniform across footprint sharing and basin size" | "no statistically distinguishable difference across footprint-sharing terciles… along the size axis it does vary, in the direction that helps rather than hurts the reading" |
| Conclusions | "largest in the very basins where the linear effect is negative" (inaccurate — it is +2.00% there vs +2.03% in the high-sharing tercile at h6) | "is worth +0.89 to +2.00% in the very basins where the linear effect is negative, and is if anything weakest in the small, most footprint-shared basins that the artifact would favour" |
| Contribution 3 | "positive in all 60 strata-by-lead cells" | "…without concentrating in shared-footprint or small basins… and gains least in the small basins an artifact would favour" |

The framing throughout is that the size gradient runs **opposite** to what leakage predicts —
leakage would make the smallest, most footprint-shared basins gain most, and they gain least —
so it is claim-safe texture and is stated as such rather than hidden.

The abstract's "positive in all 60 strata-by-lead cells with no concentration in shared-footprint
basins" was confirmed accurate by the coordinator and is unchanged.

The two remaining instances of the word "uniform" in `main.tex` are unrelated to this result:
"drawn uniformly at random" (placebo construction, Sect. 3.4) and "a spatially uniform null"
(the geography of the *linear* neighbor result, Sect. 4.4).

**Two new `\todo{INTERACTIONS-CSV}` markers** hold the formal interaction p-values pending
`results/phase8_stratification_interactions.csv`. The numbers written into them are the ones the
coordinator supplied (size: h1 p = 0.026, h6 p < 0.001; contamination: h1 p = 0.099, h3 p = 1.0,
h6 p = 0.66); **confirm them against the CSV when it lands and delete the markers.** They are
tagged distinctly from the 16 `\todo{RERUN}` markers so the two batches can be closed
independently. The surrounding prose reads correctly with the p-values absent — each sentence
states the direction and significance verdict in words before the marker.

---

## (b) Every `\todo{RERUN}` inserted, and what will fill it

18 `\todo{}` calls in total, each paired with a `% TODO-RERUN:` comment naming the pending
work: 16 `\todo{RERUN}` (below) and 2 `\todo{INTERACTIONS-CSV}` (see the second mid-flight
revision section above).

| # | Location | What fills it |
|---|---|---|
| 1 | Sect. 3.4, jump-screen methods | Flagged-basin counts and the screened neighbor estimate from the corrected `results/phase4_jump_screen.csv`. **See note below — this one has since landed.** |
| 2 | Sect. 4.2, ERA5 paragraph | Variable-ablation attribution, the fold-level reanalysis-precipitation failure mode, the continental breakdown, and per-basin helped/hurt FDR counts (phase-6 basin analysis) |
| 3 | Sect. 4.3, physical units | Pooled and median-basin RMSE in cm at leads 1–3, ours vs theirs, on the corrected 227×60 matched sample |
| 4 | Sect. 4.3, splice | Row-pooled skill vs damped persistence for our system, theirs, and the splice variants; plus the lead-1 splice gain over their product (phase-6 hybrid) |
| 5 | Table 2 (controls), IAAFT row | Surrogate rank at leads 1–3 (phase-4 surrogates) |
| 6 | Table 2 (controls), index row | Index-conditioned estimate and placebo rank (phase-4 conditioning) |
| 7 | Table 2 (controls), jump-screen row | Flag count, screened estimate, placebo rank |
| 8 | Fig. 3 caption (neighbor map) | Counts of significantly helped and hurt basins under FDR control, and the identity of the extremes |
| 9 | Sect. 4.4, heterogeneity paragraph | Per-basin helped/hurt counts, continental pooling incl. Africa under ERA5 conditioning, random-forest covariate R², LOFO and oracle selection effects |
| 10 | Sect. 4.4, alternative architectures | Skill and placebo ranks for the predictive-lag graph, the fusion filter, and the coupled bivariate filter (phase-5) |
| 11 | Sect. 4.5, evidential property 3 | Fold-level decomposition of the correction, and the per-basin FDR map plus its rank correlation with the linear neighbor map |
| 12 | Sect. 4.5, delivery mechanism | The double-counting check: stage-2 correction on top of an encoder that already ingests the neighbor |
| 13 | Sect. 4.5, required disclosure | Head-to-head between the stacked system and the neighbor-free resMLP-ERA5 ensemble at lead 1 (was a tie pre-correction) |
| 14 | Sect. 5.2, crossover design map | Basin-level rank correlation between GRACE-FCast's lead-4 edge and the neighbor benefit |
| 15 | Sect. 5.3, neighbor in context | Africa pooled neighbor effect with and without ERA5 conditioning, and the per-basin ERA5-vs-neighbor rank correlation |
| 16 | Fig. 6 caption (complementarity) | All quantitative elements of the caption: the rank correlation, continental pooled values, and the extreme reanalysis failures |

**Note on the jump screen (#1, #7).** The ledger lists the jump screen among phases still
rerunning, so I left `\todo{RERUN}` markers per instruction. The rerun has since completed:
`results/phase4_jump_screen.csv` (written 2026-08-15 11:06, corrected pipeline) reads
`train_only_6sigma`: **2 flagged, +0.2073% vs own ridge, 50/50 placebos, p_rank 0.0196`; and
`full_series_6sigma`: **27 flagged, +0.0818%, 50/50 placebos, p_rank 0.0196**. I verified these
directly from the CSV but did not insert them, because the ledger is the specification and the
values have not been through the project's audit cadence. They are ready to drop into both
`\todo` sites once confirmed. Note the screened estimate (+0.21%) is now *below* the
unscreened one (+0.31%), the opposite of the pre-audit relationship — worth a look before the
sentence "survives the jump screen" is written.

**Reruns confirmed still outstanding at the time of writing:** `results/rerun_surr.log` is
empty (surrogates still running), and `results/rerun_basin.log` shows the phase-6 basin
analysis **crashed** with `FileNotFoundError: results/phase5_coupled_coupling.csv` — it depends
on the phase-5 coupled-filter output, which has not been regenerated. Phase 5 must be rerun
before the phase-6 basin analysis can succeed, which blocks todos 2, 8, 9, 14, 15 and 16.

---

## (c) Ledger items not implemented, and why

1. **The "conservative surrogate construction" disclosure was kept but its result removed.**
   The IAAFT methods paragraph still describes the design (99 surrogates, generated from the
   full record, conservative in that leakage favours the null). Only the outcome became a
   `\todo`. The `% source:` comment now points at the script rather than at the deleted
   `phase4_surrogate_summary.csv`.

2. **The winsorization magnitude was dropped rather than rerun.** The old text quoted the 2023
   Libya flood reaching `~5e4 σ` under train statistics. That figure comes from the pre-audit
   phase-6 ERA5 log, and it depends on fold-window training statistics, so it is not
   protocol-invariant. I replaced it with the qualitative statement ("several orders of
   magnitude beyond the training standard deviation") rather than adding a 17th `\todo`, since
   the guard's existence, not its magnitude, is what the methods section needs. Flag if you
   want it quantified.

3. **The "7% of selected edges are <300 km" statistic was deleted, not marked.** It is a graph
   property of the pre-audit correlation graphs and no corrected-pipeline equivalent exists in
   any results CSV. The ≥300 km row of Table 2 now stands on the skill and placebo numbers
   alone, which is the load-bearing part.

4. **`li_lstm_nonseas` versus raw persistence at lead 1 was removed rather than updated.** The
   old sentence read "their non-seasonal variant at lead 1 is worse than *raw* persistence
   (−27.4% vs −20.3% against damped persistence)". The corrected matched sample contains no
   `persistence` arm (only `damped_persistence_rho`), so the comparison cannot be reconstructed.
   I replaced it with the ledger's designated sharpest statement — their full product at
   −11.09% vs damped persistence at h1, carrying its p = 0.057 — which is stronger and
   verifiable. The nonseasonal variant's own value on that sample is −28.67% if you want it.

5. **The lead-1 ensemble p-value for `lstm_own_era5_ens` vs `ridge_own_era5` at h4–h6 is not
   quoted.** `phase8b_h16_ensemble_headline.csv` contains no such contrast family; the values
   (+0.808 / −0.044 / −0.113) come from `phase8_corrected_audit.md` sect. 3, which reports
   skill without p. The manuscript therefore quotes the seed-0 p-value at lead 4
   (p = 1.6e-4, from `phase8b_h16_headline.csv`) and states leads 5–6 as "indistinguishable"
   without a p. Adding that contrast family to `run_phase8b_merge` would close it.

6. **Ledger §10's limitation "20 placebo draws are reused across folds and horizons" was
   strengthened, not merely retained.** The audit established that draw *k* is the identical
   graph in every fold and at every horizon, so the manuscript now says the twelve cells are
   correlated re-scorings of 20 realizations. This is a stronger disclosure than the ledger
   asked for; it is what `phase8_corrected_audit.md` sect. 4c verified.

7. **Two pre-correction values are deliberately retained in the text**, both explicitly
   labelled as superseded and neither used as a result: the pre-fix ladder vector in the
   "Robustness rather than growth" paragraph (ledger §1 asks for exactly this contrast), and
   the pre-fix neighbor contrast (+0.49%, p = 0.022) in the sentence disclosing that the
   corrected pipeline reverses our own earlier finding. Everywhere else, no pre-audit value
   survives.

8. **One 2×2 cell does not fit the required sentence, and is flagged in the text rather than
   suppressed.** The amended ledger's mandated claim is that the linear effect is confined to
   high-sharing basins regardless of size, but the small × low/mid-sharing cell is also positive
   and nominally significant (n = 14, +0.913%, p = 0.036). I state the required sentence and
   then flag the cell explicitly: it is the smallest in the table, and it is exactly where the
   contamination metric is weakest, since counting only foreign *land* understates sharing for
   coastal and island basins — which is what most small, nominally clean basins are. This is
   consistent with the ledger's own §10 caveat, and the paragraph says "we flag it rather than
   explain it away" rather than asserting the connection as fact.

9. **Not touched, and still outstanding from before this rewrite:** the author list,
   affiliations, author contributions and acknowledgements are still `TODO` placeholders; the
   Zenodo DOI is unminted; the six figures are unwritten (specs in `paper/notes/FIGURE_PLAN.md`); and the
   unverifiable J. Hydrology Gauss–Markov/Kalman citation remains a commented-out note in
   Sect. 5.1 rather than a live `\citep`, since inventing a bibliography key was out of bounds.

---

## 2026-08-15 — Voice-and-readability pass (fresh-reader rules, ledger §8b)

Scope: language only. No claim, number, p-value, CI, citation, table value, caveat, or
`\todo{RERUN}` marker was altered; all `% source:` comments are untouched.

### Model-name table introduced

A naming table now sits at the end of Methods §"Learned models" (`tab:names`), and a new
reproducibility appendix (`app:repro`, Tables `tab:codes` and `tab:sources`) is the ONLY
place code identifiers appear. Paper name ↔ repository identifier:

| Paper name | Identifier(s) |
|---|---|
| Climatology | `climatology_zero` |
| Persistence | `persistence` |
| Damped persistence | `damped_persistence_rho`, `damped_persistence_reg` |
| Pooled ridge (+ climate-index variant) | `ridge_own_lags`, `ridge_own_plus_indices` |
| Per-basin ridge | `ridge_own_perbasin` |
| Kalman forecast | `kalman_ar1` |
| Ridge on filtered states | `kalman_own_ridge` |
| Neighbor ridge (registered contrast) | `kalman_corr_top1` |
| Neighbor ridge, distance-restricted | `kalman_corr_min{300,500,1000}_top1` |
| Neighbor ridge, alternative graphs | `kalman_geo_top{1,2,3}`, `kalman_corr_top{2,3}` |
| Own-basin ridge / ERA5 ridge | `ridge_own`, `ridge_own_era5` |
| Neighbor ridge (forcing experiments) | `ridge_corr_top1`, `ridge_corr_top1_era5` |
| GBM head | `gbm_own_era5`, `gbm_corr_top1_era5` |
| MLP head | `mlp_own_era5_s0` |
| Residual MLP | `resmlp_corr_top1`, `resmlp_corr_top1_era5`, `resmlp_own_era5` |
| LSTM (+ neighbor-channel variant) | `lstm_own`, `lstm_own_era5`, `lstm_corr_top1_era5` |
| Stacked system | `lstmres_corr_top1` |
| Own-state control | `lstmres_own` |
| GAT | `gnn_corr_top1`, `gnn_corr_top2_era5`, variants |
| GRACE-FCast (full / non-seasonal) | `li_lstm_full`, `li_lstm_nonseas` |

All identifiers were verified against the model columns of the archived results CSVs
(none invented). Terminology unified per Steinhardt (one term per concept): "own-state
ridge" → "ridge on filtered states"; "lstm+corr"/"stacked LSTM" → "the stacked system";
"lstmres_own"/"own-state diagnostic (arm)" → "the own-state control"; "resMLP" →
"residual MLP"; "linear level" → "linear tier"; "stage-one" (modifier) → "stage-1".

### Jargon count, before → after (rendered non-comment text)

- `phase[- ]?[0-9]`: 19 → 5, and all 5 survivors sit INSIDE `\todo{RERUN}` brackets,
  which the task's hard constraint required leaving byte-identical (they are deleted with
  the macro before submission). Outside todo markers and the appendix: 0.
- `\barms?\b`: 32 → 0.
- `backbone`: 9 → 0.
- Code identifiers in prose (`kalman`, `kalman+ridge`, `lstm+corr`, `lstmres_own`): 4 → 0.
- `RUN_LOG` / ledger / fold codenames in rendered text: 0 before, 0 after.
- Internal file references (`paper/notes/FIGURE_PLAN.md`, `results/*.csv`) in rendered captions and
  `\belowtable` notes: moved to `%` comments ("figure build note:" lines, which never
  render) or to the appendix source-file table; `\belowtable` notes now carry only
  reader-facing content plus a pointer to Appendix A.

### Numeral preservation

Diff-based multiset check (scratchpad `main_before.tex` vs final): the ONLY numeral
removed anywhere and not re-added is one "4" — the deleted workflow label "phase-4" in
the tab:controls `\belowtable` note. Every gained numeral lives in the new appendix
identifier/source tables, LaTeX column widths, build-note comments, or restates an
already-stated fact ("12-month windows" in tab:names). No value, p, CI, count, or
percentage changed.

### Compile

pdflatex → bibtex → pdflatex ×2 (MiKTeX): clean, 0 "undefined" in main.log,
33 pages (grew from the two new appendix tables and the Methods naming table).

### Sentences flagged (meaning-preservation judgment calls, for author review)

1. fig:controls caption: the build instruction "The panel should show the confidence
   intervals crossing zero" became the caption statement "the confidence intervals cross
   zero" — true given both p-values (0.115, 0.199) and the ledger's CI, but it is now an
   assertion rather than a spec.
2. "placebos were emitted per seed" → "each seed was scored against its own placebos" —
   same protocol, stated from the scoring side rather than the file-emission side.
3. "Giving all arms the 11 ERA5 variables leaves the estimate essentially unmoved" →
   "Giving every model in the comparison ... barely moves the estimate" — "barely moves"
   chosen over "leaves unmoved" because the point estimate does shift (+0.308 → +0.322).
4. Limitation 12: "phase-5 alternative-filter, and phase-6 basin-analysis arms" →
   "alternative-filter, and per-basin analyses" — the content description replaces the
   phase labels; the enumerated list of pending reruns is unchanged.
5. The two unnumbered `center` tabulars (2×2 cross and stacked-correction stratification)
   kept their structure; only surrounding prose was touched.
