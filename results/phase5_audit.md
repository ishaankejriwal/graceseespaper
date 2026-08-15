# Phase 5 Audit — 2026-08-12 batch

Adversarial audit of the four Phase 5 experiments (predlag, fusion, coupled, nonlinear), the
consolidated stats script, and RUN_LOG.md. Every numeric claim was recomputed from the raw
`*_predictions.csv` / `*_placebo_monthly.csv` files with independent pandas/scipy code
(own implementations of skill, Harvey-corrected DM, placebo rank, block bootstrap, BH-FDR —
no gracefc imports). The coupled-filter verification script was rerun and passes.

Verdict up front: **the numbers are real and the code is clean — no leakage, no unfair
comparison, no join misalignment found. Three claims in RUN_LOG.md overstate what the CSVs
show; one of them (the coupled "placebo edge persists at ALL horizons, unlike the ridge
treatment which dies after h1") is contradicted by the project's own phase3b summary and must
not reach the paper as written.**

---

## VERIFIED

All recomputed digit-for-digit from raw prediction/placebo rows; "claim" numbers as given in
RUN_LOG.md / the task brief.

**predlag**
- h1 skill vs kalman_own_ridge: mine **−1.1926%** (claim −1.19%). DM p=.00163.
- Beaten by placebos at every horizon: mine **0/50 beaten, p_rank=1.000 at h1–h6** (claim 50/50 at h1–6). Confirmed.

**fusion**
- h1 skill vs kalman_ar1: mine **−1.8259%**, DM p=**3.47e-5** (claim −1.83%, p=3.5e-5).
- MLE variant h1: mine **−5.5978%** (claim −5.60%), DM p=1.9e-13.
- 0/50 placebos beaten at h1 (and h2, h5, h6). Confirmed at h1; see Discrepancy 2 for h3/h4.

**coupled**
- h1 skill vs kalman_ar1: mine **+0.2354%** (claim +0.235%), DM p=.562 (claim never DM-significant — confirmed: p = .56/.90/.38/.35/.55/.077 at h1–h6, none <.05).
- Placebo ranks: **50/50 beaten at h1, h3, h4, h6 (p_rank .0196); 45/50 at h2, h5 (p_rank .1176)** — matches the RUN_LOG entry ("50/50 at h1/h3/h4/h6, 45/50 at h2/h5") and the task claim of 45–50/50 at all six horizons.
- verify_coupled.py rerun: hand-coded 2×2 filter == numpy matrix reference on 20 random NaN-gapped trials (states atol 1e-9, loglik 1e-7); c=0 reduces exactly to the scalar own filter (1e-10); synthetic shared-shock recovery c_hat=**0.720** vs true ~0.69 (RUN_LOG: "0.72 fit vs 0.69 true"). All pass.

**nonlinear**
- GBM and MLP lose to their ridge twins at h1–h3, every arm, always DM-significant (largest p = 2.75e-4; see Discrepancy 1 on the "p<1e-4" wording). GBM: p 2e-8 … 4.3e-5. MLP corr_top1 seeds: p 3.5e-7 … 3.5e-5 at h1.
- ridge_corr_top1_2hop vs ridge_corr_top1: **−0.035% / −0.080% / −0.089%** at h1/h2/h3 — "slightly worse everywhere" confirmed.
- At h1, every real-graph nonlinear arm beats **20/20** placebo twins (gbm_corr_top1, gbm_corr_top1_2hop, mlp_corr_top1_s0/s1/s2; p_rank .0476 each). Confirmed. (Decays at h2–h3: e.g. mlp_corr_top1_s1 9/20 at h2 — the log makes no claim there.)
- ridge_corr_top1 is the best model at h1, h2 and h3 in the nonlinear table. Confirmed.

**stats table / FDR**
- kalman_corr_top1 vs kalman_own_ridge h1 on matched rows: skill **+0.4879%**, my bootstrap CI **(+0.1383%, +0.9009%)** (claim +0.14/+0.90), DM p **.02177** (claim .0218). Exact.
- Per-basin FDR h1 (own BH implementation): **132/234 basins favor the treatment, 16 pass q=0.10**. Exact.
- RUN_LOG stats-entry side claims verified: coupled vs own_ridge h2 = +0.9249%, p=.00737 (claim +0.92%, p=.007); coupled vs kalman_ar1 h2 = +0.03% ns; corr_top1 vs own_ridge at h4/h5/h6 = −0.066%/−0.061%/+0.070%, p=.73/.68/.65 ("~0 ns") — all confirmed.

**Cross-run integrity (design question d, verified empirically)**
- Every (model, horizon) cell in every run has exactly **19,656 rows = 234 basins × 84 test months**; zero duplicate (name, issue_date, target_date, fold, horizon, model) keys in any predictions file; every target month has exactly 234 rows in every arm and every placebo file (count column ≡ 234).
- Cross-run merges are perfectly 1:1 (n_a = n_b = n_merged = 19,656) and targets are bitwise identical across runs (max |target_a − target_b| = 0).
- kalman_ar1 rows are bitwise identical across phase3b, predlag, fusion and coupled runs; nonlinear's ridge_corr_top1 predictions are bitwise identical to phase3b's kalman_corr_top1 (max |diff| = 0) — the backbone really is shared, so twin comparisons are apples-to-apples.
- File row counts decompose exactly as (#models × #horizons × 19,656) + header for all five prediction CSVs.

---

## DISCREPANCIES

1. **RUN_LOG nonlinear entry: "DM p < 1e-4" at every horizon — not true for every arm.**
   mlp_own_s0/s1/s2 at h1 have p = 1.12e-4 / 2.75e-4 / 2.45e-4. Everything is still comfortably
   significant and in the claimed direction; the blanket "p < 1e-4" should read "p ≤ 3e-4"
   (or quote per-arm maxima). LOW impact, but it is a checkable number in a log that feeds the paper.

2. **RUN_LOG fusion entry: "loses … to 50/50 random-neighbor fusions" — not at every horizon.**
   At h3 fusion beats 4/50 placebos (46/50 lose to it is false as "50/50"), at h4 it beats 1/50.
   Exactly 0/50 at h1, h2, h5, h6. The conclusion (fusion is dominated) stands; the "50/50" phrasing
   doesn't, unless scoped to h1.

3. **RUN_LOG coupled entry (MED): "the placebo edge persists at ALL horizons, unlike the ridge
   treatment which dies after h1" — contradicted by phase3b's own summary.**
   phase3b_summary.csv / my raw recompute: kalman_corr_top1 beats **50/50 placebos at h1, h2 AND h3**
   (p_rank .0196 each), 0/50 at h4, 1/50 at h5, 48/50 at h6 (p_rank .0588). The ridge treatment's
   placebo edge survives through h3 — it does not "die after h1". Meanwhile coupled's "edge" at
   h2/h5 is 45/50 (p_rank .118), which is not significant. The defensible statement is: both
   treatments beat placebos at h1–h3 (and marginally h6); coupled additionally holds 50/50 at h4.
   The entry also omits that vs the stronger own-only comparator (kalman_own_ridge) coupled is
   significantly WORSE at h4–h6 (−1.76%/−1.97%/−0.88%, p ≤ 1.4e-10), so "persists at all horizons"
   cannot be read as usable skill. This is the one claim most likely to leak into the write-up as
   a false contrast.

---

## BUGS

1. **LOW — latent crash. `src/gracefc/experiment_nonlinear.py:28` (`two_hop_map`):**
   `graph.get(nbr, [None])[0]` raises IndexError when a basin's neighbor list is an empty list
   (corr_topk returns `[]` when a basin has no positively correlated peer). Did not trigger in this
   batch (I verified 0 empty lists in all folds), but any future subsample/distance-restricted graph
   can hit it. Failure scenario: rerun on a basin subset where one basin's train-window correlations
   are all ≤ 0 → run crashes mid-fold. Fix: `(graph.get(nbr) or [None])[0]`.

2. **LOW — stale/incorrect comment. `scripts/run_phase5_nonlinear.py:39`:**
   "MLP seeds average into one row per arm" — the code does NOT average; it emits one summary row
   per seed (mlp_*_s0/s1/s2), each ranked separately against the placebo distribution. The behavior
   is fine (arguably better); the comment describes a different design and could mislead the write-up
   into reporting a seed-averaged number that doesn't exist in the CSV.

3. **LOW — capacity mismatch in one secondary comparison. `scripts/run_phase5_fusion.py:124`:**
   fusion_corr_top1_mle is ranked against the closed-form-fit placebo distribution (MLE placebos were
   never run; `fusion.py` docstring says this is intentional). The mismatch is anti-conservative in
   principle (the real arm got extra fitting the placebos didn't), but the MLE arm lost anyway
   (0/50, −5.6%), so no conclusion rests on it. Disclose if the MLE arm is ever reported as more
   than a sensitivity note.

No leakage bugs, no join bugs, no placebo-seed bugs found.

---

## DESIGN CONCERNS (a–h)

**a. Causality — clean.**
- fusion: `fit_fusion_obs` receives `y_i[train_mask], y_j[train_mask]` with `train_mask = index < fold.test_start` (fusion.py:99–100); coupled: `fit_coupling` on the same pre-test mask (coupled.py:100, 117). Nothing test-period enters any fit.
- Graphs: all three runners build `corr_topk` on pre-test rows only (run_phase5_fusion.py:79–80, run_phase5_coupled.py:72, experiment_nonlinear.py:88–89).
- Deseasonalization and per-basin (rho,q,r) are pre-test (cached; bitwise-identical kalman_ar1 rows across runs prove the cache is consistent with each run's residual construction).
- Filters run over the full series with train-frozen parameters, using y_i(t), y_j(t) at issue t — exactly the information set of the audited `filtered_state_wide` convention in phase3b (where the neighbor's filtered state at issue t also embeds y_j through t). Consistent.

**b. Placebo fairness — matched.**
- fusion placebos refit (c, r_j) closed-form per random neighbor — identical procedure and capacity to the primary arm (mle placebo gap noted under Bugs #3).
- coupled placebos rerun the identical 1-param bounded MLE per random pair.
- nonlinear placebos use the same head, same hyperparameters, same feature count; GBM placebo `random_state=0` equals the real arm's.
- MLP: no seed-averaging happens anywhere. Each real seed-arm (s0/s1/s2, seeds 0/1/2) is compared as a single-seed model against 20 single-seed placebos (seeds 1000+i) — single-seed vs single-seed, which is fair. The p_rank is computed **per-seed-arm** (not pooled), against the same shared 20-placebo distribution, so the three ranks are correlated and constitute 3 tests; at h1 all three sit at the 20/20 floor, so no cherry-picking is possible, but the write-up should report them as per-seed ranks.
- Seed bases are crc32 of the arm name mod 1e6: corr_top1=352208, fusion=86661, coupled=876756, nonlinear=242860, pred_lag1_top1=185052, pred_lag1_top2=800102 — all distinct, no shared placebo draws across experiments. Within an experiment, placebo seed s reuses the same random graph across folds (base excludes fold) — same convention as the already-audited phase3b (models a fixed random selection rule, applied to all folds); consistent, not a bug.

**c. 2-hop placebo — verified by code AND empirically; one interpretive caveat.**
- Code: `randomized_two_hop` maps None exactly where `hop2` is None. Empirically (folds f1/f2, seed base+500+0): padding identity holds for all 234 basins in both folds.
- Coverage: **112/234 (f1) and 116/234 (f2) basins are mutual top-1 pairs with NO 2-hop node (~48%)** — the 2-hop feature is structurally zero for about half the sample in both real and placebo arms. The 2-hop null therefore has materially reduced power and the write-up should say the test covers ~half the basins.
- Yes, the 2hop placebo arm's 1-hop feature reuses `g_rand` (base+seed) — the same random graph as the corr_top1 placebo of the same seed. Is that a problem? For each arm's own rank test, no (each placebo is internally consistent: both hops random). It does mean two things: (i) the gbm 1-hop and 2-hop placebo distributions are correlated across arms (irrelevant to either p_rank); (ii) more importantly, because the placebo randomizes BOTH hops, `gbm_corr_top1_2hop` beating its placebo tests the joint graph, not the 2-hop increment. The incremental 2-hop question is (correctly) answered by direct ridge_2hop vs ridge_top1 comparisons — negative. Do not cite the 2hop placebo rank as evidence of multi-hop signal.
- Cosmetic: placebo 2-hop can collide with the same basin's placebo 1-hop node (~1/234 basins per seed; the real arm can never collide) — negligible capacity asymmetry.

**d. emit_rows / row alignment — verified 1:1.**
Same test-window predicate as experiment_kalman (`target_date` in [test_start, test_end], both inclusive). `pivot_wide` reindexes to a gapless monthly axis, so positional `shift(-h)` equals calendar `DateOffset(months=h)` — the merged target and the stamped target_date always refer to the same month, including across the 2017–18 mission gap (gap months exist as NaN rows and drop via dropna). n_rows = 19,656 everywhere; zero duplicate keys; cross-run merges lossless (see Verified).

**e. matched_compare — no silent misalignment.**
Dates parse identically (ISO strings ↔ parse_dates); suffixes touch only target/pred; merges verified lossless with targets identical. The monthly-mean weighting cannot bias anything **in this batch** because every month has exactly 234 basins in every arm — I confirmed pooled-row skill equals monthly-mean skill to all reported digits (e.g. predlag h1 −1.1926% both ways). Caveat for the future: the summary CSVs use pooled-row skill while the headline table uses monthly-mean skill; they coincide only while per-month basin counts are constant. If a future run drops basins for some months, the two definitions will silently diverge — worth one assertion (`per-month count == n_basins`) in run_phase5_stats.py.

**f. Stationary prior p01 — correct.**
For x_t = F x_{t−1} + w, F = diag(rho_i, rho_j), Cov(w) = [[q_i, cq],[cq, q_j]], the discrete Lyapunov equation P = F P Fᵀ + Q gives elementwise: p01 = rho_i·rho_j·p01 + cq ⇒ **p01 = cq/(1 − rho_i·rho_j)**, exactly what coupled.py:27 implements (with the same 1e-6 floor style as the audited scalar filter). Note the matrix reference in verify_coupled.py uses the same closed form for its prior, so the 1e-9 match does not independently test the prior — but the derivation is a two-line identity and the joint/partial covariance updates were additionally hand-checked (Joseph-free (I−KH)P forms are algebraically exact here, and the update-order of the in-place p00/p01/p11 assignments is correct in all three branches).

**g. The "persists at ALL horizons, unlike ridge which dies after h1" claim — NOT supported.**
See Discrepancy 3. phase3b corr_top1 placebo ranks by horizon: .0196/.0196/.0196/1.0/.98/.0588. The ridge treatment's placebo edge survives h1–h3; coupled's is at the .0196 floor for h1/h3/h4/h6 and .118 at h2/h5. The honest contrast is "coupled additionally holds its placebo rank at h4, where ridge dies" — and even that coexists with coupled losing to kalman_own_ridge by 1.8–2.0% (p≤1e-8) at h4–6. Rewrite before any of this reaches the paper.

**h. Multiple comparisons — real but manageable; must be disclosed.**
Today added ~33 real-arm × horizon cells, an 88-row headline table, 3 FDR families, and 3 MLP seed-arms per comparison. The registered headline (phase 3b, single pre-committed comparison) is unaffected. The exposures are: (i) coupled's p_rank .0196 is the floor attainable with 50 placebos (1/51) and was hit at 4 of 6 heavily-correlated horizon tests — fine as a robustness observation, not as an independent discovery; (ii) coupled vs own_ridge h2 (+0.92%, p=.007) is 1 of ~66 unregistered headline-table cells and RUN_LOG already correctly disarms it via the comparator trap — keep that framing; (iii) the "nonlinear arms beat 20/20 placebos" involves 5 arms × 3 horizons, and only h1 is uniformly at the floor — quote it as h1-only, which the log does. Recommended disclosure sentence: all Phase 5 comparisons are exploratory; only the Phase 3b corr_top1-vs-own_ridge h1 comparison was registered in advance.

---

## RECOMMENDATIONS (ranked)

1. **Rewrite the coupled RUN_LOG sentence** (Discrepancy 3) before write-up: "both treatments beat 50/50 placebos at h1–h3; coupled additionally at h4; coupled is significantly worse than own_ridge at h4–6". This is the only finding that would materially mislead the paper.
2. Correct "DM p < 1e-4" → "p ≤ 3e-4" (Discrepancy 1) and scope "50/50 random-neighbor fusions" to h1 (or "≥46/50 at all horizons") (Discrepancy 2).
3. In the write-up of the 2-hop null, disclose (i) ~48% of basins have no 2-hop node (test covers half the sample) and (ii) the 2hop placebo randomizes both hops, so only the direct ridge_2hop-vs-ridge_top1 comparison speaks to the increment.
4. Add the exploratory/registered distinction (concern h) to the paper's stats section; report MLP placebo ranks as per-seed (and fix the stale averaging comment, run_phase5_nonlinear.py:39).
5. Guard `two_hop_map` against empty neighbor lists (experiment_nonlinear.py:28).
6. Add an assertion in run_phase5_stats.py that per-month basin counts are constant (keeps monthly-mean and pooled skill definitions equivalent if the sample ever changes).
7. Document that 20% of coupled c fits sit at the +0.95 bound (236/1170 at ≥0.94; those basins' train neighbor correlation averages 0.82 vs 0.70 for the rest — saturation reflects genuinely shared signal, but the bound and the coarse xatol=0.02 tolerance are modeling choices worth one sentence).
8. If fusion_corr_top1_mle is ever promoted beyond a sensitivity note, run MLE-polished placebos to restore capacity matching (currently moot — it lost).
