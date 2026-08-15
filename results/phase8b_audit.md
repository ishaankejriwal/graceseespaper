# Phase 8b audit — h4-6 extension of the stacked LSTM + h1-6 Li & Kusche comparison

Auditor: adversarial audit subagent, 2026-08-13.
Scope: `scripts/run_phase8_lstm_combined.py` (new `--horizons` arg),
`scripts/run_phase8b_merge.py` (new), the h4-6 run outputs
(`results/phase8b_lstm_h46_*`), the merged tables (`results/phase8b_h16_*`), and the
matched Li comparison (`results/phase8b_li_comparison_*`), against the unchanged engine
`src/gracefc/experiment_lstm_combined.py` and the phase 6/7/8 plumbing. Everything below
was RECOMPUTED from the predictions/placebo CSVs (project venv, `stats.py` /
`models.rmse` / `per_basin_dm_fdr`), not read off the summaries. No project files were
modified except this audit file.

## Verdict: PASS WITH NOTES

Every headline number checks out digit-for-digit; the merge is provably lossless; the
matched Li sample was rebuilt independently from the source predictions and is
row-for-row identical to the shipped file. All findings below are wording-level or
already-disclosed caveats — no blocker, no leakage, no contamination of the h1-3
results.

## Findings

1. **PASS — Protocol identity between the h1-3 and h4-6 runs (severity: none).**
   The engine `src/gracefc/experiment_lstm_combined.py` was last modified 2026-08-13
   12:24, i.e. BEFORE both the phase 8 h1-3 full run (outputs 13:47) and the phase 8b
   h4-6 run (outputs 15:48) — both runs executed identical engine bytes. The only
   post-audit engine edit was the docstring caveat paragraph (lines 12-19); every code
   construct cited in `phase8_code_audit.md` is present shifted uniformly by +4 lines
   (crc base 44→48, stage-2 residual 120→124, placebo block 133-135→137-139), i.e.
   docstring-only change, behavior identical. The runner's new `--horizons` default
   "1-3" maps to `range(1, 4)` = the engine's own default, and the smoke path maps to
   `range(1, 2)` = the old smoke behavior, so the default path is semantically
   bit-identical to what produced the audited h1-3 results (no git; verified by mtime
   ordering + code comparison, and by finding 5's bit-identity of kalman_ar1 against
   the phase 6 pipeline). The h4-6 run wrote only `phase8b_lstm_h46_*` (h1-3 file
   mtimes untouched at 13:47): no contamination. Same 5 `DEFAULT_FOLDS`, seeds (0,1),
   20 placebos, same 4 ridge baselines confirmed in the outputs.

2. **PASS — h4-6 run structure is exactly as specified (severity: none).**
   `results/phase8b_lstm_h46_predictions.csv`: horizons exactly {4,5,6}, all 5 folds,
   15/15 fold-cells in the log, exactly the 13 real arms, 19,656 rows for every
   (model, horizon), frozenset-identical (name, issue_date, target_date) row sets
   across all 13 arms at each horizon, zero NaN in pred/target, zero duplicate keys,
   issue→target offsets consistent with each horizon (h4 120-123 d, h5 150-153 d,
   h6 181-184 d). Placebo file: 2 families × 20 draws, per-label monthly counts sum to
   19,656 at every horizon, no NaN. The shipped `_summary.csv` and `_headline.csv`
   reproduce exactly from the predictions (0 mismatches at 1e-6/1e-9 tolerance,
   including bootstrap CIs, which are deterministic at seed 0).

3. **PASS — Key h4-6 numbers verified digit-for-digit (severity: none).**
   Stage-2 increment (lstmres_corr_top1_sX vs lstm_own_era5_sX, same seed, pooled
   skill = 1 − MSE ratio): s0 +1.195/+1.066/+1.289% at h4/5/6, s1 +1.006/+0.984/+1.440%;
   DM p = 4.2e-6, 1.4e-10, 8.8e-9, 3.6e-5, 3.0e-8, 1.6e-10 — all < 1e-4 as claimed.
   Placebo ranks: lstmres_corr_top1 beats 20/20 at h4, h5, and h6 for BOTH seeds (raw
   ranks; real RMSE below the placebo minimum in every cell). lstm_own_era5_s0 vs
   ridge_own_era5: +0.781% (DM p=.0012) at h4, then −0.16% (p=.69) at h5 and +0.03%
   (p=.90) at h6 — the ERA5 sequence gain does die beyond h4 (s1 concurs:
   +0.59%/p=.014, then ns).

4. **PASS — The h1-3 "s1 placebo rank" caveat does NOT bite at h4-6
   (severity: none; good news worth recording).** At h1-3 the s0/s1 stage-1 gap made
   raw s1 ranks uninterpretable (6/12/20). At h4-6 the stage-1 gap
   (lstm_own_era5 s1 vs s0) is −0.19%/+0.10%/−0.15% — an order of magnitude smaller
   than the ~+1% correction — and BOTH the raw s1 ranks (20/20 everywhere) and the
   seed-matched increment comparison (s1 increment vs the 20 placebo increments over
   the s0 stage-1: 20/20 at h4, h5, h6; placebo increment max +0.05/+0.14/+0.04% vs
   real +1.0-1.4%) agree. Raw s1 ranks are quotable at h4-6; the seed-matched framing
   remains the safer default for h1-6 consistency.

5. **PASS — Merge is lossless and the matched Li sample is exactly right
   (severity: none).** (a) `phase8b_h16_{summary,headline}.csv` are pure concats of the
   phase 8 and phase 8b tables: identical columns, 39+39=78 / 21+21=42 rows, content
   identical after the script's sort — nothing lost or duplicated, h1-6 all present.
   (b) The matched sample was rebuilt from scratch from the two predictions files +
   `phase6_li_comparison_predictions.csv` following the stated rule (keys held by all
   13 models): the rebuild equals the shipped
   `phase8b_li_comparison_predictions.csv` row-for-row (outer-merge: 0 unmatched of
   1,080,066 rows; max |pred/target diff| 8.9e-16 = CSV round-trip ULP). Sample = 13
   models × 13,847 rows × 6 horizons; 13,847 = 227 basins × 61 months exactly (a
   complete rectangle, same months at every horizon), 2019-06..2024-11. (c) Targets:
   bit-identical (diff exactly 0.0) between our rows and li_lstm_full / li_lstm_nonseas;
   damped_persistence_rho differs by 1 ULP (8.9e-16) — a phase 6 file artifact,
   immaterial (the script's 1e-6 guard is against li_lstm_full). (d) Ensembles: recomputed
   ens = (s0+s1)/2 across ALL rows, max diff 1 ULP; the merge-on-(name,target_date,horizon)
   cannot duplicate (0 duplicate keys per model verified; fold test windows are disjoint)
   and the length assert would trip if it did. (e) Coverage merge caused no row
   loss/duplication (count is exactly 13×13,847×6). (f) Protocol replication:
   kalman_ar1 predictions in the matched sample are bit-identical (max diff 0.0) to the
   phase 6 file at every horizon including h4-6 — the strongest available proof that the
   h4-6 run reproduced the audited fold/deseasonalization/Kalman protocol exactly.

6. **PASS — Li comparison numbers verified digit-for-digit (severity: none).**
   all_matched, lstmres_corr_top1_ens vs li_lstm_full: +20.17% h1 (DM p=3.4e-4),
   −3.84% h2 (p=.450, ns), −12.45% h3 (p=.0182), −16.47% h4 (p=.0040), −21.57% h5
   (p=4e-4), −26.70% h6 (p=2e-6). Per-basin wins (dm_stat<0): 133/81/69/65/57/57 of
   227 — recomputed via `per_basin_dm_fdr`, max DM diff vs the shipped perbasin file
   2e-15. All 108 headline rows and all 156 summary rows (both subsets) reproduce
   exactly, `skill_vs_damped = 1 − (rmse/rmse_damped)^2` confirmed against
   independently recomputed RMSEs. The coverage≥0.5 subset (209 basins) tells the same
   story (+21.2% h1 → −28.4% h6), so the h4-6 verdict is not an island-basin artifact.
   Also verified: li_lstm_nonseas beats every arm of ours at h4-6, as the RUN_LOG
   states.

7. **NOTE (low) — RUN_LOG says the Li comparison "excludes most of f5"; it excludes
   ALL of f5.** f5's test window (2025-02..2026-05) lies entirely after Li's last
   month (2024-11): f5 contributes 0 matched rows. f4 loses its 2024-12/2025-01 months
   plus the 7 island basins; f1 loses gap-affected months. Suggest changing "most of
   f5" to "all of f5". The substantive point — matched-sample and full-sample skills
   are NOT interchangeable — is stated correctly in the RUN_LOG, and I found no place
   where a matched-sample number is presented as a full-sample number or vice versa
   (the RUN_LOG labels every block "full 234-basin sample" vs "matched sample").

8. **NOTE (low) — Known wrong-twin trap still present at h4-6, still correctly
   quarantined.** The h4-6 summary again auto-emits `dm_vs_ridge_twin` for the
   lstmres arms against no-ERA5 `ridge_corr_top1` (e.g. −4.71/p=1e-5 at h4). Grepped
   RUN_LOG and all analysis files: it is nowhere quoted as evidence, and the RUN_LOG
   phase 8b entry repeats the warning. Two additions: (a) at h4 the wrong twin
   actually UNDERSTATES the properly matched contrast (headline vs
   ridge_corr_top1_era5: DM −5.33), and at h5-6 the two ridges nearly coincide
   (ERA5 adds ~0 to ridge there), so the mismatch has little practical bite at h4-6 —
   but keep the blanket "never quote" rule for h1-6 consistency. (b) The RUN_LOG's
   "every arm sits within ~1.1% of ridge_own at h5-6" is exact for the learned arms
   (range −0.49%..+0.78% at h5, −0.57%..+1.12% at h6) but the kalman_ar1 BASELINE is
   −2.04% at h5; if "arm" is meant to include the backbone, reword to "no arm beats
   ridge_own by more than ~1.1%".

9. **NOTE (low) — Rounding edge in "all p≤.018".** The h3 Li contrast p is .0182,
   which the RUN_LOG's "all p≤.018" covers only after rounding. Quote "p≤.019" or the
   per-horizon values (.0182/.0040/.0004/2e-6) in the paper.

10. **NOTE (info) — nbrin s1 raw placebo ranks at h4-6 are confounded and mostly
    fail; not used in any claim.** lstmres_nbrin_corr_top1_s1 beats 20/20 at h4 but
    0/20 at h5 and 13/20 at h6 against placebos built on the s0 stage-1 net
    (lstm_corr_top1_era5_s0) — here the stage-1 seed gap DOES bite (unlike finding 4's
    lstmres family, whose stage-1 is lstm_own_era5). The RUN_LOG discloses the h5
    0/20; the nbrin family is dominated and appears in no headline. If nbrin ranks are
    ever discussed, use seed-matched increments only.

11. **NOTE (info) — Placebo graph draws at h4-6 are the identical 20 per-fold graphs
    as h1-3.** The crc32 base and per-fold draw seeds do not depend on `--horizons`,
    so the two runs (and, as before, the two stacked families) share graph draws. This
    is the same-protocol choice, consistent with how h1-3 shares draws across horizons
    within one run; it means the 6 horizons' placebo tests are not independent
    replicates of graph randomness — fine for the per-horizon rank claims made, just
    do not present "20/20 at six horizons" as 6 independent 1/21 events.

12. **NOTE (info) — Inherited phase 8 caveats carry over unchanged.** In-sample
    stage-2 residuals ("honest evaluation, plausibly attenuated correction" — the
    engine docstring now carries the audited wording); placebo asymmetry
    (corr_top1-vs-arbitrary-graph, ≥300 km and conditioning controls still not run for
    the stacked arm at any horizon — the h4-6 claims should stay scoped as "beats
    chance graphs"); ensemble-vs-resmlp cross-architecture equivalence was only
    established at h1-3 and has no h4-6 counterpart (resmlp h4-6 was never run) — do
    not extrapolate it.

## Bottom line

The h4-6 extension ran the audited engine unchanged under the exact h1-3 protocol; the
merge and the matched Li comparison are reproducible to the last digit from the
predictions files. The headline story is solid as stated: the stage-2 neighbor
correction is horizon-stable (~+1% at every h1-6, both seeds, 20/20 placebos, raw s1
ranks now interpretable), the ERA5 sequence gain dies after h4, and the stack does not
close the Li gap at h3-6 (−12.5% → −26.7%, robust to the coverage subset). Fix two
words in the RUN_LOG ("most of f5" → "all of f5"; the ~1.1% sentence), keep the
wrong-twin and matched-vs-full-sample quarantines exactly as they are.
