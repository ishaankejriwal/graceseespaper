# Phase 8 code audit — experiment_lstm_combined (stacked LSTM + neighbor residual MLP)

Auditor: adversarial code audit subagent, 2026-08-13.
Scope: `src/gracefc/experiment_lstm_combined.py` (new), `src/gracefc/experiment_lstm.py`
(refactored), `scripts/run_phase8_lstm_combined.py` (new), against the phase 7 plumbing
(`phase7.py`, `experiment_resmlp.py`, `experiment_nonlinear.py`, `graphs.py`, `era5.py`,
`evaluate.py`, `kalman.py`, `stats.py`). No files were modified; smoke outputs were
re-verified independently.

## Verdict: PASS WITH NOTES

No leakage, no comparability break, no blocker. The headline lstmres-vs-LSTM comparison is
honest (it cannot be optimistic in the leakage sense), but three pieces of paper language
need scoping: the docstring's "never flatters" claim is too strong as stated, the summary
file's `dm_vs_ridge_twin` column is information-mismatched for the lstmres family, and the
nbrin placebo tests only the stage-2 increment.

## Independent re-verification performed

- Re-merged `results/phase8_lstm_combined_smoke_predictions.csv` against
  `results/phase7_lstm_predictions.csv` (fold f1, h1) on (model, name, issue_date,
  target_date) for the 7 shared arms: 27,846 rows, 1:1 merge, max |pred diff| = 0.0 and
  max |target diff| = 0.0 — bit-identity confirmed, including `lstm_corr_top1_era5_s0`,
  which exercises the refactored `train_lstm`/`lstm_predict` path end to end.
- All 9 real arms in the smoke file share the identical (name, issue_date, target_date)
  row set, 3,978 rows per arm.
- Placebo label prefix collision checked both directions:
  `"lstmres_nbrin_corr_top1_rand0".startswith("lstmres_corr_top1_rand")` is False and
  vice versa — no cross-family contamination in `summarize_and_write`'s prefix match.

## Findings

1. **NOTE — Causality/leakage sweep is clean.** Deseasonalization, per-basin std, and
   Kalman params are fit strictly pre-test (`phase7.py:20-28`, `kalman.py:53-64`); train
   rows require `target_date < fold.test_start` (`phase7.py:61`); LSTM windows look only
   backward from the issue date (`experiment_lstm.py:100-104`; window index = issue − 0..11);
   the ERA5 tensor is built from `era5_fold_features` with train-only climatology/std and
   the ±10σ clip applied identically train/test (`era5.py:164-186`,
   `experiment_lstm_combined.py:50-51`); the neighbor graph uses the train window only
   (`graphs.py:6-14` via `setup["train_src"]`, `phase7.py:37`,
   `experiment_lstm_combined.py:48`); the stage-2 residual uses train rows only
   (`experiment_lstm_combined.py:120`); the early-stop mask is the last 15% of TRAIN issue
   months (`phase7.py:93-97`). No feature, channel, residual, or placebo input touches any
   value at or after `test_start`.

2. **CAVEAT — Stage-2 in-sample residual: the evaluation is honest, but soften the
   docstring's "never flatters" claim.** The headline contrast
   (`lstmres_corr_top1_s{s}` vs `lstm_own_era5_s{s}`) shares the identical stage-1 net and
   identical `lstm_te` between challenger and reference
   (`experiment_lstm_combined.py:117-122`), so the DM/skill numbers measure exactly the
   out-of-sample effect of the added MLP term, which is trained with zero test
   information. The reported skill therefore cannot be optimistic about the pipeline:
   whatever the in-sample residual construction does, it changes how GOOD the learned
   correction is, not how honestly it is scored. Direction of the design bias: mostly
   attenuating — the LSTM absorbs neighbor-correlated structure on the 85% it trains on,
   shrinking the train residuals the MLP learns from, which pushes the correction toward
   zero (conservative). However, the docstring's claim "handicaps (never flatters) the
   correction stage" (`experiment_lstm_combined.py:12-15`) is too strong as a theorem:
   in-sample residuals also carry LESS target noise (the LSTM partially memorizes noise on
   its training rows, and the epoch is selected to minimize loss on the 15% val tail, so
   those residuals are mildly optimistic too), which can raise the signal-to-noise of the
   neighbor-predictable component and make the MLP's learning problem easier than it would
   be from honest out-of-sample residuals. That still produces genuine, honestly-measured
   test skill — but "never flatters the correction" should read "the comparison is honest;
   the in-sample construction plausibly attenuates the correction" in the paper.

3. **CAVEAT — What real-vs-placebo can and cannot certify (inherited).** The corr_top1
   neighbor is by construction the basin most correlated with the target basin, while
   degree-matched random neighbors are arbitrary basins. If the stage-1 residual contains
   own-state-predictable structure (e.g., LSTM miscalibration of persistence at test
   time), the real neighbor can proxy for the own signal and beat all 20 placebos without
   any cross-basin information transfer. This asymmetry is identical to the phase 5-7
   designs (`experiment_resmlp.py:88-122`) and was covered for the LINEAR effect by the
   ≥300 km and conditioning controls; those controls have not been run for this stacked
   arm. Scope the claim as "this specific graph beats chance graphs", not "neighbor
   information transfer proven".

4. **NOTE — Placebo mechanics are apples-to-apples.** Placebos share the seed-0 stage-1
   net, its residual, and its test prediction (`experiment_lstm_combined.py:124,133-135`),
   mirroring resmlp's shared ridge stage; the placebo pooled RMSE (summed monthly
   sum/count, `phase7.py:105-106`) is algebraically identical to the real arm's row-pooled
   RMSE (`phase7.py:111`); family prefix matching cannot collide (verified above). Two
   mild asymmetries, both neutral or anti-real-arm: (a) the `_s1` real arms are ranked
   against a placebo distribution built on the seed-0 net — per the phase 7 analysis the
   s0 net is the stronger one, so this is conservative for s1; (b) the placebo
   distribution confounds graph randomness with MLP seed randomness (seeds 1000+draw,
   `experiment_lstm_combined.py:135`), widening it relative to pure graph noise — the
   phases 5-7 convention, unchanged. With 20 draws the p_rank floor is 1/21 ≈ 0.048.

5. **CAVEAT — `dm_vs_ridge_twin` in the summary CSV is information-mismatched for the
   lstmres family; do not quote it.** `summarize_and_write` derives the ridge reference by
   string surgery: fam `lstmres_corr_top1` → `ridge_corr_top1` (`phase7.py:121`), which
   carries NO ERA5, while lstmres carries the full ERA5 sequence. Smoke confirms the
   inflation: summary DM −7.32 (p = 2e-6) vs the properly matched headline contrast
   against `ridge_corr_top1_era5` of −2.32 (p = .034). The runner's CONTRASTS
   (`scripts/run_phase8_lstm_combined.py:24-32`) contain the correct pairings — quote
   those. (`lstmres_nbrin_corr_top1` derives `ridge_nbrin_corr_top1`, absent, so it
   correctly gets no column.)

6. **NOTE — The nbrin placebo tests only the stage-2 increment.** Placebo arms of the
   nbrin family keep the REAL neighbor channel inside the stage-1 LSTM
   (`lstm_te_s0["lstm_corr_top1_era5"]`, `experiment_lstm_combined.py:133-135`) and
   randomize only the stage-2 MLP input. Its p_rank says nothing about the stage-1
   neighbor channel. Relatedly, `lstm_corr_top1_era5_s*` has no placebo distribution in
   the phase 8 files at all (its placebos live in the phase 7 outputs under a different
   crc base — different graph draws, so ranks are not directly transplantable).

7. **NOTE — experiment_lstm.py refactor is behavior-preserving.** `fit_lstm`
   (`experiment_lstm.py:91-97`) is now a pure composition of `train_lstm` + `lstm_predict`
   with identical defaults (hidden 32, 60 epochs, patience 6, batch 2048, lr 1e-3).
   `torch.manual_seed(seed)` sits at `train_lstm` entry (`experiment_lstm.py:49`) and the
   shuffle generator is separately seeded (`experiment_lstm.py:57`), so every torch RNG
   draw happens after reseeding inside the call — the extra `lstm_predict(net, Xtr)` calls
   phase 8 interleaves (`experiment_lstm_combined.py:120`) run under `no_grad`, consume no
   RNG, and cannot shift any stream. `lstm_predict`'s 8192-row batching is per-sequence
   independent, so batch size has no numeric effect. Empirically confirmed by the 0.0-diff
   merge against the phase 7 CSV (which includes the two LSTM arms). The phase 7 runner
   path through `run_lstm_experiment` calls only `fit_lstm` and is textually the same
   pipeline as before.

8. **NOTE — Row-set comparability holds.** All arms per fold-horizon are emitted from the
   same `te` frame; the ERA5 dropna is applied up front for EVERY arm including
   `kalman_ar1` and the GRACE-only ridges (`phase7.py:59-60`), matching phase 7's
   convention. Verified on smoke: identical row sets, 3,978 rows per arm. DM and bootstrap
   align models on common `target_date`s (`stats.py:46-49,62-65`), so equal row sets make
   the pooled tests exactly paired.

9. **NOTE — Seed/RNG hygiene is clean.** Placebo graph base =
   `crc32(b"phase8_lstm_combined") % 1e6` (`experiment_lstm_combined.py:44`) is distinct
   from phase 7's `b"phase7_lstm"` and resmlp's `b"phase7_resmlp"`, so phase 8 placebo
   graphs are fresh draws. Both phase 8 families share the same 20 graphs per fold
   (intentional: the same chance graph tests both stacks; the phase 7 "shared draws across
   families" caveat carries over within phase 8). Real MLP seeds {0,1} are disjoint from
   placebo seeds {1000..1019}; `random_degree_matched` uses a local `default_rng`
   (`graphs.py:75`); sklearn MLP is deterministic per `random_state` with its own internal
   10% random early-stop split (within train only — no leakage, though a time-ordered
   split would be the stricter control already flagged in the phase 7 analysis). No order
   dependence found.

10. **NOTE — Edge conventions inherited and symmetrical.** The −1 neighbor sentinel
    zero-fills real and placebo channels identically (`phase7.py:75-90`,
    `experiment_lstm.py:107-110`); degree-matched placebos produce an empty list exactly
    where the real graph has no positive-correlation neighbor (`graphs.py:78-81`), so
    capacity stays matched; pre-record window months zero-pad (the state prior mean);
    `_era5_state_tensor`'s `nan_to_num` (`experiment_lstm.py:119`) only affects
    basins/months already excluded by the issue-date dropna; the CLIP_SIGMA winsorization
    is applied inside `era5_fold_features` before both the flat ridge features and the l0
    tensor, so linear and sequence heads see consistently clipped values.

11. **NOTE — Stage-2 features are the rho^h-PROPAGATED neighbor state
    (`propagated_neighbor_features`), while the stage-1 nbrin channel is the raw filtered
    state history.** This matches resmlp's stage 2 and phase 7's LSTM channel
    respectively, so it is the intended like-for-like stacking of the two winners, but the
    methods section should state the two representations explicitly.

## Bottom line

The code is a faithful stacking of the two audited phase 7 winners with fresh placebo
draws and verified bit-identical shared arms. Nothing here can manufacture test skill.
The three CAVEATs are all about how results may be phrased, not whether they are real:
quote the runner's matched contrasts (not the summary's auto ridge twin), soften "never
flatters" to "honest but plausibly attenuated", and present placebo ranks as
graph-specificity evidence with the nbrin rank scoped to the stage-2 increment.
