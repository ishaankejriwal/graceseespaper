# Phase 8 corrected-pipeline audit — the `lstmres_own` control

Auditor: adversarial audit subagent, 2026-08-15.
Scope: the new neighbor-free stage-2 control `lstmres_own` in
`src/gracefc/experiment_lstm_combined.py`, and the phase 8 (h1-3) + phase 8b (h4-6) +
merge outputs produced with it. Everything numeric below was RECOMPUTED from the
prediction and placebo CSVs with the project venv (`.venv\Scripts\python.exe`,
`src/gracefc/stats.py`, `models.rmse`), or reconstructed by re-running the fold/frame
plumbing directly. No project file was modified except this report.

## Overall verdict: PASS WITH NOTES

Every claimed number reproduces exactly (max |skill diff| 3.3e-14, max |p diff| 3.0e-19).
The control is mechanically fair — I verified the self-index path produces exactly the
right quantity with zero off-by-one, zero wrong-axis indexing, zero future leakage, and
byte-identical row alignment to the neighbor arm. There is no leakage and no blocker.

But the control is **inferentially weaker than the RUN_LOG claims for it**, and one
placebo statement in `phase8b_h16_summary.csv` is an artifact rather than a real result.
Both need fixing before the manuscript. Details in sections 1, 2 and 4.

| # | Section | Verdict |
|---|---------|---------|
| 1 | Is the control a fair control? | **PASS** (mechanically), with a scoping note |
| 2 | Is the own-state control being negative expected? | **PASS WITH NOTES** — expected in sign, but the mechanism is not the one claimed |
| 3 | Independent recomputation of headline numbers | **PASS** — exact to 3e-14 |
| 4 | Placebo integrity | **PASS WITH NOTES** — two real defects, one claimed anomaly is an artifact |
| 5 | Integrity checks (n, dups, cache, ridge twins) | **PASS** |
| 6 | The Li crossing | **PASS** — exact |

---

## 1. Is `lstmres_own` actually a fair control? — PASS (mechanically)

### 1a. The self-index path produces exactly the right quantity

`experiment_lstm_combined.py:115` builds `self_idx = np.arange(len(names)).reshape(-1, 1)`
and passes it to the identical function the neighbor arm uses,
`propagated_neighbor_features` (`phase7.py:83-89`), at `:116-117`. That function does

```python
node = nbr_idx[p]                                   # p = tr_pos / te_pos
vals = frame["prop"][t[:, None], np.clip(node, 0, None)]
return np.where(node >= 0, vals, 0.0)
```

With `self_idx`, `node[r] == p[r]`, so `vals[r] == prop[t_idx[r], tr_pos[r]]`. And
`prop` is exactly the matrix whose `.ravel()` becomes the `kalman` column
(`phase7.py:47-53`). I verified this empirically rather than by inspection:

> **`max |fs - kalman| == 0.000e+00` (exact float equality) on every train and test row,
> in all 5 folds, at h1, h3 and h6.**

So the self feature IS the row's own propagated Kalman state — the intended quantity,
with no off-by-one in the time axis, no basin/time axis swap, and no lookahead (both arms
index `prop` at the **issue-date** row `t_idx`, never at the target date; the propagation
factor is each *source* basin's own `rho^h`, symmetric between arms).

### 1b. Everything except the information source is identical

`experiment_lstm_combined.py:129-137` is the decisive block:

```python
net     = train_lstm(Xtr, ytr, val_mask, s)        # ONE net
lstm_te = lstm_predict(net, Xte)                   # ONE test prediction
resid2  = ytr - lstm_predict(net, Xtr)             # ONE stage-2 target
emit(f"{stacked_of[arm]}_s{s}", kal_te + lstm_te + _fit_head("mlp", fn_tr, resid2, fn_te, s))
if arm == "lstm_own_era5":
    emit(f"lstmres_own_s{s}",   kal_te + lstm_te + _fit_head("mlp", fs_tr, resid2, fs_te, s))
```

- **Stage 1**: literally the same `net` object and the same `lstm_te` array. Not a
  retrain — the same tensor is added to both arms. Perfect.
- **Stage 2 target**: the same `resid2` array.
- **Stage 2 model**: same `_fit_head("mlp", ...)` (`experiment_nonlinear.py:57-60`) —
  `StandardScaler` + `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000,
  early_stopping=True, random_state=seed)`, **same seed `s`** in both calls.
- **Capacity**: both feature matrices are `(rows, 1)` — `k=1` for both `nbr_idx` and
  `self_idx` — so input dimension, scaler, and net width are identical.
- **Row alignment**: both derive from the same `frame`, the same `t_idx`/`e_idx` and the
  same `tr_pos`/`te_pos`. Byte-identical row sets confirmed downstream: all 15 arms have
  the same n at every horizon and the join in `stats._paired_losses` is `validate=
  "one_to_one"` with zero unmatched rows.

**Side check — did adding the control perturb the pre-existing arms?** No. `_fit_head`'s
MLP uses `check_random_state(seed)` (its own RNG) and `StandardScaler` uses none, so the
extra fit consumes no torch or numpy global RNG. Confirmed empirically: all 9 shared arms
(`kalman_ar1`, both `lstm_own_era5` seeds, both `lstm_corr_top1_era5` seeds, and all four
ridge arms) are **bit-identical** (max |diff| = 0.000e+00, n=57,564 each) between
`phase8_lstm_combined_predictions.csv` and `phase7_lstm_predictions.csv`. The control was
added for free.

### 1c. The `-1` sentinel asymmetry is EMPTY — quantified

This was the sharpest a-priori threat and it does not exist:

| Fold | Basins | Top-1 neighbor absent (`-1`) | Neighbor == self | Zero-filled rows, neighbor arm | Zero-filled rows, self arm |
|------|--------|------------------------------|------------------|-------------------------------|----------------------------|
| f1-f5 | 234 each | **0** | **0** | **0** (tr and te, h1/h3/h6) | **0** |

Every one of the 234 basins has at least one positively-correlated source in every fold's
training window, so `corr_topk` (`graphs.py:6-14`) never returns an empty list and
`neighbor_rank_matrix` (`phase7.py:74-80`) never writes a `-1`. Three sampled random
placebo draws also produce 0 absent neighbors (`random_degree_matched`, `graphs.py:71-81`,
draws a pool that excludes self and matches degree 1).

> **The zero-fill asymmetry accounts for exactly 0 of the 0 rows involved. It explains
> none of the gap.** This concern can be closed and stated as closed in the paper.

### 1d. Scoping note — what the control does and does not vary

Mechanically only the *index* differs. But the two features are not equally *novel to
stage 1*:

- The neighbor feature is a basin stage 1 never saw (stage 1 for the `lstmres_corr_top1`
  arm is `lstm_own_era5` — neighbor-free by construction, `:121`).
- The self feature is `own_state × rho^h`, a per-basin deterministic rescaling of channel
  0 of the stage-1 input at lag 0. Measured `corr(fs, own_state)` = 0.98 (h1), 0.92-0.94
  (h3), 0.84-0.87 (h6).

So `lstmres_corr_top1` vs `lstmres_own` contrasts *novel cross-basin information* against
*information stage 1 already consumed* — not against *equally novel but uninformative*
information. That second thing is what the random-graph placebo does, and it is the
control the "architecture vs information" claim actually needs. See section 2.

---

## 2. Is the own-state control being negative a red flag or expected? — PASS WITH NOTES

Expected in **sign**. But the stated mechanism ("re-feeding it should add nothing and may
overfit") is **not** what the numbers show, and the difference matters for the paper.

### 2a. The feature is not degenerate, and it is not fitting pure noise

Degeneracy check: the self feature is a single non-constant column with
`std(fs)` = 0.34-0.69 vs `std(fn)` = 0.37-0.72 — comparable spread, not collinear-to-death
(there is only one column, so there is nothing to be collinear *with*). Its linear
correlation with the stage-2-adjacent target is near zero:

| | `corr(feature, ytr)` |
|---|---|
| self feature | **-0.008 to +0.012** (essentially zero, all folds, h1/h3/h6) |
| neighbor feature | **+0.052 to +0.084** (small but consistently positive) |

So on a linear read the self feature carries no signal for the residual and the neighbor
feature carries a little. That is the naive "should add nothing" story.

But decomposing the actual test-time correction
`c = pred(lstmres_X_s) − pred(lstm_own_era5_s)` against the stage-1 error
`e = target − pred(stage1)` — where ΔMSE = E[c²] − 2·E[c·e] — shows the own-state
correction is **actively anti-informative, not noisy**:

| Arm | seed | h | corr(c, e) | variance cost (%) | signal term (%) | net skill (%) |
|---|---|---|---|---|---|---|
| lstmres_corr_top1 | 0 | 1 | **+0.111** | 0.268 | **+0.907** | **+0.639** |
| lstmres_corr_top1 | 0 | 6 | **+0.161** | 0.596 | **+2.492** | **+1.896** |
| lstmres_own | 0 | 1 | **-0.080** | 0.219 | **-0.811** | **-1.031** |
| lstmres_own | 0 | 6 | **-0.163** | 0.348 | **-1.956** | **-2.304** |

(Full 24-cell table recomputed; the pattern holds in every cell and both seeds:
`corr(c,e)` is **+0.11 to +0.16** for the neighbor arm and **-0.08 to -0.16** for the
own-state arm.)

Read that carefully. **Only 0.16-0.54 percentage points of the own arm's 1.03-2.30%
harm is the variance cost of adding a second stage. The other 79-85% is a correction
pointed in the systematically wrong direction.**

The mechanism is the documented in-sample-residual caveat
(`experiment_lstm_combined.py:12-19`): `resid2` is computed on rows the LSTM trained on,
where the LSTM has *over*-explained the own-state channel. The in-sample residual-vs-own-
state relation is therefore the sign-flipped version of the out-of-sample one, and the
stage-2 MLP faithfully learns it and applies it out of sample, where it hurts. So the
control is not "adds nothing"; it is "adds a reliably wrong thing", and the amount it
adds grows with the LSTM's own-state overfit.

### 2b. This makes `lstmres_own` a WEAK (too-easy) control — quantified against the placebos

The random-graph placebo family gives the calibration point the own-state arm cannot.
Placebos supply a feature that is novel to stage 1 (a random basin's propagated state)
but has the wrong identity — i.e. a genuine "any second stage on plausible-looking
information" baseline. Pooled over all 5 folds, in skill vs the shared stage-1 net:

| h | placebo mean skill (%) | placebo sd (RMSE units) | `lstmres_own_s0` skill (%) | `lstmres_corr_top1_s0` skill (%) |
|---|---|---|---|---|
| 1 | **-0.268** | 0.00066 | -1.031 | +0.639 |
| 2 | **-0.134** | 0.00076 | -1.451 | +1.330 |
| 3 | **-0.121** | 0.00094 | -1.836 | +1.288 |
| 4 | **-0.119** | 0.00078 | -2.030 | +1.256 |
| 5 | **-0.103** | 0.00101 | -1.769 | +1.512 |
| 6 | **-0.050** | 0.00074 | -2.304 | +1.896 |

Two things fall out:

1. **The placebo mean harm (-0.27% to -0.05%) matches the pure-variance cost predicted by
   the decomposition (0.22-0.44%) almost exactly.** So a random-information second stage
   really is "pure noise cost" — the naive story is true of the *placebo*, not of
   `lstmres_own`.
2. **`lstmres_own` is a far outlier, 4x to 30x worse than any placebo.** It sits **+5.7
   to +21.4 placebo standard deviations on the WRONG side** of the placebo mean, and is
   beaten by **0/20 placebos at every single horizon**.

### 2c. Explicit answer: fair comparator, or does the paper need the random-graph placebo as primary?

**`lstmres_own` is a mechanically fair but inferentially insufficient control. The
random-graph placebo must be the primary control for the architecture-vs-information
claim.** Reasons, in order:

- The RUN_LOG claim "the two-stage ARCHITECTURE alone does not explain the gain — a
  second stage on information stage 1 already consumed actively overfits" is *true as
  written* but does not establish what the paper wants. It rules out one specific
  degenerate second stage. It does not rule out "any novel feature at stage 2 helps",
  because the own-state feature is not novel. Only the placebo tests that, and it does:
  novel-but-wrong-identity features give **-0.27% to -0.05%**, while the real neighbor
  gives **+0.64% to +1.90%**.
- The **+1.65 to +4.11%** contrast is inflated, and now quantifiably so: 79-85% of the
  control's deficit is anti-informative mis-direction, an artifact of the in-sample
  residual protocol, not a property of "the architecture". The RUN_LOG discipline note
  ("never quote it as the neighbor's skill contribution") is correct and should be
  hardened to "do not quote this contrast as the architecture term either."
- The two defensible numbers agree with each other, which is the strongest thing in this
  batch:

  | h | vs its own stage-1 net (s0) | vs random-graph placebo median (s0) |
  |---|---|---|
  | 1 | +0.64 | **+0.92** |
  | 2 | +1.33 | **+1.45** |
  | 3 | +1.29 | **+1.39** |
  | 4 | +1.26 | **+1.40** |
  | 5 | +1.51 | **+1.60** |
  | 6 | +1.90 | **+1.96** |

  Two independently-constructed references land within 0.1-0.3pp of each other at every
  lead. That is the paper's claim, and it is robust.

- Keep `lstmres_own` — but reframe it. Its honest value is as evidence *for* the in-sample
  residual caveat the docstring already flags, and as a demonstration that the stage-2
  MLP has enough capacity to hurt (so the neighbor arm's gain is not a capacity artifact).
  Report it as a diagnostic, not as the architecture control.

---

## 3. Independent recomputation of the headline numbers — PASS

Recomputed all 36 cells (3 contrast families × 2 seeds × 6 leads) from
`phase8_lstm_combined_predictions.csv` + `phase8b_lstm_h46_predictions.csv` using
`stats.block_bootstrap_skill_ci` and `stats.pooled_monthly_dm`, and compared against
`results/phase8b_h16_headline.csv`.

> **max |skill_pct difference| = 3.33e-14; max |dm_p difference| = 2.98e-19. All 36 cells
> match, including the bootstrap CIs (deterministic at seed 0). No mismatch anywhere.**

Spot values, claimed vs recomputed (identical):

| contrast | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| `lstmres_corr_top1_s0` vs `lstm_own_era5_s0` | +0.639 | +1.330 | +1.288 | +1.256 | +1.512 | +1.896 |
| `lstmres_corr_top1_s1` vs `lstm_own_era5_s1` | +1.191 | +1.563 | +1.356 | +1.369 | +1.282 | +1.935 |
| `lstmres_own_s0` vs `lstm_own_era5_s0` | -1.031 | -1.451 | -1.836 | -2.030 | -1.769 | -2.304 |
| `lstmres_own_s1` vs `lstm_own_era5_s1` | -1.300 | -1.445 | -1.978 | -2.233 | -1.941 | -1.328 |
| `lstmres_corr_top1_s0` vs `lstmres_own_s0` | +1.653 | +2.741 | +3.068 | +3.220 | +3.224 | +4.105 |
| `lstmres_corr_top1_s1` vs `lstmres_own_s1` | +2.459 | +2.966 | +3.269 | +3.523 | +3.161 | +3.220 |

All DM p ≤ 1.78e-5 for the first two rows and ≤ 8.71e-9 for the last two, as claimed.
Claim "+0.6 to +1.9% at every lead, both seeds" — **confirmed** (range +0.639 to +1.935).
Claim "+1.65 to +4.11%, all p ≤ 8.7e-9" — **confirmed** (range +1.653 to +4.105).
Claim "own-state control is itself NEGATIVE, -1.0 to -2.3%" — **confirmed** (range
-1.031 to -2.304).

### 2-seed ensemble (NOT currently in any shipped table — computed here)

Averaged the two seeds' predictions per `(name, target_date, horizon)` row, then scored:

| challenger vs reference | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| `lstmres_corr_top1_ens` vs `lstm_own_era5_ens` | **+0.908** | **+1.447** | **+1.334** | **+1.323** | **+1.420** | **+1.961** |
| `lstmres_own_ens` vs `lstm_own_era5_ens` | -1.165 | -1.461 | -1.904 | -2.118 | -1.839 | -1.801 |
| `lstmres_corr_top1_ens` vs `lstmres_own_ens` | +2.049 | +2.866 | +3.178 | +3.370 | +3.200 | +3.696 |
| `lstm_own_era5_ens` vs `ridge_own_era5` | +2.174 | +2.355 | +1.431 | +0.808 | -0.044 ns | -0.113 ns |
| `lstmres_corr_top1_ens` vs `ridge_corr_top1_era5` | +2.748 | +3.856 | +2.923 | +2.516 | +1.785 | +2.229 |

All ensemble contrasts in the first three rows have DM p < 1e-9. The ensemble number
(+0.91 to +1.96%) is the tidiest form of the headline claim and sits inside the per-seed
range at every lead. **Recommend the paper quote the ensemble row and report the per-seed
range as the robustness check.** Note `phase8b_h16_headline.csv` contains no ensemble
rows at all — `run_phase8_lstm_combined.py:25-41` never asks for them (only
`run_phase8b_merge.py` builds ensembles, and only for the Li comparison). Worth adding.

---

## 4. Placebo integrity — PASS WITH NOTES

### 4a. The reported ranks are correct

Verified against `results/phase8b_h16_summary.csv`: `lstmres_corr_top1` beats **20/20**
placebos (`p_rank = 0.047619`) at every lead and seed **except s1 at h2 (16/20,
p_rank = 0.238095)**. Reproduced exactly by re-pooling the placebo monthly losses.
Placebos ride the correct shared stage-1 network for seed 0 — `experiment_lstm_combined.py:138-139`
caches `resid_s0`/`lstm_te_s0` only when `s == seeds[0]`, and `:148-150` adds
`lstm_te_s0` to the placebo prediction, so the placebo family is the seed-0 stage-1 net
with only the stage-2 graph randomized. Correct by design.

### 4b. DEFECT 1 — the placebo test is only valid for seed 0, and that fully explains the s1/h2 "anomaly"

The placebos use the **seed-0** stage-1 net but `summarize_and_write` (`phase7.py:112-119`)
scores **both** seeds against them, by raw pooled RMSE. The seed-1 stage-1 net is worse
than seed-0 by **+0.0019 to +0.0093 RMSE**, while the entire placebo distribution spans
only **0.0007-0.0010 RMSE (sd)**. So the s1 arm is handicapped by roughly 2-12 placebo
standard deviations before the graph is even considered. At h2 the handicap is largest
(+0.0093) and that is exactly the cell that fails.

Repeating the test in **skill space** — each arm scored against *its own* stage-1 net,
placebos against theirs — removes the handicap:

| h | placebo median skill (%) | placebo max skill (%) | real s0 (%) | real s1 (%) | beaten s0 | **beaten s1 (fair)** |
|---|---|---|---|---|---|---|
| 1 | -0.283 | +0.091 | +0.639 | +1.191 | 20/20 | **20/20** |
| 2 | -0.123 | +0.106 | +1.330 | +1.563 | 20/20 | **20/20** |
| 3 | -0.099 | +0.063 | +1.288 | +1.356 | 20/20 | **20/20** |
| 4 | -0.144 | +0.104 | +1.256 | +1.369 | 20/20 | **20/20** |
| 5 | -0.083 | +0.172 | +1.512 | +1.282 | 20/20 | **20/20** |
| 6 | -0.062 | +0.127 | +1.897 | +1.935 | 20/20 | **20/20** |

The real arm's skill exceeds the best of 20 placebos by an order of magnitude in every
cell. **The "16/20 at s1 h2" caveat is a scoring artifact of comparing a seed-1 arm to
seed-0 placebos, not evidence of a weak cell.** This is good news for the claim, but the
fix is to *rerun 20 placebos on the seed-1 stage-1 net*, not to quote my skill-space
repair (which assumes the placebo corrections transfer across stage-1 nets). Until that
rerun exists, the honest statement is: "the raw placebo rank test is a like-for-like test
for seed 0 only; it is 20/20 at every lead."

### 4c. DEFECT 2 — the 12 cells are NOT independent; there are only 20 draws total

`experiment_lstm_combined.py:142-143` calls `random_degree_matched(graph, base + seed)`
with `base = zlib.crc32(b"phase8_lstm_combined") % 1_000_000 = 768044` (`:48`). The seed
depends on the draw index only — **not on the fold and not on the horizon**. And because
every basin has degree exactly 1 and the `names` ordering is identical in every fold
(verified: `names` list is equal across all 5 folds), `random_degree_matched` returns the
**same graph** for a given draw index everywhere. Verified directly:

> **Draw `rand{k}` is the identical random graph in all 5 folds — and, by construction
> (horizon is not in the seed), at all 6 horizons and in both placebo families.**

So the entire placebo evidence base for phase 8/8b is **20 random-graph realizations**,
reused across 5 folds × 6 horizons × 2 families × 2 seeds. The 12 lead/seed cells are
strongly dependent, essentially 12 re-scorings of the same 20 draws. **The paper must not
present them as 12 independent p<0.05 events.** (The pre-audit caveat named horizon reuse;
the fold reuse is additional and was not previously stated.)

Also note the rank test's p-value floor: with 20 draws, `p_rank = (1 + #{≤})/(1 + 20)`
bottoms out at **1/21 = 0.0476**. Every "significant" cell is pinned at that floor, so the
rank p-value understates the evidence badly. The z-scores are the real story and are
enormous: the real s0 arm sits **-6.8 to -18.6 placebo sd** below the placebo mean (h1
-6.8, h2 -11.1, h3 -9.2, h4 -11.3, h5 -10.8, h6 -18.6). Report the z-score or the
skill-space margin, not the floored rank p.

### 4d. Missing — there is no placebo family for `lstmres_own`

`placebo_beaten`/`p_rank` are NaN for `lstmres_own_s{0,1}` in
`phase8b_h16_summary.csv` because no `lstmres_own_rand*` family is emitted. That is
correct behavior for `summarize_and_write` (`phase7.py:112-117` finds an empty
distribution and skips), not a bug — but it means the control has no null distribution of
its own. Reported here so no one reads the blank cells as "0/20".

---

## 5. Integrity checks — PASS

- **Row counts / issue-date protocol.** 234 basins at every horizon; issue months
  83/82/81/80/79/78 for h1..h6, decreasing by exactly one per lead as the shifted target
  drops the last issue month. n = 19,422 / 19,188 / 18,954 / 18,720 / 18,486 / 18,252,
  **identical for all 15 arms at each horizon** (`cnt.nunique(axis=1) == 1` everywhere).
- **Duplicates.** 0 duplicated `(model, name, target_date, horizon)` and 0 duplicated
  `(model, name, issue_date, horizon)` across the merged h1-6 predictions.
- **Date consistency.** 0 rows where `issue_date + h months != target_date`.
  0 keys where the stored `target` disagrees across models.
- **Fold membership.** Every fold's test issue dates lie inside `[test_start, test_end]`:
  f1 2019-06..2020-10, f2 2020-11..2022-03, f3 2022-04..2023-08, f4 2023-09..2025-01,
  f5 2025-02..2026-04 (against test_end 2026-05). Issue-date protocol respected.
- **Kalman params cache — the fingerprinted one was used.** Recomputed
  `cache.data_fingerprint` over the current basin table + fold spec + `"issue-date-v1"`:
  `1e4447c6…64` — **identical** to the `__fingerprint__` stored in
  `results/kalman_fold_params.pkl`. `load_params_cache` returns all 5 folds (no silent
  refit). File mtime 2026-08-14 01:04, predating both runs.
- **Ridge twin agreement.** At h1-3 (the only leads where counterparts exist —
  phase 6/7 files are h1-3 only), all shared arms are **bit-identical**, max |diff| =
  0.000e+00 at n = 57,564 each:
  - vs `phase7_lstm_predictions.csv`: `kalman_ar1`, `ridge_own`, `ridge_own_era5`,
    `ridge_corr_top1`, `ridge_corr_top1_era5`, `lstm_own_era5_s{0,1}`,
    `lstm_corr_top1_era5_s{0,1}` — 9/9 bit-identical.
  - vs `phase7_resmlp_predictions.csv` and `phase6_era5_predictions.csv`: 5/5
    bit-identical each.
  h4-6 has no phase 6/7 counterpart to check against (all prior files stop at h3).
- **Merge losslessness.** `phase8b_h16_summary.csv` = 45+45 = 90 rows,
  `phase8b_h16_headline.csv` = 33+33 = 66 rows; both equal the concat-and-sort of their
  parts to CSV round-trip ULP (max value diff 4.4e-16 in `skill_pct`/`ci_hi_pct`,
  1.3e-29 in `dm_p`; identical columns, dtypes, and NaN placement). All four
  `lstmres_own` contrast pairs are present at all 6 horizons in the merged headline.
- **Run ordering.** Engine `experiment_lstm_combined.py` mtime 2026-08-14 22:42;
  h1-3 predictions 2026-08-15 00:18; h4-6 predictions 01:44; merged tables 01:45-01:46.
  Both runs post-date the engine edit — the two horizon blocks used identical engine
  bytes.

---

## 6. The Li crossing — PASS

`results/phase8b_li_comparison_predictions.csv`:

- **13 models × 13,620 rows at each of h1-h6**, with **227 basins × 60 months** at every
  horizon for every model — a complete rectangle, exactly as claimed. 0 duplicated
  `(model, name, target_date, horizon)`.
- The matched-sample rule in `run_phase8b_merge.py:80-84` (keep keys held by all 13
  models) is what produced it, and our predictions on the matched subset are identical to
  the full-sample predictions (max |pred diff| 8.9e-16 = CSV round-trip ULP at h1 and h6),
  so the restriction subsets rows without altering any of them.
- The `target` equality guard against the Li rows (`run_phase8b_merge.py:87-92`, tolerance
  1e-6) is present and passed at run time.

Recomputed every crossing number independently with `block_bootstrap_skill_ci` /
`pooled_monthly_dm` on the matched file:

| challenger vs `li_lstm_full` | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| `lstmres_corr_top1_ens` | **+20.011** | -3.493 ns | -12.936 | -17.436 | -23.642 | -30.341 |
| DM p | 3.44e-4 | 0.504 | 0.0153 | 2.13e-3 | 1.33e-4 | 1.36e-6 |
| `kalman_ar1` | **+14.628** (p=.0039) | -6.675 ns | -16.297 | -22.675 | -28.237 | -33.820 |
| `lstm_own_era5_ens` | +19.358 | -5.069 ns | -14.540 | -19.129 | -25.489 | -32.967 |
| `ridge_corr_top1_era5` | +17.346 | -7.961 ns | -16.767 | -21.023 | -26.283 | -33.513 |

| other pairs | h1 | h2 | h3 | h4 | h5 | h6 |
|---|---|---|---|---|---|---|
| `lstmres_corr_top1_ens` vs `li_lstm_nonseas` | +30.939 | +8.964 (p=.0083) | -1.741 ns | -8.910 | -16.202 | -23.383 |
| `lstmres_corr_top1_ens` vs `damped_persistence_rho` | +11.142 | +11.784 | +13.382 | +16.586 | +15.548 | +14.161 |
| `li_lstm_full` vs `damped_persistence_rho` | **-11.089** (p=.057) | +14.761 | +23.303 | +28.971 | +31.697 | +34.143 |

> **Every value matches `results/phase8b_li_comparison_headline.csv` to 0.0 (max
> difference 0.0 across all 30 recomputed cells).** The RUN_LOG's quoted figures
> (+20.0 / -3.5 / -12.9 / -17.4 / -23.6 / -30.3; kalman +14.6 p=.0039; nonseas +30.9 and
> +9.0 p=.0083; damped +11.1..+16.6; Li's -11.1 at h1) all reproduce.

Methodology is the phase 6 protocol verbatim: same `stats.py` functions, same matched-rectangle
construction, same standardized-deseasonalized space with train-only offsets carried over
from `phase6_li_comparison_predictions.csv`. **Internally consistent — PASS.**

Two wording cautions, neither a numeric error:
- The crossing point "between h2 and h3" rests on an h2 value that is **not significant**
  (-3.49%, p=0.504, CI -16.5% to +7.6%). State it as "indistinguishable at h2".
- `li_lstm_full` vs `damped_persistence_rho` at h1 is **p=0.057**, not significant at 5%.
  The RUN_LOG's "THEIRS IS -11.1% AT h1 (worse than damped persistence)" should carry that
  p-value; it is suggestive, not established.
- Prior `phase8b_audit.md` (2026-08-13) recorded the matched sample as 13,847 = 227 × 61.
  It is now 13,620 = 227 × 60 — one month fewer, consistent with the issue-date protocol
  correction. Expected, but worth a line in the RUN_LOG so the change is not read as drift.

---

## Ranked list of required fixes before these numbers go in the manuscript

**P0 — must fix or the claim is mis-stated**

1. **Demote `lstmres_corr_top1` vs `lstmres_own` (+1.65 to +4.11%) from the
   architecture-vs-information claim.** Section 2 shows 79-85% of the control's deficit is
   anti-informative mis-direction from the in-sample residual protocol, not an
   "architecture" cost. Promote the **random-graph placebo** to primary control for that
   claim; it gives the neighbor increment as **+0.92 to +1.96%**, in agreement with the
   vs-stage-1 number (+0.64 to +1.90%). Quote those two; report the `lstmres_own` contrast
   as a diagnostic only, with the mis-direction decomposition attached.
2. **Rerun the 20 placebos on the seed-1 stage-1 net** (currently
   `experiment_lstm_combined.py:138-150` caches seed-0 only). Without it, the "20/20 at
   every lead and seed except s1 h2" sentence is not a like-for-like test for seed 1. My
   skill-space repair says all 12 cells are 20/20 once the handicap is removed, but that
   is a diagnostic, not a substitute. If the rerun is not affordable, state explicitly:
   "the placebo rank test is like-for-like for seed 0 only, where it is 20/20 at all six
   leads."
3. **State the placebo dependence structure honestly.** The 20 draws are reused across
   all 5 folds, all 6 horizons, both families and both seeds (verified). The 12 cells are
   one correlated evidence body of 20 realizations, not 12 independent events. Also stop
   quoting `p_rank = 0.0476` as if it were a measured p — it is the 1/21 floor. Use the
   z-score (-6.8 to -18.6 placebo sd) or the skill-space margin instead.

**P1 — strongly recommended additional control**

4. **Add a random-graph placebo family for `lstmres_own`**, or better, add a *matched-novelty*
   control: a second stage fed a feature that is novel to stage 1 and of comparable
   variance but causally irrelevant (the existing random-graph placebo already is this —
   so the cheaper fix is simply to *report the placebo family as the control* rather than
   building a new arm). This is the single change that closes the architecture-vs-information
   question cleanly.
5. **Address the in-sample residual protocol directly**, since section 2 shows it does real
   damage. An out-of-fold stage-2 residual (inner time-split, LSTM predicting held-out
   train months) would make `lstmres_own` behave like the placebos (≈0) and would make the
   neighbor increment interpretable without the caveat. This is the most valuable
   additional experiment in the batch, and it would likely *raise* the neighbor number.

**P2 — reporting hygiene, no rerun needed**

6. **Add the 2-seed ensemble contrasts to `phase8b_h16_headline.csv`.** They are the
   cleanest headline form (+0.91 / +1.45 / +1.33 / +1.32 / +1.42 / +1.96%, all p<1e-9) and
   are currently computed nowhere — `run_phase8_lstm_combined.py:25-41` has no ensemble
   pairs. Values are in section 3 of this report.
7. **Close the zero-fill concern in writing.** 0 of 234 basins lack a top-1 neighbor in any
   fold; 0 rows are zero-filled in either arm. The neighbor/self asymmetry does not exist
   and the paper can say so.
8. **Soften two Li sentences**: h2 (-3.5%) is not significant (p=0.504), and Li-vs-damped
   at h1 (-11.1%) is p=0.057.
9. **Log the 13,847 → 13,620 matched-sample change** in `RUN_LOG.md` so it reads as the
   expected consequence of the issue-date correction rather than unexplained drift.
10. **Note that h4-6 ridge twins have no cross-file counterpart** (phase 6/7 stop at h3).
    The bit-identity evidence covers h1-3 only; h4-6 comparability rests on the shared
    engine and the fingerprinted cache, both verified.
