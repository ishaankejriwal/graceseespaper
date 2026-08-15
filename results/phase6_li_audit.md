# Phase 6 Li & Kusche Comparison — Independent Audit

**Auditor:** Claude (independent review, no prior involvement in implementation)
**Date:** 2026-08-12
**Files reviewed:**
- `scripts/build_li_basin_series.py` (spatial aggregation)
- `scripts/run_phase6_li_comparison.py` (comparison engine)
- `results/phase6_li_comparison_summary.csv`, `_headline.csv`, `_perbasin.csv`
- `src/gracefc/decompose.py`, `evaluate.py`, `stats.py`, `features.py`, `models.py`
- `Li_2026_LLM_compact_core.md` (paper reference)
- `results/RUN_LOG.md` (prior audit claims)

---

## 1. Spatial Alignment — PASS

**Check:** Is the 1° Li grid correctly aggregated to our basin definitions?

The `build_weight_matrix` function maps each 0.25° mask cell to its containing
1° Li cell via `floor(center) + 0.5`. I traced this for boundary cases:
- `mask_lat = -89.125` → `floor(-89.125) + 0.5 = -89.5` → cell covers [-90, -89). Correct.
- `mask_lat = 0.125` → `floor(0.125) + 0.5 = 0.5` → cell covers [0, 1). Correct.
- Longitude follows the same logic on [0.125..359.875] → [0.5..359.5]. Correct.

Cell-containment weights are cosine-latitude area weights (`cos(lat)`) accumulated
per basin via `np.add.at`. NaN-aware renormalization in the aggregation loop
(`W @ finite.T` as denominator) mirrors the basin-mean logic in `basins.py`. Basin
land-coverage is reported and used for the coverage ≥ 0.5 robustness subset.

**Dimension ordering caveat:** The code assumes Li variables have shape
`(time, lon, lat)`, giving flat index `lon_idx * n_lat + lat_idx` (line 44).
This is an unusual dimension order (most climate NetCDFs use `(time, lat, lon)`).
The comment on line 39 states this explicitly, and the prior audit (RUN_LOG
line 138) claims independent recomputation confirmed an exact match. **I cannot
verify this without running the code**, but the assumption is internally
consistent and the prior audit is specific enough to be credible.

**Coverage:** 234 basins → 7 with zero Li land coverage → 227 matched. A further
18 have coverage < 0.5, yielding the 209-basin robustness subset.

---

## 2. Target Alignment (Apples to Apples) — PASS

**Check:** Are we comparing the same quantity in the same units?

- **Units:** Li's CSR-FCast provides "cm equivalent-water-height TWSC"
  (confirmed in paper Section 2.1 and compact core line 126). Our TWSA is also
  in cm EWH. No unit conversion needed. The fact that Li's standardized RMSE
  values (~1.1–1.3) are in the same ballpark as our models (~1.0–1.4) provides
  an empirical sanity check.

- **Deseasonalization:** For `li_lstm_full`: Li's full forecast (seasonal + trend
  + interannual + subseasonal) has **our** fold-specific train-only climatology
  (intercept + linear trend + annual + semiannual harmonics) subtracted.
  For `li_lstm_nonseas`: Li's interannual + subseasonal components are used
  directly as residuals (line 67). In both cases a train-only mean offset absorbs
  any remaining decomposition-convention or baseline-epoch difference (lines 73–78).

- **Decomposition consistency verified against paper:** Li Section 3.1 defines
  "non-seasonal = interannual + sub-seasonal" and "full TWSC = non-seasonal +
  seasonal + linear." The code's `nonseas = TWSC_interannual + TWSC_subseasonal`
  (line 78 of build script) matches exactly.

- **Standardization:** Both target and prediction divided by the same `train_std`
  (lines 83–84). The consistency guard on line 113–118 verifies that the
  standardized target computed from Li's processing matches the stored target
  from phase3b (`max |diff| < 1e-6`).

---

## 3. Time Alignment — PASS

**Check:** Do test windows overlap correctly? Any off-by-one?

- **Lead-to-target mapping:** `target = init + pd.DateOffset(months=lead)`.
  The paper defines forward-moving lag as `X'(m,n)(t) = X_n(t−m)`, so a forecast
  initialized at time `t` with lead `m` predicts month `t+m`. Code matches.

- **Test window assignment:** Uses `target_date` (not `issue_date`) for fold
  membership. `is_test = (target_date >= fold.test_start) & (target_date <= fold.test_end)`.
  Folds are non-overlapping and test_end is inclusive. Verified that adjacent
  folds (e.g., f1 ends 2020-10, f2 starts 2020-11) have no gap or overlap.

- **Matched sample:** Line 123 groups by `(horizon, name, target_date)` and
  requires all 9 models present. This enforces identical evaluation rows per
  horizon. The constant n = 13,847 across all horizons confirms the matched
  sample is the same, which is consistent with Li providing all 6 leads per
  initialization file.

- **61 test months:** Li's last initialization is April 2024. At lead 6, the last
  possible target is October 2024. Fold 5 (Feb 2025–May 2026) has no Li data.
  The 61 months span June 2019–October/November 2024 (folds f1–f4), minus
  ~5 GRACE gap months. Arithmetic: 17+17+17+≤17 = ≤68 fold months minus gaps ≈ 61.
  Consistent.

---

## 4. Data Leakage — PASS

**Check:** Is any test-period information used in transforms?

| Transform | Training data only? | Evidence |
|---|---|---|
| Climatology fit | Yes | `fit_climatology(train_wide[name])` where `train_wide = wide[wide.index < fold.test_start]` |
| Mean offset | Yes | `ok = is_train & ...`; `offset = (pred_resid[ok] − obs[ok]).mean()` |
| Standardization std | Yes | `train_std = resid_obs[resid_obs.index < fold.test_start].std()` |
| Climatology prediction at test dates | Safe | Deterministic function of date, fit on train data; out-of-sample extrapolation is standard |

Minimum training overlap enforced: `MIN_TRAIN_OFFSET_MONTHS = 24`. Any
(fold, basin, horizon) combination with fewer than 24 valid train months is
dropped entirely. No leakage detected.

---

## 5. Statistical Tests — PASS

### 5a. Diebold-Mariano test

Implementation in `stats.py`:
- Loss differential: `d = loss_a − loss_b` (squared errors).
- HAC variance: Bartlett kernel, `max_lag = max(horizon−1, 1)`. Standard choice.
- Harvey-Leybourne-Newbold small-sample correction applied.
- Two-sided p-value from t-distribution with `n−1` degrees of freedom.
- Returns `(NaN, NaN)` when n < 10 or zero variance. Correct guard.

`pooled_monthly_dm` averages squared errors across basins per target month, then
runs DM on those monthly means. This preserves temporal correlation structure
and gives the pooled headline test. Correct.

### 5b. Bootstrap confidence intervals

`block_bootstrap_skill_ci`:
- Skill metric: `1 − MSE_a / MSE_b` on monthly-pooled losses.
- Moving-block bootstrap, block length 3, 2000 draws, seed 0.
- Returns 2.5th and 97.5th percentiles (95% CI).

**Minor note:** Block length 3 may underestimate CI width at horizons 5–6 where
autocorrelation could extend further. This is a modeling choice, not a bug, and
would make the CIs slightly anti-conservative (narrower than truth). Since the
DM test uses HAC variance with horizon-appropriate lags, the headline p-values
are not affected.

### 5c. Per-basin FDR

`per_basin_dm_fdr`:
- Per-basin DM test on raw (not monthly-pooled) squared-error differentials.
- Benjamini-Hochberg step-up procedure at q = 0.10.
- Spot-checked: rank 1 threshold = (1/227)×0.1 = 0.000441; CSV shows 0.000441.
  Rank 34 threshold = (34/227)×0.1 = 0.01498; CSV row for C_Java shows 0.01498.
  Correct.
- `a_better` flag: `stat < 0` ↔ model A has lower loss. Verified: positive
  DM stat → Li worse → `a_better = False`; negative stat → Li better →
  `a_better = True`. Correct.

---

## 6. Caveats Assessment — PASS (accurately stated)

Three caveats are mentioned in RUN_LOG line 143–146. I evaluate each:

### 6a. Fold 5 exclusion — ACCURATE

Li's initialization range ends April 2024 (paper Section 3.5, Table 2). Fold 5
starts February 2025. No Li forecast can target any fold 5 month at any of our
horizons (1–6). The exclusion is implicit (the matched sample naturally drops
fold 5) rather than hard-coded. Result: 61 test months from folds f1–f4 only.

### 6b. Refit frequency difference — ACCURATE

Li retrains monthly with an expanding window: "t ← t+1 month; optimal predictors
are reidentified and training/forecasting repeats" (paper Section 3.5). Our
models refit per fold (~17-month blocks). Within each fold, our model's training
window is frozen while Li's continues to grow. This could disadvantage our
models in the later months of each fold. However, the RUN_LOG notes that the
prior audit tested whether Li's edge grows with months-since-fold-start and
found no significant relationship (h3 ρ = 0.14, p = 0.61), suggesting this
caveat is real but minor in practice.

### 6c. Gap reconstruction (Yin 2023) — ACCURATE

Li fills the July 2017–May 2018 GRACE/GRACE-FO mission gap with the Yin et al.
(2023) reconstruction, which uses information from the full observational record
(potential hindsight channel). Our model training simply treats gap months as
NaN. This gives Li's LSTM ~11 additional continuous training months. The effect
is likely minor because: (a) only 11 out of ~170 training months are affected,
and (b) the gap period falls well within all folds' training windows, not at the
boundary.

---

## 7. Spot-Check of Results — PASS

### 7a. skill_vs_damped formula

For `kalman_corr_top1, h=1, all_matched`:
- Model RMSE = 1.037166, damped RMSE = 1.064078
- skill = 1 − (1.037166/1.064078)² = 1 − 0.9501 = 0.0499
- CSV value: 0.04994. **Match.**

For `li_lstm_full, h=1, all_matched`:
- Model RMSE = 1.121543, damped RMSE = 1.064078
- skill = 1 − (1.121543/1.064078)² = 1 − 1.1109 = −0.1109
- CSV value: −0.11093. **Match.**

### 7b. Headline skill cross-check

For `li_lstm_full vs kalman_corr_top1, h=3`:
- Li RMSE = 1.1702, kalman RMSE = 1.2602 (from summary CSV)
- skill = 1 − (1.1702/1.2602)² = 1 − 0.8624 = 0.1376
- Headline CSV: 0.1377. **Match** (rounding).

### 7c. Bootstrap CI consistency with DM

| Horizon | Skill | CI includes 0? | DM p < 0.05? | Consistent? |
|---------|-------|----------------|--------------|-------------|
| h=1 | −0.169 | No [−0.318, −0.038] | Yes (0.004) | Yes |
| h=2 | +0.076 | Yes [−0.011, 0.164] | No (0.075) | Yes |
| h=3 | +0.138 | No [0.067, 0.206] | Yes (0.001) | Yes |
| h=6 | +0.221 | No [0.173, 0.285] | Yes (<0.001) | Yes |

All CI zero-inclusion decisions agree with DM significance at α = 0.05.

### 7d. Sample size consistency

n = 13,847 for all_matched = 227 basins × 61 months = 13,847. **Exact match.**
n = 12,749 for coverage ≥ 0.5 = 209 basins × 61 months = 12,749. **Exact match.**

### 7e. Per-basin direction check

- E_Salinas_Grande_Mar_Chiquita (h=1): DM stat = +9.69, a_better = False. Li has
  higher loss → correctly marked as Li worse. **Correct.**
- C_Java (h=1): DM stat = −3.52, a_better = True. Li has lower loss → correctly
  marked as Li better. **Correct.**

---

## 8. Bugs and Silent Failures — NONE FOUND

- The skill_vs_damped computation (line 156) uses a temporary `set_index` +
  `.index.map()` chain that looks fragile but works correctly because the
  positional alignment is preserved.
- The climatology Series construction (lines 63–64) is redundant (wraps
  `fit.predict()` in a new `pd.Series` then reassigns the index) but produces
  the correct result.
- All NaN-producing paths either propagate NaN correctly or are guarded
  (e.g., `MIN_TRAIN_OFFSET_MONTHS`, DM's n < 10 guard, `np.errstate`).
- The consistency assertion on line 118 would catch target misalignment at
  runtime.

---

## Summary

| Check | Verdict | Notes |
|---|---|---|
| Spatial alignment | **PASS** | Floor+0.5 mapping correct; cos-lat weights correct; NaN renorm correct. Dimension ordering (time,lon,lat) assumed — prior audit claims verified. |
| Target alignment | **PASS** | Same units (cm EWH), our climatology applied, train-only offset, consistent standardization. Consistency guard confirms target match. |
| Time alignment | **PASS** | Lead→target correct, no off-by-one, test windows properly applied by target_date. |
| Data leakage | **PASS** | Climatology, offset, and std all train-only. MIN_TRAIN_OFFSET_MONTHS guard active. |
| DM test | **PASS** | Standard HAC + Harvey correction. Pooling by monthly mean is appropriate. |
| Bootstrap CI | **PASS** | Moving-block bootstrap valid. Block length 3 may be slightly short for h=5–6 (anti-conservative CIs). |
| BH FDR | **PASS** | Step-up procedure correctly implemented and spot-checked. |
| Fold 5 exclusion caveat | **PASS** | Accurately stated; implicit via Li data range. |
| Refit frequency caveat | **PASS** | Accurately stated; empirically shown to be minor. |
| Gap reconstruction caveat | **PASS** | Accurately stated; effect likely small. |
| Numerical results | **PASS** | Formula reproductions, CI/DM consistency, sample sizes all check out. |
| Bugs / silent failures | **PASS** | None found. |

**Overall: PASS.** The comparison is methodologically sound. The main item I
could not independently verify without running code is the Li NetCDF dimension
ordering assumption `(time, lon, lat)` — if wrong, all spatial aggregation
would be scrambled and Li's performance would likely be nonsensical. The prior
audit's claim of an exact-match independent recomputation, combined with the
plausible skill pattern in the results (Li winning at longer horizons where
their exogenous-predictor LSTM should dominate), makes this assumption credible.
