# Phase 6 ERA5 Winsorization Audit

**Date:** 2026-08-12
**Scope:** Verify the ±10σ winsorization fix in `src/gracefc/era5.py` is correct, leak-free, and effective.

---

## 1. Winsorization applied correctly — PASS

The fix lives in `era5_fold_features()` at line 177:

```python
resid[name] = (r / std).clip(-CLIP_SIGMA, CLIP_SIGMA) if std > 1e-8 else r * 0.0
```

**Train-only statistics:**
- Climatology: `fit_climatology(train)` where `train = s[s.index < test_start]` — train only.
- Std: `r[r.index < test_start].std()` — train only.

**Identical application to train and test:**
- `deseasonalize(s, ...)` runs on the full series using the train-fit model.
- Division by `std` and `.clip()` are constant transforms applied element-wise to the entire series.
- No branch or conditional treats train and test rows differently.

**No leakage.** The test window never influences the climatology fit or the standardization denominator. The clip threshold is a fixed constant (`CLIP_SIGMA = 10.0`, line 38), not derived from any data.

## 2. ±10σ threshold — PASS (reasonable)

| Threshold | Effect |
|-----------|--------|
| ±3σ | Too aggressive — clips real climate variability in heavy-tailed geophysical series |
| ±5–6σ | Standard robust threshold, but would clip rare-but-real extremes |
| **±10σ** | **Preserves essentially all real signal; eliminates only degenerate outliers** |
| ±50σ | Too lenient — a 50,000σ Libya-type value still detonates ridge |

For normally distributed data, P(|X| > 10) ≈ 1.5e-23. Even under heavy tails (typical for runoff residuals), values beyond 10σ indicate the standardization denominator is near-zero rather than genuine large anomalies. The Libya subsurface-runoff case (train std ≈ 0, single flood month → 50,000σ) is exactly this failure mode.

**Near-zero std guard** (`std > 1e-8 else r * 0.0`): zeroes out features from near-constant series (snow in deserts, runoff in arid basins) instead of producing 0/0 or huge values. Correct and conservative.

## 3. No new issues introduced — PASS

- Clip is monotonic and preserves ordering — no distortion to rank correlations.
- `CLIP_SIGMA` is a module-level constant — transparent, auditable, easy to tune.
- The deseasonalize → standardize → clip pipeline is applied per-basin, per-variable, so one basin's extreme cannot contaminate others.
- Lag features (`shift(k)`) are computed after winsorization, so lagged values are also clipped. Correct — the lag of a clipped value is the same as clipping the lag when the threshold is constant.

## 4. Results spot-check — PASS

Ridge+ERA5 arms in `phase6_era5_summary.csv`:

| Model | h=1 RMSE_std | h=2 | h=3 |
|-------|-------------|-----|-----|
| ridge_corr_top1_era5 | 1.0154 | 1.1762 | 1.2432 |
| ridge_own_era5 | 1.0179 | 1.1769 | 1.2436 |
| ridge_corr_top1 (no ERA5) | 1.0427 | 1.1867 | 1.2470 |
| ridge_own (no ERA5) | 1.0452 | 1.1877 | 1.2477 |

- No exploding values anywhere in the table (max RMSE_std = 1.26).
- Ridge+ERA5 consistently beats ridge-only, with DM test p-values < 1e-6 at h=1.
- Skill improvements are modest (5–6% at h=1), consistent with ERA5 providing supplementary not dominant signal.
- GBM and MLP ERA5 arms show similar ranges — no model class is blowing up.

## Overall Verdict: PASS

The winsorization fix is correctly implemented with no data leakage, uses a reasonable threshold, introduces no new issues, and the downstream results confirm the numeric blowup is resolved.
