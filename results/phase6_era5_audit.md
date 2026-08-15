# Phase 6 ERA5 Code Audit

Independent review of the ERA5 ingestion pipeline (`src/gracefc/era5.py`),
experiment engine (`src/gracefc/experiment_era5.py`), runner script
(`scripts/run_phase6_era5.py`), and output data (`data/processed/era5_basin_month.csv`).

---

## 1. Data Leakage — PASS

Verified every path where test-window information could contaminate training features.

**ERA5 climatology fit** (`era5.py:165-170`): `fit_climatology(train)` is called on
`train = s[s.index < test_start]` only. The fitted model is then applied to the full
series via `deseasonalize(s, fit)`, which uses frozen train-window coefficients — no
test statistics enter.

**ERA5 standardization** (`era5.py:169`): `std = r[r.index < test_start].std()` is
computed strictly on the pre-test residual. Division by this train-only std normalizes
both train and test rows identically.

**Train/test split** (`experiment_era5.py:74-75`): Split key is `target_date`, not
`issue_date`. Training targets are strictly before `fold.test_start`. Test rows have
`target_date >= fold.test_start` and `<= fold.test_end`.

**ERA5 feature join** (`experiment_era5.py:72-73`): Features join on `(issue_date, name)`.
Since lag-0 ERA5 is the issue month's meteorology and targets are h months ahead, all
features are temporally realized before the forecast target.

**MLP scaler** (`experiment_nonlinear.py:57`): `StandardScaler().fit(Xtr)` fits only on
training rows. No second path.

**Verdict**: No leakage found at any stage.

---

## 2. Unit Handling — PASS

**Accumulated variables** (tp, e, ro, sro, ssro): ERA5-Land monthly means of accumulated
fields are stored as mean daily rates (m/day). Code correctly multiplies by
`days_in_month` per time step (`era5.py:106`) and converts m→cm (×100). Leap years are
handled by `pd.Timestamp.days_in_month`.

**Evaporation sign** (`era5.py:23`): scale = −100. ERA5 convention is negative = water
leaving surface. Flip makes positive = evaporative loss. Spot-checked: small negative
outputs (e.g., −0.0007 cm for Alaska, Dec 2003) represent net condensation events —
physically valid, not a sign error.

**Temperature** (`era5.py:109-110`): K→°C via `val - 273.15` applied only when
`out_name == "t2m_c"`. Spot-checked: Alaska Jan 2001 = −7.9°C, Hudson Bay May 2026 =
−0.4°C. No values below −100°C or above +60°C in the entire dataset.

**Soil moisture** (swvl1–4): dimensionless m³/m³, scale = 1.0. Correct.

**SWE**: m→cm, scale = 100, no days scaling (instantaneous state). Alaska Jan 2001 =
145 cm — plausible deep snowpack.

**Precipitation sanity**: no negative values found. Alaska coast Jan 2001 = 18.9 cm
(~189 mm/month) — reasonable.

---

## 3. Spatial Alignment — PASS

**Grid mapping** (`era5.py:38-47`): 0.25° mask cells are mapped to 0.5° ERA5 cells via
nearest-neighbor rounding. Since mask centers sit at 0.125° offsets from ERA5 grid points,
no cell is ever equidistant between two ERA5 cells — rounding is deterministic.

**Weight matrix** (`era5.py:50-61`): Cos-latitude mask weights accumulate into ERA5 cells
via sparse matrix. Up to 4 mask cells (2×2 block) map to each ERA5 cell, correctly
aggregating area-weighted contributions.

**NaN handling** (`era5.py:95-99`): Ocean cells in ERA5-Land are NaN. Weighted mean
excludes them (`num/den` where `den` counts valid cells only). Coverage fraction
(`den/w_total`) is stored per basin for downstream screening.

**Coverage diagnostic**: 284 basins in both the data table and coverage file. Lowest
coverage basins are islands and ocean polygons (W_Hudson_Bay at 0.57%, Nusa_Tenggara at
24.5%). W_Hudson_Bay is already excluded as `water_body` in basin_meta.csv.

**Note (non-blocking)**: Five island basins with <50% ERA5-Land coverage remain in the
`keep` set (Nusa_Tenggara 24.5%, Solomon_Islands 27.0%, Puerto_Rico 25.8%, Maluku 27.7%,
Jamaica 33.3%). Their ERA5 features aggregate from thin coastal slivers. Not a code bug —
the coverage file is produced for exactly this kind of screening — but these basins'
ERA5-derived features may be unrepresentative.

---

## 4. CV Consistency — PASS

The experiment imports and defaults to `DEFAULT_FOLDS` from `evaluate.py`:

| Fold | Test start  | Test end    |
|------|-------------|-------------|
| f1   | 2019-06-01  | 2020-10-01  |
| f2   | 2020-11-01  | 2022-03-01  |
| f3   | 2022-04-01  | 2023-08-01  |
| f4   | 2023-09-01  | 2025-01-01  |
| f5   | 2025-02-01  | 2026-05-01  |

Same 5 rolling-origin folds used in Phases 2, 3, 3b, 4, and 5. Verified by grep across
all experiment files — every one imports from the same `DEFAULT_FOLDS` constant.

ERA5 data spans 2001-01 to 2026-05, covering all folds with ample training history.

---

## 5. Placebo Methodology — PASS

**Design** (`experiment_era5.py:124-140`): Placebos call
`random_degree_matched(graph, seed)` to rewire neighbors, then build feature matrices as
`[own_state, random_neighbor, era5]`. ERA5 columns (`era_tr`, `era_te`) are the same
objects used by the real arms — not recomputed, not randomized.

This correctly isolates the test: "does the *specific* correlated neighbor add signal
beyond own state + shared meteorology?" The null hypothesis is that any random neighbor
should do as well. ERA5 features are held fixed so they cannot confound the neighbor test.

**No-ERA5 anchor** (`experiment_era5.py:137-140`): A parallel `ridge_corr_top1_rand{seed}`
placebo without ERA5 re-anchors the Phase 3b null distribution on exactly the same rows.
This enables a controlled comparison of placebo distributions with vs. without ERA5.

**Degree matching** (`graphs.py:71-81`): Random graphs preserve per-node degree (number
of neighbors), drawing without replacement from all other nodes. Correctly prevents
structural confounds.

---

## 6. Bugs and Edge Cases — PASS (no issues found)

**Row alignment**: All arms — with and without ERA5, real and placebo — are trained and
evaluated on the identical row set. ERA5 NaN rows are dropped *before* any arm is fit
(`experiment_era5.py:73`), preventing sample-size confounds between arms.

**Gapless month guard** (`era5.py:122-126`): The aggregation function raises `ValueError`
if the monthly time axis has holes between ERA5 files. Prevents silent interpolation
errors from missing years.

**Near-constant series** (`era5.py:170`): Features where `std < 1e-8` (e.g., SWE in the
tropics) are zeroed out rather than producing 0/0 blowups. Correct.

**Minimum data guard** (`era5.py:165-166`): Basins with <36 months of pre-test ERA5 data
get NaN features (climatology fit requires ≥36 months). These propagate to `dropna` and
are excluded from all arms equally.

**Coverage computed once** (`era5.py:100-104`): ERA5 coverage fraction is computed from
the first time step's valid cells only. Since ERA5-Land is a reanalysis with spatially
complete land coverage, this should be stable across time. Minor — not a bug.

**Off-by-one in lag features**: `rw.shift(k)` with k=0 gives the current value, k=1 gives
the previous month, k=2 gives two months ago. Since features join on `issue_date` and
targets are `issue_date + h`, lags are temporally valid.

---

## Summary

| Check                  | Verdict | Notes                                    |
|------------------------|---------|------------------------------------------|
| Data leakage           | PASS    | Clean train-only fit and standardization |
| Unit handling          | PASS    | Accumulated vars, K→°C, sign all correct |
| Spatial alignment      | PASS    | Nearest-cell mapping, NaN-aware weights  |
| CV consistency         | PASS    | Same 5 folds as entire pipeline          |
| Placebo methodology    | PASS    | Only neighbor randomized, ERA5 fixed     |
| Bugs / off-by-one      | PASS    | No issues found                          |

**One advisory**: 5 island basins in the `keep` set have <50% ERA5-Land coverage.
Consider screening basins with `era5_coverage < 0.5` from ERA5-specific headline results,
or noting this as a limitation. The coverage file already supports this.
