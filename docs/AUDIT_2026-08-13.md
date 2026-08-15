# Repository-wide audit: GRACE TWSA forecasting study

Audit date: 2026-08-13  
Scope: current global study, manuscript, all Python source/runners, notebooks, raw and processed data schemas, archived results, and end-to-end lineage.  
Historical Africa context files were treated as background only, as requested.

## Executive verdict

The repository contains a substantial, unusually well-documented experiment archive, and the principal table arithmetic agrees with the archived CSVs. Python syntax/imports pass, the notebooks are read-only display notebooks, the core target/ERA5/Li schemas are mostly coherent, and many comparisons have explicit row-matching audits.

However, the study is **not submission-ready**. Four defects require corrected data or evaluation reruns, and two central manuscript interpretations need to be withdrawn or narrowed:

1. Two raw CSR solutions are assigned to the wrong calendar month and averaged into the preceding solution.
2. Target-date folds leak information relative to forecast issue dates at leads 2--6.
3. The significant neighbor result is concentrated in 35 basins that the code itself flags as smaller than one nominal GRACE resolution element and as leakage receivers.
4. The final stacked model was designed after inspecting the same test folds used for its reported inference.
5. The Kalman gain does not, by itself, identify a measurement-noise cost; the fitted observation-noise variance frequently collapses to a numerical boundary.
6. The manuscript's claim that neighbor input is "dead or harmful" in the LSTM selectively reports seed 0; the two-seed ensemble is nonsignificant.

The safest immediate decision is to freeze figure polishing and submission work, preserve the current archive, repair the data/evaluation foundation, and then rerun a locked paper pipeline.

## Pipeline and where the blockers enter

```text
CSR RL0603M NetCDF + HydroSHEDS/mascon mask
  -> src/gracefc/basins.py + scripts/build_basin_series.py
  -> basin_month_twsa_global.csv + basin_meta.csv
  -> fold-specific decomposition/standardization
  -> Kalman states + lag features + train-selected graphs
  -> phases 2--5 baselines, neighbors, nulls, architecture tests
  -> phases 6--8b ERA5, Li comparison, sequence/stack models
  -> paper ladder/headline tables -> paper/main.tex -> figures

NOAA text endpoints -> indices.csv -> phase-2 index baseline + phase-4 controls
CDS ERA5 yearly files -> basin ERA5 table -> phase 6/7/8 models
PANGAEA CSR-FCast files -> basin forecasts -> phase 6/8b matched comparison
```

The calendar-assignment defect is at the first transformation, so every derived target, fit, graph, and result is formally stale. The fold defect is in the shared evaluator, so it affects nearly every lead-2--6 experiment even after the calendar mapping is repaired.

## P0: publication and correctness blockers

### 1. Two CSR solutions are collapsed into the wrong months

`decode_csr_time()` converts the raw arc midpoint to a timestamp, after which `build_basin_series()` bins by the midpoint calendar month and averages duplicates (`src/gracefc/basins.py:122-126`, `153-156`).

Direct inspection found 257 raw solutions but only 255 processed months:

- centers 2011-10-16 and 2011-10-31 are both assigned to October 2011;
- centers 2015-04-16 and 2015-04-27 are both assigned to April 2015;
- the latter arcs extend into November 2011 and May 2015, and those months are not listed as missing in the product metadata;
- the builder averages each pair and leaves November 2011 and May 2015 absent.

Required action: obtain/use the official CSR solution-month convention, assert one processed month per raw solution except documented missing months, rebuild the basin table, invalidate all caches, and rerun the paper pipeline. Do not patch this with a general "round to nearest month" rule without validating all 257 solutions.

### 2. Leads 2--6 contain issue-relative future leakage

The decomposition and scale are fit on all observations before `fold.test_start` (`src/gracefc/evaluate.py:34-43`), while train/test membership is based on `target_date` (`src/gracefc/evaluate.py:53-56`; `src/gracefc/phase7.py:54-62`). The leakage guard only checks that training targets precede the target-window start (`src/gracefc/evaluate.py:47-50`).

For f1 at lead 6, the forecast targeting June 2019 is issued in December 2018, but its climatology, scale, graph, Kalman parameters, and learned model may use observations from January--May 2019. Archived phase-3b affected-row fractions are:

| Lead | Affected rows | Fraction |
|---:|---:|---:|
| 1 | 0 | 0.00% |
| 2 | 1,170 | 5.95% |
| 3 | 2,340 | 11.90% |
| 4 | 3,510 | 17.86% |
| 5 | 4,680 | 23.81% |
| 6 | 5,850 | 29.76% |

A diagnostic that merely drops the affected rows suggests the broad crossing remains, but the claim that Kalman beats per-basin ridge significantly at all six leads fails at lead 4 (`p=.0599`). This diagnostic is not a substitute for refitting on correct cutoffs.

Required action: define the evaluation on issue dates or use a horizon-specific embargo/cutoff no later than the earliest issue date in the fold. Refit every learned transform and model under that cutoff. The paper's blanket no-leakage statement at `paper/main.tex:305-314` must not remain until the rerun passes an issue-relative audit.

### 3. The pooled neighbor effect is driven by nominally unresolved basins

The metadata marks retained basins below 90,000 km2 as `below_resolution`, with the code comment "smaller basins are leakage receivers" (`src/gracefc/basins.py:111-112`). All 35 such basins remain in the 234-basin headline sample because runners filter only `exclude_reason == "keep"` (for example `scripts/run_phase3b_kalman_neighbors.py:43` and `scripts/run_phase8_lstm_combined.py:47`).

Read-only lead-1 recomputation of the registered linear contrast:

| Stratum | Basins | Skill | 95% bootstrap CI | DM p |
|---|---:|---:|---:|---:|
| >=90,000 km2 | 199 | +0.299% | [-0.101, +0.780] | .209 |
| <90,000 km2 | 35 | +1.145% | [+0.599, +1.644] | .000210 |
| Full sample | 234 | +0.488% | positive | .0218 |

The split persists with the >=300 km centroid restriction: larger basins +0.343% (`p=.147`), smaller basins +1.052% (`p=.00059`). A centroid cutoff does not remove shared mascons, overlapping effective footprints, or boundary-adjacent leakage.

Required action: make the >=90,000 km2 sensitivity a primary result; construct controls based on shared mask/mascon cells, effective-footprint overlap, or minimum boundary distance; and narrow "real regional hydroclimate" to "consistent with" unless the resolved-basin effect survives. As currently supported, the neighbor contribution cannot carry a strong standalone novelty claim.

### 4. Final-model inference is post-selection

Phases 3--7 inspected the same 2019--2026 folds across graph types, architectures, controls, horizons, and seeds (`results/RUN_LOG.md:302-341`). Phase 8 then explicitly combined the Phase-7 winners and evaluated that combination on those folds again (`results/RUN_LOG.md:343-359`; `src/gracefc/experiment_lstm_combined.py:1`, `27-40`). Nominal Phase-8 CIs and DM p-values do not include this architecture-selection uncertainty.

Required action: freeze a final specification and evaluate it on an untouched time period/product or nested outer evaluation. If no untouched evaluation is possible, label Phase 8 exploratory and remove language implying confirmatory registration or strict final-model inference.

### 5. The measurement-noise mechanism is not identified

The AR(1)-plus-white-noise model can assign true high-frequency hydrology, non-AR(1) dynamics, decomposition error, or correlated satellite error to `r`; performance against damped persistence is therefore not a clean measurement-noise ablation (`paper/main.tex:342-363`, `539-554`). The optimizer accepts the lowest objective without checking convergence or parameter boundaries (`src/gracefc/kalman.py:26-36`).

Archived diagnostics show:

- 273/1,170 basin-fold fits have `r < 1e-6`;
- 32 basins are at that near-zero boundary in all five folds;
- fitted `q/r` spans about 0.006 to 1.1e10.

Required action: store convergence diagnostics; compare the same dynamics with `r=0`; inspect residuals and profile likelihoods; test alternative latent dynamics; and replicate across CSR/JPL/GSFC or published uncertainty estimates. Until then, use "AR(1) state-space filtering" rather than "optimal observation-noise separation" or "the cost of forecasting measurement noise."

### 6. The LSTM delivery claim is seed-selective

The manuscript says direct neighbor input is dead/harmful using seed-0 contrasts (`paper/main.tex:892-901`) despite its ensemble-reporting rule (`paper/main.tex:415-419`). The archived analysis notes a sign flip (`results/phase7_analysis.md:113-117`). Recomputed two-seed results:

- seed 0: +0.036%, -0.778%, -0.725% at leads 1--3;
- seed 1: +0.845% (`p=.0062`), +0.682% (`p=.0476`), +0.099% (`p=.69`);
- ensemble: +0.428%, -0.120%, -0.312%, all nonsignificant.

Required action: report the input-channel result as seed-unstable with no robust ensemble gain. Remove "dead or harmful" from the abstract, contribution list, discussion, and conclusion.

## P1: major methodological and reporting issues

1. **LSTM architecture and information are confounded.** The LSTM gets 12-month state/ERA5 sequences (`src/gracefc/experiment_lstm.py:30`, `151`), while the ridge twin gets ERA5 lags 0--2 (`src/gracefc/era5.py:148`; `experiment_lstm.py:185`). Add a flattened 12-month ridge/MLP twin before attributing the gain to sequence modeling.

2. **The GNN twin has different inputs.** GNN messages include neighbor ERA5 (`src/gracefc/experiment_gnn.py:4`, `174`), while the flat ridge twin gets target ERA5 plus neighbor state (`153`, `166`). The comparison does not isolate graph architecture.

3. **The skill-curve crossing is not a causal source decomposition.** The two systems differ in architecture, training data, gap reconstruction, seasonal handling, and predictors (`paper/main.tex:446-469`); the local stack itself uses ERA5 soil-state variables. Say "empirical crossing consistent with an initial-condition/forcing crossover," and say "not detectably different" rather than "tie" at lead 2.

4. **Stage-2 stacking is not cross-fitted.** The MLP learns in-sample LSTM residuals (`src/gracefc/experiment_lstm_combined.py:12`, `119`). Test evaluation is separate, but this is not clean stacked generalization. Train stage 2 on rolling/out-of-fold stage-1 predictions.

5. **Statistical helpers do not enforce paired rows.** DM and bootstrap aggregate each model separately and intersect only target dates (`src/gracefc/stats.py:37-49`, `52-75`). Merge on name, issue date, target date, fold, and horizon; assert target equality and balanced coverage.

6. **Multiplicity and dependence are underhandled.** FDR is per basin, not across the many adaptively inspected headline arms/horizons/seeds (`src/gracefc/phase7.py:137-146`; `stats.py:79-103`). Bootstrap blocks remain fixed at three months even for leads 4--6. Declare hypothesis families, use adjusted pooled p-values, and add block-length 6/12 sensitivity.

7. **The stacked control battery is incomplete.** The >=300 km and conditioning controls were run for the linear arm, not the stack (`paper/main.tex:1184-1191`), while the abstract grammatically links the full controls to the six-lead stack. Separate those claims or port the controls.

8. **IAAFT uses the full record.** Surrogates include the test interval (`scripts/run_phase4_surrogates.py:45`, `60`). This may define an unconditional null, but it contradicts the blanket fold-pure description. Add fold-local/prefix sensitivity and avoid claiming that full-record construction is necessarily conservative.

9. **Phase-3 graph selection has a small interpolation leak.** Feature rows mask future interpolation anchors, but graph selection uses the filled training slice without the same legality mask (`src/gracefc/features.py:17-37`; `experiment.py:53`). The headline Kalman graph uses raw residuals and is unaffected.

10. **The LOFO selective policy is not forward-causal.** Selection for a fold uses the other four folds, including later folds (`scripts/run_phase6_basin_analysis.py:146-174`). Remove operational/honest-in-time language or rerun forward-only.

11. **ERA5 coverage is not screened by runners.** Thirteen of 234 retained basins have coverage below 50%, 79 below 90%, minimum 24.5%; the runner ignores the coverage table (`src/gracefc/era5.py:67`, `103`; `scripts/run_phase6_era5.py:43`). Existing sensitivity suggests the main gains survive, so this is a disclosure/automation issue rather than a blocker.

12. **Caches can silently go stale.** Kalman caches are keyed only by fold name and reused without data/code/preprocessing hashes (`scripts/run_phase3b_kalman_neighbors.py:53`; `src/gracefc/phase7.py:22-27`). Canonical output tags can be overwritten. Add run manifests and content-addressed caches.

13. **ERA5 ZIP/resume handling is inconsistent.** ZIP extraction can leave a directory that `year_done()` accepts, while ingestion globs only flat NetCDF files (`scripts/download_era5.py:29-45`; `src/gracefc/era5.py:77`). Validate requested months/variables/coordinates and normalize extracted outputs.

14. **Operational framing is too strong.** The evaluation assumes current GRACE and retrospective ERA5 values are available at issue time, yet the introduction motivates bridging release latency (`paper/main.tex:89-93`). Call this retrospective potential skill and define issue time relative to the last available observation.

## P2: concrete manuscript and repository inconsistencies

- `paper/main.tex:229` expands the file label to "RL06.3," but metadata says title/source `RL0603M` and `product_version=RL06.2`. Use the literal product title and disclose the metadata version.
- `context_global_study.md:23-26` says two-parameter Kalman; the manuscript says three fitted scalars. `paper/DECISIONS.md:26-29` incorrectly says the context was corrected.
- `paper/main.tex:262-268` says all 12 indices are only controls. Phase 2 drops AMM/PDO and uses the remaining 10 in the Table-1 ridge-plus-indices arm (`scripts/run_phase2_baselines.py:47-51`).
- `paper/main.tex:305-308` says n=19,656 per lead, while the matched baseline table shrinks to 18,486 at lead 6 (`paper/main.tex:497-500`). Scope the former to experiments with complete filtered-state rows.
- "45 fold-lead-seed cells" includes two actual seeds plus a derived ensemble. Report 30/30 per-seed cells and 15/15 ensemble cells separately (`paper/main.tex:973-976`).
- Figure 1 promises CIs not fully present under the stated references; recompute or revise (`paper/main.tex:534-543`).
- Figure-plan F4 requires 99 individual surrogate values, but only a summary is archived (`paper/FIGURE_PLAN.md:90-99`).
- Newey-West and moving-block bootstrap methods lack citations (`paper/main.tex:319-323`).
- Six figure includes are commented out and `figures/` is empty; LaTeX tools are absent; the draft has never been compiled.
- Authors/coauthors, affiliation, contributions, acknowledgements, code/data DOI, and an unverified Kalman/GRACE citation remain TODOs (`paper/main.tex:13-17`, `1067`, `1258-1278`).
- Six manuscript source comments point to absent `memory/...` files, weakening claim provenance.

## Reproducibility and engineering assessment

### What passed

- All Python files compile and core modules import in the current virtual environment.
- `pip check` reports no broken installed requirements.
- Both notebooks only read processed/results tables and do not mutate the pipeline.
- All 42 manuscript citation keys exist in the 42-entry bibliography.
- Table 1, the crossing table, the linear neighbor contrast, and the reported per-seed stack increments match their archived CSVs.
- ERA5 has 305 gapless months, 11 expected variables, and correct documented unit conversions/area aggregation.
- Li has 174 contiguous initializations and the 227-basin matched sensitivities are internally guarded.

### What is missing

- no Git metadata in this workspace;
- no README, license, `.gitignore`, `pyproject.toml`, requirements/lock file, or environment export;
- no automated test suite;
- no machine-readable raw-data manifest/checksum inventory;
- no single configuration/run manifest linking inputs, parameters, code version, outputs, and paper claims;
- no LaTeX toolchain or successful manuscript build.

The current environment uses NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, scikit-learn 1.9.0, xarray 2026.7.0, and PyTorch 2.13.0+cpu, but these are not pinned.

## Redundancy, archive, and cut plan

Do not delete evidence-bearing artifacts until the corrected rerun is complete. After that, separate the repository into `active/`, `archive/`, and release artifacts.

### Safe first cuts from the active tree

- Move/delete 25 smoke and superseded artifacts: about 122.2 MB.
- Remove generated `__pycache__` directories.
- Remove `data/raw/li2026/__MACOSX` (177 files, about 46 KB).
- Do not distribute `.venv` (about 1.4 GB); replace it with pinned metadata.
- Keep the seven logs (only about 90 KB) because they contain useful provenance.

### Compress or archive, not delete

- `results/` is about 2.88 GB; `phase3_sweep_predictions.csv` alone is about 874 MB.
- Keep small summaries/headlines/audits as CSV/Markdown.
- Convert retained row-level predictions to compressed Parquet/Zstandard.
- Archive large phase-3/5/7 row predictions and placebo-basin tables outside the active paper path after checksums are recorded.
- Retain compact summaries of failed architectures because the manuscript relies on those negative results.
- For Li raw data, keep either the verified source ZIP or the extracted tree in the working release, not both; retain the ZIP checksum in the manifest.

### Code consolidation targets

1. Create one fold context that owns issue cutoff, transforms, graph fit, row keys, provenance, and leakage assertions.
2. Replace repeated experiment loops with a declarative arm registry and a single paired evaluator.
3. Centralize summary/DM/bootstrap/FDR output and force paired-row validation.
4. Consolidate the nearly identical phase-7 runner boilerplate.
5. Move legacy Africa regression compatibility, old smoke paths, and superseded paper material to an archive.
6. Keep negative experiment engines only if needed for replication; otherwise retain their manifests/summaries and move code to `archive/experiments`.

## Recommended recovery sequence

1. **Freeze current outputs** as `archive/pre_audit_2026-08-13` with checksums; do not overwrite them.
2. **Fix CSR month assignment** using the official solution-month convention and add raw-to-processed cardinality/date tests.
3. **Redesign folds on issue time**, add horizon embargo tests, and make every transform accept an explicit information cutoff.
4. **Hash inputs/config/code into caches and output directories**; then invalidate the existing Kalman caches.
5. **Lock a minimal paper model set** before rerunning: damped persistence, per-basin ridge, Kalman, resolved-basin neighbor controls, ERA5 ridge, a 12-month flattened twin, final LSTM/stack, and Li comparison.
6. **Rerun spatial controls** with the 35 small basins excluded and with shared-mascon/effective-footprint exclusion.
7. **Add Kalman identification diagnostics** and an `r=0`/alternative-dynamics ablation.
8. **Use untouched confirmation or label the final stack exploratory.** Do not present post-selection p-values as confirmatory.
9. **Rewrite the manuscript from the corrected outputs**, removing seed-selective and causal overclaims.
10. **Only then generate figures, compile, package the environment/data manifest, and mint a reviewable archive DOI.**

## Suggested manuscript cuts

- Move the post-hoc hybrid splice (`paper/main.tex:715-731`) to a supplement or remove it.
- Remove the LOFO deployment paragraph unless rerun forward-only (`paper/main.tex:860-869`).
- Move the phase-5 failure laundry list (`paper/main.tex:871-887`) to a supplement.
- Compress repeated benchmark interpretation (`548-564`, `1038-1057`, `1223-1232`).
- Compress repeated neighbor mechanism text (`824-858`, `1097-1125`, `1240-1250`).
- Move detailed ERA5 regional failures/ablations (`576-591`) to a supplement so the corrected contributions remain visible.

## Bottom line

The archive is valuable and the numeric bookkeeping is stronger than average, but internal consistency is not the same as valid prospective evaluation. The current lead-1 filtering result is the most robust component; the long-lead inference requires issue-time correction; the neighbor mechanism is not established once nominal spatial resolution is respected; and the final stacked model is exploratory because the same folds shaped and evaluated it.
