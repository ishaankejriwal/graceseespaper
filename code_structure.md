# Code Structure

A guide to how this project's code is laid out and what each piece does.
Last updated: 2026-08-15 (through corrected reruns, stratification, manuscript rewrite, figures).

## The basic idea

- **`src/gracefc/`** — the engine. Plain functions that do the real work. Nothing runs on its own.
- **`scripts/`** — the runners. Each one does a job start to finish and saves results as CSVs.
- **`notebooks/`** — the display layer. They only read result CSVs and make tables and plots.
- **`paper/`** — the HESS manuscript (LaTeX), figure plan, and decision log.

**Rule of thumb:** if it computes something, it lives in `src/` and is called by a script. If it draws something, it lives in a notebook. If it argues something, it lives in `paper/`.

## Running anything

```bash
.venv/Scripts/python scripts/run_phase2_baselines.py
```

The venv has everything including CPU-only PyTorch (for the phase 7/8 models).

## `src/gracefc/` — the engine

**Core pipeline**

| File | What it does |
|---|---|
| `basins.py` | GRACE file + basin masks → one table of monthly water storage per basin, plus each basin's centre, area, continent, keep/exclude flag. |
| `decompose.py` | Removes trend + seasonal cycle, fitted on training months only (no future leaks). Can add them back. |
| `features.py` | Series → forecasting rows (lags + target), safe gap filling, neighbour features. |
| `graphs.py` | Chooses neighbours: correlation, lagged correlation, distance, random (placebo). Distance-restricted variants. |
| `kalman.py` | The headline baseline: per-basin AR(1)+observation-noise filter that separates signal from satellite noise before forecasting. |
| `models.py` | Persistence, damped persistence, ridge (pooled and per-basin), small neural net. |
| `evaluate.py` | Test windows, baseline harness, leakage checks, scoring. |
| `stats.py` | Diebold-Mariano tests (Harvey-corrected, HAC), bootstrap CIs, FDR correction; all pairing goes through `_paired_losses` (row-paired, target-equality asserted). |
| `cache.py` | Content-addressed Kalman-parameter cache (SHA256 over data + fold spec + protocol tag) so a protocol change can never silently reuse stale fits. |
| `surrogates.py` | IAAFT surrogates: scrambled-timing fakes used as the strictest control. |

**Experiment engines** (one per architecture family; each has a `run_phase*.py` runner)

| File | What it does |
|---|---|
| `experiment.py` | Neighbour experiment on ridge (phase 3). |
| `experiment_kalman.py` | Neighbour experiment on the Kalman backbone (phase 3b — the core result). |
| `experiment_nonlinear.py` | GBM/MLP heads on identical features (phase 5). |
| `coupled.py` | Bivariate coupled Kalman with process-noise correlation (phase 5). |
| `fusion.py` | Single-latent fusion filter (phase 5; failed informatively). |
| `era5.py` + `experiment_era5.py` | ERA5-Land ingestion (11 channels, unit conversion, ±10σ winsorization) and ERA5-conditioned arms (phase 6). |
| `experiment_resmlp.py` | Ridge + neighbour-only residual MLP (phase 7). |
| `experiment_lstm.py` + `phase7.py` | LSTM over 12-month windows of filtered state + ERA5 (phase 7). |
| `experiment_gnn.py` | 1-layer GAT (phase 7; never beat ridge — closed the graph question). |
| `experiment_lstm_combined.py` | The stacked winner: kalman + LSTM + neighbour residual-correction MLP (phase 8/8b). |

## `scripts/` — the runners

**Data build:** `inspect_inputs.py`, `download_indices.py`, `download_era5.py`, `build_basin_series.py`, `build_era5_basin_table.py`, `build_li_basin_series.py` (aggregates the Li & Kusche PANGAEA hindcast to our basins).

**Experiments, in phase order:**

| Script | Phase — what it answers |
|---|---|
| `regression_check.py` | 0 — does the new pipeline reproduce the old Africa benchmark? |
| `run_phase2_baselines.py`, `run_kalman_baseline.py` | 2 — baseline ladder; Kalman beats damped everywhere, per-basin ridge at 5 of 6 leads |
| `run_phase3_neighbors.py`, `run_phase3b_kalman_neighbors.py` | 3/3b — do neighbours help? (linear: no — +0.31% ns at lead 1, a controlled null) |
| `run_phase4_surrogates.py`, `run_jump_screen.py` | 4 — controls: surrogates, outlier screen |
| `run_phase5_nonlinear.py`, `run_phase5_coupled.py`, `run_phase5_fusion.py`, `run_phase5_stats.py` | 5 — can architecture enlarge the effect? (no) |
| `run_phase6_era5.py`, `run_phase6_li_comparison.py`, `run_phase6_hybrid.py`, `run_phase6_basin_analysis.py` | 6 — ERA5 forcing, Li & Kusche head-to-head, hybrid splice, who benefits |
| `run_phase7_resmlp.py`, `run_phase7_lstm.py`, `run_phase7_gnn.py` | 7 — three architectures on identical features |
| `run_phase8_lstm_combined.py`, `run_phase8b_merge.py` | 8/8b — stacked model, leads 1–6, final Li comparison |
| `run_resolution_sensitivity.py` | audit — recovers native CSR tiles, builds the per-basin `contamination` metric, area/contamination sweeps + 2×2 (the "small basins was a proxy" result) |
| `run_phase8_stratification.py` | audit — stratifies the stacked correction by contamination/size + formal interaction tests (the anti-leakage result) |
| `build_paper_ladder.py` | paper — baseline ladder recomputed on matched rows (the manuscript's Table 1) |
| `make_figures.py` | paper — figures F1/F2/F5/F8 from results CSVs only, with in-script asserts that plotted values match the ledger |

## `notebooks/`

`01_data_and_sample.ipynb` (sample map, example series), `02_baselines_and_kalman.ipynb` (why the filter wins). Read-only over `results/`; re-running never changes a result. A figures notebook (nb03) is planned — see `paper/FIGURE_PLAN.md`.

## Data, results, paper

| Folder | Contents |
|---|---|
| `data/raw/` | Downloads: climate indices, ERA5-Land, Li & Kusche hindcast (PANGAEA 973113). |
| `data/processed/` | Built tables: `basin_month_twsa_global.csv`, `basin_meta.csv`, `era5_basin_month.csv`, `indices.csv`, `li2026_csr_basin_forecasts.csv`. |
| `results/` | Every experiment output, named by phase. `*_predictions.csv` = every forecast (large); `*_summary.csv` = scores (start here); `*_analysis.md` = what it means; `RUN_LOG.md` = every run + audit notes. |
| `figures/` | Paper figures: F1/F2/F5/F8 built (PDF+PNG, `BUILD_NOTES.md` maps every plotted number to its source); F3/F4/F6/F7 await the supporting-phase reruns. |
| `paper/` | `main.tex` (HESS manuscript, Copernicus class; compiles under MiKTeX, 33 pp), `references.bib`, `FIGURE_PLAN.md` (+ 2026-08-15 addendum), `DECISIONS.md`, `REWRITE_LEDGER.md` (single source of truth: final numbers, framing rules, banned claims, voice rules), `REWRITE_NOTES.md` (rewrite change log). Every number in main.tex carries a `% source:` comment naming its results file. |
| `archive/pre_audit_2026-08-13/` | 117 pre-audit result artifacts + SHA256 manifest, frozen. Never overwrite. |
| `.claude/skills/ml-paper-writing/` | Writing-craft skill (user-supplied) governing all manuscript prose passes. |

Root-level `.nc` files are the two inputs (CSR mascons, basin masks). `Li_2026_*` files are the Li & Kusche paper (reference material). `context_*.md` are project-context docs; `context_africa_jpl.md` / `context_africa_csr.md` are the older Africa-only work.

## The pipeline, start to finish

```
CSR mascons + basin masks
    ↓ build_basin_series.py
basin_month_twsa_global.csv
    ↓ run_phase2_baselines.py + run_kalman_baseline.py
baseline ladder: Kalman beats damped at all leads     (paper contribution 1)
    ↓ run_phase3b + phase4 + phase5 (controls, architectures)
neighbour effect: linear null; stacked correction +0.9-2.0% all leads  (paper contribution 3)
    ↓ build_li_basin_series.py + run_phase6_li_comparison.py
crossing vs Li & Kusche: we win lead 1, they win 3-6   (paper contribution 2)
    ↓ run_phase6_era5.py + phase7 + phase8/8b
stacked model: kalman + LSTM(ERA5) + neighbour correction stage
    ↓ build_paper_ladder.py
paper/main.tex
```

## Two things the code is careful about

**1. No peeking at the future.** Every learned step — deseasonalizing, neighbour choice, model fits, scaling — uses only months before the test period, refitted per fold. `evaluate.py` fails loudly if this breaks.

**2. Fair comparisons.** Real vs placebo graphs get identical features, models, and rows; placebo RNG is seeded per (kind, k) cell so reruns reproduce exactly; stacked-model placebos ride the same-seed stage-1 network they are scored against (seed-matched — a 2026-08-15 audit fix).

**3. Issue-date protocol.** Fold membership is defined on issue dates (model frozen at fold start, used forward); all five engines share one `split_fold`. The pre-audit target-date convention is gone everywhere, including the five runner scripts that once had inline splits.

## Conventions

- Comments explain *why*, not *what*.
- Water storage in cm; results also in standardised units (per-basin std) so big basins don't drown out small ones.
- Fixed seeds; identical numbers on rerun.
- Every phase gets an independent audit pass before the next phase starts; findings live in `results/RUN_LOG.md` and the `*_audit.md` files.
