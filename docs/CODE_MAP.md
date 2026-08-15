# Code Map

A tour of the code, written for someone who has never opened this project.
Read the [README](../README.md) first — it explains what the study is actually about. This page
explains where things live and which file to open when.

Last updated: 2026-08-15.

---

## The one rule that explains the layout

**`src/` thinks. `scripts/` runs. `results/` remembers.**

- **`src/gracefc/`** is a library. It's a pile of functions. Nothing in here does anything on its
  own — you can't run it. It just knows *how* to do things.
- **`scripts/`** are the programs you actually run. Each one imports from `src/`, does one job
  start to finish, and saves what it found as a CSV file in `results/`.
- **`results/`** is where every answer ends up, plus `RUN_LOG.md`, a written diary of every run.

So: if you want to *change how something is computed*, edit `src/`. If you want to *make
something happen*, run a script.

Running anything looks like this:

```bash
.venv/Scripts/python scripts/run_phase2_baselines.py
```

---

## `src/gracefc/` — the library

### Files you'll actually need to understand

| File | What it does |
|---|---|
| `basins.py` | Turns the raw satellite file into a plain table: one row per basin per month, with how much water it held. Also records each basin's centre, area, and continent, and flags the ones we exclude (ice sheets, tiny basins). |
| `decompose.py` | Strips out the long-term trend and the yearly wet/dry cycle, so we only forecast the hard part. Fitted on training months only, so it can't peek ahead. |
| `kalman.py` | **The heart of the project.** Separates the real water level from satellite measurement noise, then forecasts. See README section 5 for how it works. |
| `evaluate.py` | Decides which rows are training and which are testing, and scores forecasts. Refuses to run if it detects the model peeking at the future. |
| `stats.py` | Answers "is this improvement real or luck?" — Diebold–Mariano tests, bootstrap confidence intervals, false-discovery-rate correction. |
| `graphs.py` | Decides which basins count as a basin's "neighbours" — by correlation, by distance, or **randomly** (the placebo version used as a control). |

### Files you'll only need if you go deeper

| File | What it does |
|---|---|
| `features.py` | Reshapes a time series into forecasting rows (past values → the value to predict), handling gaps safely. |
| `models.py` | The simple comparison models: persistence, damped persistence, ridge regression. |
| `surrogates.py` | Makes scrambled fake data (IAAFT) that keeps each basin's own statistics but destroys its timing relationship with other basins. Our strictest control. |
| `cache.py` | Saves fitted Kalman parameters so we don't refit constantly. Keyed by a hash of the data *and* the protocol, so changing the method can never silently reuse old fits. |
| `era5.py` | Loads ERA5 weather data (rain, temperature, soil moisture, and 8 more) and averages it per basin. |

### The experiment engines

Each of these runs one family of models. Each has a matching `run_phase*.py` script that drives
it. You mostly won't edit these unless you're adding a new model.

| File | What it tries |
|---|---|
| `experiment_kalman.py` | Neighbours added to the Kalman baseline. This is the core neighbour result (phase 3b). |
| `experiment.py` | The same question, but on ridge regression instead (phase 3, older). |
| `experiment_nonlinear.py` | Gradient boosting and small neural nets on the same inputs (phase 5). |
| `coupled.py`, `fusion.py` | Two ways of letting a neighbour's data enter the filter directly (phase 5). Both informative failures. |
| `experiment_era5.py` | Weather data added to the Kalman baseline (phase 6). |
| `experiment_resmlp.py` | Ridge for the basin's own history, plus a small network correcting from neighbours only (phase 7). |
| `experiment_lstm.py` + `phase7.py` | A neural network that reads 12 months of history in sequence (phase 7). |
| `experiment_gnn.py` | A graph neural network. Never beat plain ridge — this is what closed the graph question (phase 7). |
| `experiment_lstm_combined.py` | **The winner.** Kalman + LSTM correction + neighbour correction, stacked (phase 8). |

---

## `scripts/` — the programs

### Start here

`run_chain.py` runs every stage in the correct order, checking dependencies as it goes. If you
want to reproduce anything, this is the file. `--list` shows the plan without running it.

### Getting data in

`download_era5.py` and `download_indices.py` fetch raw data. `build_basin_series.py` turns the
satellite file into our main table. `build_era5_basin_table.py` does the same for weather.
`build_li_basin_series.py` converts the published Li & Kusche forecasts onto our basins so we can
compare fairly.

### Utilities

`inspect_inputs.py` prints what's inside the raw satellite files (changes nothing).
`make_manifest.py` checksums the big result files that git doesn't store.
`make_figures.py` builds the paper's charts — and crashes on purpose if a plotted number
disagrees with the recorded value in `paper/notes/REWRITE_LEDGER.md`.

### The experiments, in order

| Script | What question it answers |
|---|---|
| `run_phase2_baselines.py`, `run_kalman_baseline.py` | How good are the simple baselines, and does the Kalman filter beat them? (Yes — at every lead.) |
| `run_phase3b_kalman_neighbors.py` | Do neighbours help, added linearly? (No — +0.31% at lead 1, not significant.) |
| `run_phase4_surrogates.py`, `run_jump_screen.py` | Sanity checks: does the result survive scrambled data, and is it driven by a few outlier months? |
| `run_phase5_*.py` | Can a fancier model architecture make the neighbour effect bigger? (No.) |
| `run_phase6_era5.py` | Does weather data help? (Yes — biggest single gain at lead 1.) |
| `run_phase6_li_comparison.py` | How do we compare to a published forecast product? (We win lead 1, they win leads 3–6.) |
| `run_phase6_basin_analysis.py` | *Which* basins benefit, and why? |
| `run_phase7_*.py` | Three neural architectures on identical inputs, head to head. |
| `run_phase8_lstm_combined.py`, `run_phase8b_merge.py` | The stacked winner, across all six leads. |
| `run_r0_ablation.py` | Which half of the Kalman filter actually earns the win? (The noise removal.) |
| `run_resolution_sensitivity.py` | Are results contaminated by the satellite's coarse resolution? Builds the leakage metric. |
| `run_phase8_stratification.py` | Is the neighbour result just leakage in disguise? (No — it's positive everywhere.) |
| `build_paper_ladder.py` | Recomputes the paper's main comparison table on exactly matched rows. |

---

## Everything else

| Folder | What's in it |
|---|---|
| `data/raw/` | Downloaded files: weather, climate indices, the Li & Kusche forecasts. |
| `data/processed/` | The clean tables everything else reads, mainly `basin_month_twsa_global.csv`. |
| `results/` | Every output. `*_summary.csv` = the scores (**start here**); `*_predictions.csv` = every individual forecast (large); `*_analysis.md` = what it means; `RUN_LOG.md` = the diary. |
| `figures/` | The paper's charts, plus `BUILD_NOTES.md` tracing every plotted number to its source file. |
| `paper/` | `main.tex` is the manuscript. `paper/notes/` holds the drafting record — most importantly `REWRITE_LEDGER.md`, the only authoritative list of the paper's numbers. |
| `archive/` | A frozen snapshot of results from before the 2026-08-13 audit, checksummed. Never overwrite it — it's how we prove what changed. |
| `notebooks/` | Two notebooks for interactive poking. They only read results; running them can't change anything. Outputs are cleared on purpose. |
| `docs/reference/` | The Li & Kusche paper, compressed, for the head-to-head comparison. |
| `docs/history/` | The older Africa-only pilot study that preceded this global one. |

The two `.nc` files at the repository root are the raw satellite measurements and the basin
boundaries.

---

## The whole pipeline in one picture

```
raw satellite file + basin boundaries
    ↓  build_basin_series.py
one table: water storage per basin per month
    ↓  run_phase2_baselines.py, run_kalman_baseline.py
FINDING 1: the Kalman filter beats the field's standard baseline
    ↓  run_phase3b, phase4, phase5   (neighbours + all the controls)
FINDING 3: neighbours help, but only as a correction stage
    ↓  build_li_basin_series.py, run_phase6_li_comparison.py
FINDING 2: we win at lead 1, the published product wins at leads 3-6
    ↓  run_phase6_era5.py, phase7, phase8
the stacked model: Kalman + LSTM + neighbour correction
    ↓  build_paper_ladder.py, make_figures.py
paper/main.tex
```

---

## Three things the code is deliberately careful about

**1. It never peeks at the future.** Every fitted step — removing the trend, choosing neighbours,
training models, scaling numbers — uses only data from before the test period, refitted
separately for each fold. `evaluate.py` raises an error if this is violated. This matters because
an earlier version got it wrong and made results look better than they were.

**2. Comparisons are genuinely fair.** When we test real neighbours against random ones, both get
identical features, identical models, identical rows, and — this was a bug we fixed on
2026-08-15 — the *same random seed*, so the only difference is which basins are connected.

**3. Reruns give identical numbers.** Seeds are fixed everywhere. If you rerun a phase and get
different numbers, something is wrong; don't shrug it off.

## House style

- Comments explain **why**, not what. If the code says `x += 1`, don't write "add one to x."
- Water storage is in centimetres; results are also reported in standardized units so a huge
  basin doesn't drown out a small one.
- Every number in the manuscript carries a `% source:` comment naming the results file it
  came from.
- Every phase gets an independent audit pass before the next one starts. Findings go in
  `results/RUN_LOG.md`.
