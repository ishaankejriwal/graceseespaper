# Code Map

A tour of the code, written for someone who has never opened this project.
Read the [README](../README.md) first — it explains what the study is actually about. This page
explains where things live and which file to open when.

Last updated: 2026-09-10.

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
| `experiment_flat12.py` | Builds the flat 12-month window of filtered state and ERA5, and fits the ridge correction on it. The engine behind the study's strongest own-basin model. |
| `comparison.py` | Puts the published Li and Kusche forecasts into our target space and builds the subset rules, including the strict `joint_full_cells` rule. |
| `runtime.py` | Resolves every input and output path for the selected mascon product, so the same code scores CSR or JPL without hardcoded directories. |

### Files you'll only need if you go deeper

| File | What it does |
|---|---|
| `features.py` | Reshapes a time series into forecasting rows (past values → the value to predict), handling gaps safely. |
| `models.py` | The simple comparison models: persistence, damped persistence, ridge regression. |
| `surrogates.py` | Makes scrambled fake data (IAAFT) that keeps each basin's own statistics but destroys its timing relationship with other basins. The strictest control for a cross-basin claim. Extended steps only. |
| `cache.py` | Saves fitted Kalman parameters so we don't refit constantly. Keyed by a hash of the data *and* the protocol, so changing the method can never silently reuse old fits. |
| `era5.py` | Loads ERA5 weather data (rain, temperature, soil moisture, and 8 more) and averages it per basin. |
| `kalman_mission.py` | The mission-split variant of the filter: a separate observation-noise variance for GRACE-FO. Tested and rejected, kept as a sensitivity. |
| `graphs.py` | Decides which basins count as a basin's "neighbours", by correlation, by distance, or **randomly** (the placebo version used as a control). Extended steps only. |

### The experiment engines

Each of these runs one family of models, and each has a matching `run_*.py` script that drives
it. You mostly won't edit these unless you're adding a new model. Only the first row is reached
by the default chain; everything marked *extended* is reached with `--extended` or `--steps`.

| File | What it tries |
|---|---|
| `experiment_flat12.py` | Kalman forecast plus a ridge correction over a flat 12-month window of filtered state and ERA5. The strongest own-basin model in the study. |
| `experiment_kalman.py` | *Extended.* Neighbours added to the Kalman baseline (phase 3b), with the seed-matched placebo graphs. |
| `experiment.py` | *Extended.* The same question on ridge regression instead (phase 3, older). |
| `experiment_nonlinear.py` | *Extended.* Gradient boosting and small neural nets on the same inputs (phase 5). |
| `coupled.py`, `fusion.py` | *Extended.* Two ways of letting a neighbour's data enter the filter directly (phase 5). Both informative failures. |
| `experiment_era5.py` | *Extended.* Weather data added to the Kalman baseline (phase 6). |
| `experiment_resmlp.py` | *Extended.* Ridge for the basin's own history, plus a small network correcting from neighbours only (phase 7). |
| `experiment_lstm.py` + `phase7.py` | *Extended.* A neural network that reads 12 months of history in sequence (phase 7). It imports the window construction from `experiment_flat12.py`, so the two engines cannot drift apart. |
| `experiment_gnn.py` | *Extended.* A graph neural network. Never beat plain ridge (phase 7). |
| `experiment_lstm_combined.py` | *Extended.* The stacked system: Kalman plus LSTM correction plus neighbour correction (phase 8). Loses to `experiment_flat12.py` once history length is equalized. |

---

## `scripts/` — the programs

### Start here

`run_chain.py` runs every stage in the correct order, checking dependencies as it goes, from
building the processed tables out of the raw downloads all the way to figures and the checksum
manifest. Only the downloads themselves (network and credentials) stay manual. If you want to
reproduce anything, this is the file. `--list` shows the plan without running it, and marks each
extended step `[extended]`.

It keeps two lists. The **default** is the 13 steps behind the paper's claims:
`build_basin`, `build_era5`, `build_li`, `phase2`, `kalman`, `flat12_ridge`, `kalman_mission`,
`r0_ablation`, `ladder`, `li_comparison`, `conventional_metrics`, `figures`, `manifest`. The
**extended** list is everything else, run with `--extended` or `--steps`. The default list needs
no torch. One rough edge: `figures` is in the default list but still reads extended outputs
(`phase3b_*`, `phase5_*`, `phase6_era5_*`, `phase8b_*`), because the manuscript figures have not
been rebuilt for the reframe, so a default-only machine stops there with the missing files
named.

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

The default list first, in the order the chain runs them.

| Script | What question it answers |
|---|---|
| `run_phase2_baselines.py` | How good are the simple baselines: climatology, persistence, damped persistence, three ridges? |
| `run_kalman_baseline.py` | Does the Kalman filter beat them? (Yes, at every lead.) |
| `run_flat12_ridge.py` | Does a ridge over a flat 12-month window of filtered state and ERA5 improve on the filter? (Yes, most at lead 1.) No torch. |
| `run_kalman_mission_sensitivity.py` | Does a separate GRACE-FO observation-noise variance help? (No, it hurts at every lead.) |
| `run_r0_ablation.py` | Which half of the Kalman filter earns the win? (The noise removal.) |
| `build_paper_ladder.py` | Recomputes the paper's main comparison table on exactly matched rows. |
| `run_phase6_li_comparison.py` | How do we compare to a published forecast product, under one subset rule? (We win lead 1, they win the long leads.) |
| `compute_conventional_metrics.py` | Restates the retained systems in the literature's own metrics (per-basin RMSE in cm, CC, NSE; anomaly and full signal). |

The extended list, which is every remaining experiment.

| Script | What question it answers |
|---|---|
| `run_phase3b_kalman_neighbors.py` | Do neighbours help, added to the filter? Also drives the `predlag` and `conditioned` variants and the placebo graphs. |
| `run_phase4_surrogates.py`, `run_jump_screen.py` | Does a cross-basin result survive scrambled data, and is it driven by a few outlier months? |
| `run_phase5_*.py` | Can a fancier architecture make the neighbour effect bigger? (No.) |
| `run_phase6_era5.py`, `run_phase6_era5_attribution.py` | Does weather data help, and which variables carry it? |
| `run_phase6_basin_analysis.py` | *Which* basins benefit, and why? |
| `run_phase6_hybrid.py` | Splices our forecasts with the published product. |
| `run_phase7_*.py` | Three neural architectures on identical inputs, head to head. |
| `run_phase8_lstm_combined.py`, `run_phase8b_merge.py` | The stacked system and its neighbour correction, across all six leads. |
| `run_resolution_sensitivity.py` | Are results contaminated by the satellite's coarse resolution? Builds the leakage metric and the official tile geometry that the strict Li subset rule reuses. |
| `run_phase8_stratification.py` | Was the neighbour result leakage in disguise? |
| `run_flat12_train85_sensitivity.py` | Does the flat ridge still beat the LSTM when both get the same training window? (Yes.) |

---

## Everything else

| Folder | What's in it |
|---|---|
| `data/raw/` | Downloaded files: weather, climate indices, the Li & Kusche forecasts. |
| `data/processed/` | The clean tables everything else reads, mainly `basin_month_twsa_global.csv`. |
| `results/` | Every output. `*_headline.csv` and `*_summary.csv` = the scores (**start here**, and see `results/README.md`); `*_predictions.csv` = every individual forecast (large); `RUN_LOG.md` = the diary. `results/jpl/` holds the compact JPL tables from the collaborator's run. |
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

The default chain, in order:

```
raw satellite file + basin boundaries
    |  build_basin_series.py
one table: water storage per basin per month  (+ build_era5, build_li)
    |  run_phase2_baselines.py
the reference ladder: climatology, persistence, damped persistence, three ridges
    |  run_kalman_baseline.py
CLAIM 1: the Kalman filter is the reference forecast these should be scored against
    |  run_flat12_ridge.py
CLAIM 2: filter + ridge over a flat 12-month window of filtered state and ERA5 is the
         strongest own-basin model
    |  run_kalman_mission_sensitivity.py, run_r0_ablation.py
the two sensitivities: the mission split loses, the noise removal is what earns the win
    |  build_paper_ladder.py, run_phase6_li_comparison.py
CLAIM 3: both models against the published product, CSR and JPL, one subset rule
    |  compute_conventional_metrics.py, make_figures.py, make_manifest.py
paper/main.tex
```

The extended chain hangs off the same processed tables and answers the questions that did not
become claims: neighbours, the sequence models, the resolution work, the hybrid splice.

---

## Three things the code is deliberately careful about

**1. It never peeks at the future.** Every fitted step — removing the trend, choosing neighbours,
training models, scaling numbers — uses only data from before the test period, refitted
separately for each fold. `evaluate.py` raises an error if this is violated. This matters because
an earlier version got it wrong and made results look better than they were.

**2. Comparisons are genuinely fair.** Every pairwise number is computed on exactly matched
rows, which is why `build_paper_ladder.py` exists: the per-model summary files lose rows as the
lead grows, so their RMSEs are not cross-comparable. In the extended neighbour steps, a real
graph and a random one get identical features, identical models, identical rows, and, after a
bug fixed on 2026-08-15, the *same random seed*, so the only difference is which basins are
connected.

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
