# GRACE TWSA Forecasting Study

**In one sentence:** we try to predict how much water is stored in the world's river basins a
few months from now, and we argue that the honest starting point for that job is a filter which
separates the satellite's signal from its noise, not the persistence baseline the field usually
scores against.

---

## 1. What are we even measuring?

There are two satellites called **GRACE** (and its successor GRACE-FO) that fly in formation
around the Earth. They measure gravity. When a region has more water in it — in rivers, lakes,
soil, snow, and underground — that region is slightly heavier, and the satellites detect it.

So GRACE gives us, once per month, a map of how much water each part of the world is holding.

The number we work with is called **TWSA — Terrestrial Water Storage Anomaly**:

- *Terrestrial water storage* = all the water in a place (surface + soil + groundwater).
- *Anomaly* = how far that is from normal, not the absolute amount. So a positive number means
  "wetter than usual here," negative means "drier than usual."

We chop the world into **234 river basins** (a basin is the area that drains into one river
system — think "everything that flows into the Amazon"). For each basin we have one TWSA number
per month, from April 2002 to May 2026.

**Our job:** given every measurement up to today, predict the TWSA number 1, 2, 3, 4, 5, or 6
months from now. The number of months ahead is called the **lead**. Lead 1 is next month, lead 6
is half a year out. Longer leads are harder.

### Two things we remove before forecasting

Raw water storage is dominated by two boring, already-known patterns:

1. **A long-term trend** — some places are steadily drying out or filling up over decades.
2. **The seasonal cycle** — every basin has a wet season and a dry season, every year.

Anyone can predict those. If we left them in, our model would look great while actually
predicting nothing interesting. So we subtract both first, and forecast only what's left over.
This is called **deseasonalizing**. What remains is the genuinely unpredictable part — and it's
the only honest thing to be judged on.

---

## 2. What did we find?

Two claims about our own forecasts, and one head-to-head against somebody else's.

### Claim 1: the reference forecast should be a filter, not persistence

To say your forecast is *good*, you have to beat something. Almost every paper in this field
compares against **damped persistence**, which is a fancy name for a simple idea:

> "Next month will look like this month, but a bit closer to normal."

You take the most recent measurement and shrink it toward zero. That is it. It works
surprisingly well, because water storage changes slowly.

Here is the flaw. The satellite measurement is noisy. It is not the true water level, it is the
true water level plus measurement error. Damped persistence shrinks the *measurement*, so it
carries the noise forward along with the signal. It also has nothing to say about a month with
no measurement, and the gap between the GRACE and GRACE-FO missions in 2017 and 2018 is exactly
that.

We propose a per-basin **Kalman filter** (explained properly in section 5) as the reference
instead. It separates the real underlying water level from the measurement noise first, then
shrinks the clean estimate. A missing month costs it nothing: the filter propagates its state
forward until the next observation arrives.

That change alone buys **+5.0% at lead 1 and +8.8% at lead 2**, and +2.5% to +5.6% at leads 3
to 6, against the stronger damped variant at each lead (`results/paper_baseline_ladder.csv`).
The Diebold-Mariano p-values do not run in order of lead: they are smallest at leads 2 and 3
(6.3e-19 and 5.7e-19), 3.5e-12 at lead 1, and largest at lead 5 (7.9e-7), same file.

It also beats a per-basin ridge regression at all six leads, by
+2.0/+3.0/+1.4/+1.0/+2.2/+3.9%, and the margin is significant at five of them
(`results/paper_baseline_contrasts.csv`, `kalman_ar1` against `ridge_own_perbasin`). Lead 4 is
the one that misses significance: it is still a gain, +1.04%, but at p = 0.078.

We also know *which half* of the filter does the work. Switch the noise removal off and keep
everything else, and the filter gives up 5.4% at lead 1, rising to a peak of 12.7% at lead 5
and easing back to 12.3% at lead 6 (`results/r0_ablation_summary.csv`). That stripped version
is *worse than plain damped persistence* at leads 2 to 6 (-0.9% to -11.6%, same file). The
filtering is the win, not better drift estimation.

### Claim 2: at leads 1 to 4 the strongest own-basin model is the filter plus a ridge correction

Take the filtered state and the eleven ERA5 weather variables for the last 12 months, flatten
that window into one long row of numbers, and fit a ridge regression to predict how wrong the
Kalman forecast is about to be. That is `ridge_own_era5_flat12`, the strongest own-basin model
in the study at leads 1 to 4: **+12.2% over damped persistence at lead 1**, then
+13.7/+9.6/+6.2% at leads 2 to 4, and +4.6/+4.5% at leads 5 and 6
(`results/paper_baseline_ladder.csv`).

At leads 5 and 6 the own-state ridge correction `kalman_own_ridge` is marginally better.
`ridge_own_era5_flat12` scores -0.58% against it at lead 5 and -0.73% at lead 6
(`results/flat12_ridge_summary.csv`, column `skill_vs_ridge_own`), which in RMSE is 1.3804
against 1.3764 at lead 5 and 1.4421 against 1.4369 at lead 6
(`results/paper_baseline_ladder.csv`). The ERA5 window stops paying for itself at the long
leads.

"Strongest" here means lowest pooled MSE in standardized units, where every basin counts
equally. In raw centimetres, where the largest basins dominate, the ranking changes: the plain
flat-12 ridge without ERA5, `ridge_own_flat12`, has the lowest pooled RMSE at leads 1 to 3
(5.118 cm at lead 1 against 5.128 for the filter and 5.231 for the ERA5 ridge), the own-state
ridge correction is lowest among retained models at leads 4 to 6, and the ERA5 ridge sits behind
the plain filter at every lead (`results/conventional_metrics_summary.csv`, column
`pooled_rmse_cm`). The ERA5 gain is therefore concentrated in basins with small storage
variance, and the paper has to state which weighting a ranking refers to. Measured against the Kalman reference itself, `ridge_own_era5_flat12` adds +7.6% at
lead 1, decaying to +0.9% and not significant by lead 6
(`results/paper_baseline_contrasts.csv`).

The interesting part is what it beats. Every sequence model we trained loses to it once the two
are given the same amount of history: the LSTM arms, the residual MLP and the ridge twins all
sit behind it at lead 1 (`results/phase7_lstm_summary.csv`), and with the training window
equalized the flat ridge is ahead of the two-seed LSTM ensemble by +1.0/+2.8/+2.3% at leads 1 to
3 (`results/flat12_train85_sensitivity.csv`). A flat window and a linear fit are enough.

### Claim 3: against a published product we win at short leads and lose at long ones

We compare both of our models against Li and Kusche's published GRACE-FCast product on the CSR
mascons, and the Kalman reference alone on the JPL mascons, under one protocol and one strict
subset rule. The JPL run predates the flat-12 step, so `ridge_own_era5_flat12` has no JPL rows
yet (`results/jpl/phase6_li_comparison_headline.csv`); the JPL rerun is pending. The rule
(`joint_full_cells`) keeps a basin only if it fully contains at least one native mascon of the
product being scored and at least one valid 1-degree Li cell. That leaves 209 of 227 basins on
CSR and 67 on JPL (`results/phase6_li_comparison_summary.csv`,
`results/jpl/phase6_li_comparison_summary.csv`). The 227 is the Li-matched population, the
`all_matched` subset in that file, and not the 234 basins the study keeps overall: seven of the
234 have no usable Li forecast and never enter this comparison.

On CSR, over 209 basins and 60 months (`results/phase6_li_comparison_headline.csv`):

- the Kalman reference beats Li's full product by **+16.4% at lead 1** (p = 2.2e-3), ties at
  lead 2 (-6.0%, p = 0.23), and loses from lead 3 on (-16.3% to -35.4%);
- `ridge_own_era5_flat12` beats Li's non-seasonal product by **+32.8% at lead 1** and **+11.2%
  at lead 2** (p = 2.4e-10 and 2.1e-3), ties at lead 3 (-0.2%, p = 0.94), and loses from lead 4
  on.

Basin by basin it is a close split that tips over as the lead grows: Li's full product has the
lower error in 98 of the 209 basins against the Kalman reference at lead 1 and in 157 of 209 at
lead 6, and in 87 of 209 against `ridge_own_era5_flat12` at lead 1 and 162 of 209 at lead 6
(`results/phase6_li_comparison_perbasin.csv`, column `a_better`, scored on the
`joint_full_cells` basins).

On JPL, over 67 basins and 59 months, on the collaborator's run
(`results/jpl/phase6_li_comparison_headline.csv`, reported Li-first, so a negative skill means
we are ahead): Li's non-seasonal product scores -1.12 against the Kalman reference at lead 1
(p = 1.3e-13) and -0.29 at lead 2 (p = 2.1e-5), draws at lead 3 (-0.07, p = 0.15), and is ahead
at leads 5 and 6.

The shape is the same on both products. Next month is mostly determined by where the water
already is, so a clean read on the current state matters most; six months out the current state
has washed out and what matters is what the weather is going to do.

### What we tested and dropped

Three things were run properly and are not claims.

- **Neighbouring-basin information.** It does not replicate on the JPL mascons, which is why it
  is not a claim. Every neighbour variant there is worse than own-basin at every lead, and the
  real graph beats 0 of 50 seed-matched placebo graphs for the correlation-selected neighbour at
  every lead and for every arm at leads 1 to 3; the distance-selected arm is the exception and
  beats at most 26 of 50, at lead 5. IAAFT surrogates are 0 of 99 at every lead
  (`results/jpl/phase3b_summary.csv`, `results/jpl/phase4_surrogate_summary.csv`). What it was
  on CSR: a neighbour's propagated state, used as a correction, worth +0.31% at lead 1 over the
  own-basin ridge, beating 50 of 50 seed-matched random graphs and 99 of 99 IAAFT surrogates
  (`results/phase3b_summary.csv`, `results/phase4_surrogate_summary.csv`). Working
  interpretation: JPL's 3-degree mascons with the CRI filter already do the spatial denoising
  that a CSR neighbour was supplying. A controlled effect that reverses on a second mascon
  solution of the same observations is not one we are willing to publish. The experiments are
  kept as extended chain steps.
- **A separate observation-noise variance for GRACE-FO.** It hurts at every lead, from -1.84% at
  lead 1 (p = 1.7e-7) to -0.38% at lead 6 (`results/kalman_mission_summary.csv`), and fold 1
  cannot fit it at all (234 of 1170 basin-folds fall back to one variance). The single-variance
  filter is what we keep.
- **Sequence models.** LSTM, residual MLP, graph network and the stacked combination all lose to
  the flat 12-month ridge once history is equalized, as described in claim 2.

Details and dates: the JPL non-replication is tabulated in the `results/RUN_LOG.md` entry
"JPL neighbor experiments: non-replication record (from results/jpl/, collaborator run
2026-09-05/06)"; the CSR neighbour runs it is contrasted with are under the 2026-08-12 to
2026-08-17 entries; the mission-split and sequence-model decisions are under the 2026-09-10
"Kalman-benchmark reframe" entry.

---

## 3. Repository map

| Folder | What's in it | Do I need to touch it? |
|---|---|---|
| `src/gracefc/` | The engine. All the real logic: loading data, models, statistics. | Yes, if changing methods |
| `scripts/` | One runnable file per stage of the pipeline. These are the entry points. | Yes, to run things |
| `tests/` | Automated checks. Run them after any change. | Run, rarely edit |
| `results/` | Every output table, plus `RUN_LOG.md`, the diary of what was run when. | Read only |
| `figures/` | The charts that go in the paper. | Generated, don't hand-edit |
| `paper/` | The manuscript itself (`main.tex`) and its drafting notes in `paper/notes/`. | Yes, if writing |
| `docs/` | Longer explanations: code map, project status, past audits, reference papers. The one past audit kept in full is [`docs/AUDIT_2026-08-13.md`](docs/AUDIT_2026-08-13.md), historical: it is the audit the `archive/pre_audit_2026-08-13/` freeze is named for, and its findings were fixed at the time. | Read |
| `notebooks/` | Two Jupyter notebooks for poking at the data interactively. | Optional |
| `data/`, `archive/` | Raw inputs and frozen old results. Not stored in git — see section 4. | Download once |

**If you're new, read in this order:** this README → [`docs/STUDY_CONTEXT.md`](docs/STUDY_CONTEXT.md)
(the claims, the numbers behind them, and what's still open) →
[`docs/CODE_MAP.md`](docs/CODE_MAP.md) (what each file does) →
[`results/README.md`](results/README.md) (which result file to open) →
[`results/RUN_LOG.md`](results/RUN_LOG.md) (the diary).

---

## 4. Getting set up

You need Python 3.11 or newer. Everything runs on a normal laptop CPU — no GPU needed.

**Install:**

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-lock.txt
```

**Check it worked** by running the test suite. This takes about 30 seconds and needs no data:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

You should see `28 passed` (on a fresh clone without the data downloaded yet, some
data-dependent tests skip instead, and passed-plus-skipped is also fine).

### Downloading the data

The data files are big, so they aren't in git. You download them once:

| What | Where it goes | How to get it |
|---|---|---|
| GRACE satellite measurements | repo root, `CSR_..._Mascons_....nc` | [CSR mascon page](http://www2.csr.utexas.edu/grace/RL0603_mascons.html) |
| CSR ancillary files (mascon mapping + land mask) | `data/raw/csr_ancillary/` | same [CSR mascon page](http://www2.csr.utexas.edu/grace/RL0603_mascons.html), "ancillary files" section |
| River basin boundaries | repo root, `HydroShed+Mascon_Basins_L3.nc` | ships with the project |
| ERA5 weather data | `data/raw/era5/` | run `scripts/download_era5.py` (needs a free [CDS account](https://cds.climate.copernicus.eu/)) |
| Climate indices | `data/processed/indices.csv` | run `scripts/download_indices.py` (NOAA PSL, no account needed) |
| GRACE-FCast hindcast (Li & Kusche 2026) | `data/raw/li2026/CSR-FCast/global_gridded/` | [PANGAEA 973113](https://doi.pangaea.de/10.1594/PANGAEA.973113) |

Curious what's inside those satellite files? `scripts/inspect_inputs.py` prints their structure
and doesn't change anything.

---

## 5. How the Kalman filter works

This is the heart of the project, so it's worth understanding properly. It's genuinely simple.

**The problem.** Imagine a bathroom scale that's a bit unreliable. Your real weight changes
slowly and smoothly. But each time you step on, the scale reads a couple of pounds off in a
random direction. If you want to predict tomorrow's weight, you should *not* just take today's
reading — that reading includes today's random error.

GRACE is that scale. The satellite measurement is the real water level plus noise.

**The model.** We assume two things about each basin:

1. The real water level drifts back toward normal each month, at a rate we call **ρ** (rho).
   If ρ is 0.9, the basin keeps 90% of its anomaly month to month — slow, sticky. If ρ is 0.3,
   it snaps back fast.
2. What we *observe* is that real level, plus measurement noise.

**The trick.** Each month, the filter has a prediction of where the water level should be
(based on last month, decayed by ρ). Then the new measurement arrives and disagrees with it.
The filter has to decide who to believe.

It splits the difference using a number called the **Kalman gain**:

```
new estimate  =  prediction  +  gain × (measurement − prediction)
```

- If gain = 1, it fully trusts the measurement and ignores its own prediction.
- If gain = 0, it ignores the measurement entirely and coasts on the model.
- In practice it lands in between, and the exact value comes from how noisy this particular
  basin's measurements have been historically.

That's the whole thing. Three numbers are estimated per basin from training data: **ρ** (how
sticky), **q** (how much the real level genuinely jumps around), and **r** (how noisy the
measurements are). The gain follows from q and r.

**Forecasting** is then trivial: take the clean current estimate and decay it by ρ once per
month ahead. Six months out, multiply by ρ⁶.

**Why this beats damped persistence:** damped persistence is mathematically the *same thing with
gain forced to 1* — full trust in the noisy measurement, no filtering. That's precisely the
ablation we ran, and it loses. The filtering is the win.

**One honest caveat.** We call `r` "measurement noise," but strictly we can't prove that's what
it is — it's whatever part of the signal the model can't carry forward. 259 of 1170 basin-fits
land at r ≈ 0 (`results/kalman_mission_summary.csv`, diagnostic row `n_r_one_at_boundary`).
The paper is careful about this wording, and you should be too.

### How the correction stage uses the filter

Every own-basin model that beats the filter is a correction on top of it, not a replacement for
it. Take `ridge_own_era5_flat12`, the strongest of them at leads 1 to 4. Its prediction is two
things added together:

```
prediction  =  kalman forecast  +  ridge correction
```

- The **Kalman forecast** is the baseline guess described above.
- The **ridge correction** reads a flat 12-month window of the basin's own *filtered* state plus
  its 11 ERA5 weather variables, and predicts *how wrong the Kalman guess will be*.

Every model in this study that works is shaped like "good baseline plus small learned
correction". Models that tried to predict water storage from scratch, ignoring the filter, did
worse. The sequence models (LSTM, residual MLP, graph network, and the stacked combination) are
all still in the repository as extended chain steps, and none of them beats this flat ridge over
the same history.

---

## 6. How we avoid fooling ourselves

This section matters more than it sounds. An earlier audit of this project found real bugs that
made results look better than they were, so the conventions below are load-bearing.

### Training and testing are split by *when the forecast was made*

The cardinal sin in forecasting is letting the model peek at the future. We split data into
5 "folds" — 5 separate train/test rounds, each testing on a later stretch of time.

Two dates matter for every row of data:

- the **issue date** — when the forecast was made;
- the **target date** — the month being predicted.

For a fold that freezes the model at some date `T`:

- a row is in **test** if its *issue date* is on or after `T`;
- a row is in **train** only if its *target* was already observed before `T`.

Rows that fall between (issued before `T`, but predicting a month after `T`) are thrown away
entirely. That gap is deliberate and correct.

Why it matters: an earlier version split on target dates instead. That meant a 6-month-ahead
forecast could use data-processing steps fitted on measurements from five months *after* it was
supposedly issued. It made results look better than reality. Don't undo this. It's pinned by
`tests/test_regressions.py::test_split_fold_membership_invariants`.

Same principle everywhere else: the trend, the seasonal cycle, and the standardization are all
computed using training data only, never the full record.

### Is the improvement real, or luck?

Every headline number comes with a statistical test:

- **Diebold–Mariano test** — the standard test for "is forecast A genuinely better than forecast
  B, or is this within noise?" It's adjusted for the fact that overlapping forecasts are
  correlated with each other.
- **Bootstrap confidence intervals** — resample the data thousands of times in contiguous
  blocks, see how much the answer wobbles.
- **False discovery rate (FDR) correction** — when you test 234 basins separately, some will
  look significant by pure chance. This corrects for that.

### Controls that the default chain does not run

The default chain compares model against model on identical rows, with the three tests above.
It runs no placebo graphs and no surrogates, because none of the claims it supports involves a
graph, and there is nothing to randomize.

The machinery is still in the repository and still runs, as extended steps. `graphs.py` builds
seed-matched random-neighbour graphs (same number of connections per basin, same model seed,
only the wiring differs) and `run_phase4_surrogates.py` builds IAAFT surrogates, which keep each
basin's own statistical character but destroy its timing relationship with every other basin.
Those two controls are what the dropped neighbour claim was tested against, in both directions:
it passed them on CSR and failed them on JPL. See "What we tested and dropped" in section 2.

---

## 7. Running the pipeline

`scripts/run_chain.py` runs the stages in the right order. It checks each stage's inputs exist
*before* starting it, confirms the outputs were actually written afterwards, and stops
immediately on failure. (It exists because an older version let a crashed stage slip by
unnoticed and everything downstream silently used stale files.)

See what it will do, without running anything:

```bash
.venv/Scripts/python scripts/run_chain.py --list
```

Run the default list:

```bash
.venv/Scripts/python scripts/run_chain.py
```

### The default list and the extended list

There are two lists. The **default** is the 12 steps that produce the paper's claims, in this
order:

```
build_basin  build_era5  build_li  phase2  kalman  flat12_ridge  kalman_mission
r0_ablation  ladder  li_comparison  conventional_metrics  manifest
```

It ends at `manifest`. `figures` used to be the second-to-last default step and moved to the
extended list on 2026-09-10, for the reason two paragraphs down.

Everything else is **extended**: every neighbour-only experiment, every torch model, the
resolution and stratification work, the hybrid splice. Nothing was deleted and every extended
step is still registered, still runnable, and still declares its inputs and outputs to `--list`.
Reach them with `--extended`, which runs the extended list and only the extended list rather
than the default list followed by it, or by naming steps with `--steps name1 name2`. If you
want both lists, run the chain twice, default first.

The default list needs no torch.

Wall times for the default tail, measured on the 2026-09-10 CSR rerun on this machine
(`results/RUN_LOG.md`, "Kalman-benchmark reframe"): `build_li` 5.9 min, `flat12_ridge` 2.0 min,
`kalman_mission` 29 min, `ladder` 1.0 min, `li_comparison` 6.0 min, `conventional_metrics`
0.7 min, `manifest` 0.2 min. Four steps were not re-timed in that run: `build_basin`,
`build_era5`, `phase2` and `kalman`. `r0_ablation` was not re-timed either, and it is not ahead
of the timed tail: it runs after `kalman_mission` and before `ladder`, and the 2026-08-16 chain
put it at 4.4 min. The extended list is where the long runtimes live. The 14-step extended
rerun of 2026-08-16 took about 30.5 hours of wall clock, while its fourteen recorded per-step
times sum to about 33.6 hours; both figures are in the `results/RUN_LOG.md` entry "corrected
rerun chain COMPLETE", and the neural stages dominate either one.

**Where `figures` went.** It is an extended step now, because it reads outputs only extended
steps produce. It declares sixteen inputs: the basin mask, `basin_meta.csv`, and fourteen
results files. Two of the fourteen come from the default list, `paper_baseline_ladder.csv` and
`paper_baseline_contrasts.csv`. The other twelve exist only after extended steps have run, and
they are `phase8b_li_comparison_headline.csv`, `phase8b_li_comparison_perbasin.csv`,
`phase8b_h16_ensemble_headline.csv`, `phase8b_h16_headline.csv`, `phase8_stratification.csv`,
`phase3b_summary.csv`, `phase3b_placebo_monthly.csv`, `phase4_surrogate_summary.csv`,
`phase5_perbasin_fdr_h1.csv`, `phase6_era5_headline.csv`, `phase6_era5_predictions.csv` and
`phase4_conditioned_predictions.csv`. `run_chain.py --list` prints the same list.

`scripts/make_figures.py` also opens four files at runtime that the step does not declare:
`phase8_lstm_combined_predictions.csv`, `phase8b_lstm_h46_predictions.csv`,
`phase8_lstm_combined_placebo_monthly.csv` and `phase8b_lstm_h46_placebo_monthly.csv`, read at
around lines 375 and 392. Those are extended outputs too, so the dependency check will not
report them missing and the script fails on the open instead. A machine that has only ever run
the default list has none of the sixteen files, twelve declared and four not. The manuscript
figures have not been redone for the reframe; until they are, run the extended steps first, or
name `figures` in `--steps` once
its inputs exist. `--extended --source jpl` skips `figures` rather than failing on it, because
the figure asserts pin archived CSR numbers.

Each stage writes its own log to `results/chain_<name>.log`.

Run the same experiment suite with the JPL RL06.3Mv04 mascons:

```powershell
.venv\Scripts\python.exe scripts\run_chain.py --source jpl --list
.venv\Scripts\python.exe scripts\run_chain.py --source jpl
```

Place the JPL file used by the historical pilot at
`data/raw/GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc`. The JPL target table is
written to `data/processed/jpl/`, and every derived result and cache is written to
`results/jpl/`; the archived CSR files are not overwritten. The JPL reader maps the
0.25-degree basin-mask cells to JPL's 0.5-degree sampled grid and uses the product's official
missing-month metadata. If the selected JPL file contains the optional `scale_factor` field
(as the recommended CRI product does), it is applied and recorded in `basin_meta.csv`; the
expert non-CRI product has no scale factors and is used as distributed. For an explicitly
unscaled sensitivity run, add `--no-scale-factors` to the `run_chain.py --source jpl`
command.

The June 2026 CRI product leaves six small-island masks with no finite scaled JPL cells
(`Nusa_Tenggara`, `Maluku`, `Halmahera_Islands`, `Solomon_Islands`, `Jamaica`, and
`Puerto_Rico`). The JPL build records these as `jpl_unavailable` in `basin_meta.csv` and
uses the remaining 228 hydrology basins. This check is computed from the selected product,
so a future release can restore a basin if it supplies valid cells.

To use a downloaded CRI file without renaming it, pass its path through the full chain:

```powershell
.venv\Scripts\python.exe scripts\run_chain.py --source jpl `
  --mascon-file data\raw\GRCTellus.JPL.latest.GLO.RL06.3M.MSCNv04CRI.nc
```

The default JPL chain also expects the matching Li & Kusche files under
`data/raw/li2026/JPL-FCast/global_gridded/`. As with CSR, `phase7_gnn` is defined but omitted
from the default because it is exceptionally expensive; run it explicitly after its
dependencies with `--source jpl --steps phase7_gnn`. Publication figures are not run for
JPL because their assertions intentionally pin the manuscript's archived CSR numbers.

The JPL-versus-Li tables report two spatial samples. `all_matched` retains every basin/date
available to every compared model. `joint_full_cells` is the strict resolution sensitivity:
a basin must contain every 0.25-degree mask subcell of at least one native JPL `mascon_ID`
and all sixteen subcells of at least one finite 1-degree Li forecast cell. This is a literal
containment test, not an area or fractional-coverage proxy. The per-basin Li diagnostics and
JPL hybrid comparison use this strict subset, while both pooled samples remain in the summary
and headline CSVs so the effect of the spatial restriction is visible.

Compact JPL headline, summary, statistical, and basin-diagnostic tables are versioned under
`results/jpl/` for cross-machine review. Large prediction-level tables, placebo draws, model
state, and chain logs remain local; their checksums are recorded in
`results/jpl/SHA256_MANIFEST_LIVE.csv`. The exact releases, scale-factor setting, run dates,
comparison populations, seeds, and input checksums are documented in
`results/jpl/RUN_PROVENANCE.md`.

The only things the chain does *not* do are the downloads themselves, because those need a
network or credentials: the CSR satellite and ancillary files, the basin mask,
`scripts/download_era5.py`, the Li 2026 archive, and `scripts/download_indices.py` (section 4
covers where each comes from). Once those are on disk, `run_chain.py` with no arguments is the
reproduction recipe for everything except the figures.

**Figures:** `scripts/make_figures.py` builds the paper's charts. It deliberately *crashes* if
any plotted value disagrees with expected numbers transcribed into the script from
`paper/notes/REWRITE_LEDGER.md`. That's a feature — a changed result can't silently redraw a
figure. (The transcriptions live in the script itself, so when a result legitimately changes
you update the ledger *and* the script's assert next to it.)

---

## 8. Rules for working here

A few conventions. Please don't break them — each one exists because something went wrong before.

- **`results/RUN_LOG.md` is append-only.** It's a diary of what was true when each batch ran.
  Never edit an old entry to match a newer result. Add a new entry instead.
- **`paper/notes/REWRITE_LEDGER.md` is the only authoritative source for numbers in the paper.**
  If you change a result, update the ledger and the matching hardcoded assert in
  `scripts/make_figures.py`; the figure build then re-verifies the plotted values. The ledger as
  it stands is pre-reframe: its numbers still describe the earlier three-finding manuscript, and
  it will be regenerated after `paper/main.tex` is rewritten. `docs/STUDY_CONTEXT.md` and
  `docs/ARCHIVE_MANIFEST.md` carry the same status note.
- **Don't touch `archive/`.** It's a frozen snapshot of pre-audit results, checksummed. It exists
  so we can always show what changed and when.
- **Big result files aren't in git.** They regenerate from the code plus raw data.
  `scripts/make_manifest.py` checksums them; `--check` verifies them later.
- **The notebooks have their outputs deliberately cleared**, so old numbers sitting in a saved
  cell can't be mistaken for current ones.
- **Run the tests before you commit.** `pytest tests/ -q`, about 30 seconds, as in section 4.
- **`docs/STUDY_CONTEXT.md` and `docs/CODE_MAP.md` are living docs** — when a milestone lands,
  update their status lines in the same commit, or the next cold reader inherits a stale map.

---

## 9. Glossary

| Term | Plain meaning |
|---|---|
| **GRACE / GRACE-FO** | Twin satellites that measure Earth's gravity, and therefore how much water each region holds. |
| **TWSA** | Terrestrial Water Storage Anomaly — how much wetter or drier a place is than normal. |
| **Basin** | The land area that drains into one river system. We use 234 of them. |
| **Lead** | How many months ahead we're forecasting. Lead 1 = next month. |
| **Deseasonalized** | Long-term trend and the annual wet/dry cycle removed, so only the hard-to-predict part is left. |
| **Damped persistence** | The field's standard baseline: "next month looks like this month, but closer to normal." |
| **Kalman filter** | A method that separates the true signal from measurement noise before forecasting. Section 5. |
| **ρ (rho)** | How "sticky" a basin is — what fraction of its anomaly carries over each month. |
| **Kalman gain** | How much the filter trusts a new measurement versus its own prediction. |
| **Mascon** | A tile (~120 km across) that the satellite data is delivered on. Basins are built from these. |
| **Contamination / leakage** | When a basin's number partly reflects water in *neighbouring* land, because the tiles are coarser than the basin. |
| **LSTM** | A type of neural network built for sequences, reading a run of months in order. Every LSTM arm here is an extended step, and none of them beats the flat ridge. |
| **Flat 12-month window** | The last 12 months of filtered state and weather, flattened into one row of numbers and handed to a ridge regression. The correction stage of our best model. |
| **GRACE-FCast** | Li and Kusche's published TWSA forecast product, the external system we compare against. |
| **Placebo test** | Rerunning with deliberately fake (random) neighbours, to check a result isn't just "more inputs help." Used by the extended neighbour steps only. |
| **Surrogate (IAAFT)** | Scrambled data that keeps each basin's own statistics but destroys cross-basin timing. A second null check, also extended-only. |
| **Fold** | One train/test round. We use 5, each testing on a later time period. |
| **Skill** | Percent improvement in error over a baseline. Higher is better. |
| **p-value** | Roughly, the chance of seeing a result this good if there were really no effect. Smaller is stronger. |
| **DM test** | Diebold–Mariano — the standard test for whether one forecast genuinely beats another. |
| **FDR** | A correction applied when testing many basins at once, so random flukes don't get counted as findings. |
| **OOF** | Out-of-fold — training a correction on predictions the model hasn't seen, so it can't cheat. |
