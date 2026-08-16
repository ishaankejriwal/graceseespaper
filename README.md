# GRACE TWSA Forecasting Study

**In one sentence:** we try to predict how much water is stored in the world's river basins a
few months from now, and we found that the method everyone in the field uses as their
comparison point is leaving easy accuracy on the table.

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

Three things. Each one is explained in plain terms below.

### Finding 1: The standard benchmark is beatable, and we know exactly why

To claim your forecast is *good*, you have to beat something. Almost every paper in this field
compares against **damped persistence**, which is a fancy name for a simple idea:

> "Next month will look like this month, but a bit closer to normal."

You take the most recent measurement and shrink it toward zero. That's it. It works
surprisingly well, because water storage changes slowly.

**Here's the flaw.** The satellite measurement is noisy — it's not the true water level, it's
the true water level plus measurement error. Damped persistence shrinks the *measurement*. So it
faithfully carries the noise forward along with the signal.

Our fix is a **Kalman filter** (explained properly in section 5). In one line: it separates the
real underlying water level from the measurement noise *first*, then does the shrinking on the
clean estimate.

That change alone buys **+5.0% accuracy at lead 1 and +8.8% at lead 2**. It's better than
per-basin ridge regression at five of the six leads. And it's not a small technical footnote —
it's a free improvement available to anyone in this field who is currently benchmarking the
usual way.

We also proved *which part* of the filter does the work. The filter does two things: it removes
noise, and it estimates how fast each basin drifts back to normal. We built a version with the
noise removal switched off, keeping everything else. It performs *worse than plain damped
persistence* at leads 2–6. So the noise removal is where the win comes from — not better
drift estimation.

### Finding 2: Cleaning up the measurement wins at short leads; weather data wins at long leads

We compared our system head-to-head against a published forecast product by Li & Kusche (2026),
over 227 basins they and we both cover, across 60 months. Their system uses climate and weather
inputs; ours mostly cleans up the satellite signal.

- **Lead 1: we are 20.0% better.**
- **Lead 2: a tie.**
- **Leads 3–6: they are 12.9% to 30.3% better.**

There's a crossover between leads 2 and 3. This makes intuitive sense: next month is mostly
determined by where the water already is, so getting a clean read on the current state matters
most. Six months out, the current state has washed out and what matters is what the weather is
going to do.

The striking detail: their lead-1 forecast is 11% *worse* than damped persistence. A whole
weather-driven modeling system, beaten at short range by "next month looks like this month."

### Finding 3: Neighbouring basins help — but only if you use the information the right way

The original question was: if I know what's happening in the basins around me, does that help me
forecast my own?

The answer turns out to depend entirely on *how* you feed that information to the model:

- **Wired in as a plain linear term:** nothing. +0.31% at lead 1, not statistically significant.
- **Fed to a neural network as an extra input:** nothing, or slightly harmful.
- **Used as a separate correction step, applied after a first model has made its guess:**
  **+0.91% to +1.96%** across leads 1–6, and overwhelmingly statistically significant
  (p ≤ 3.2e-8 at every lead).

Same information. Same data. Three delivery mechanisms, wildly different outcomes. That's the
result — *delivery decides*.

We were careful here, because "my model improved" is easy to fool yourself about. See section 6
on how we checked.

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
| `docs/` | Longer explanations: code map, project status, past audits, reference papers. | Read |
| `notebooks/` | Two Jupyter notebooks for poking at the data interactively. | Optional |
| `data/`, `archive/` | Raw inputs and frozen old results. Not stored in git — see section 4. | Download once |

**If you're new, read in this order:** this README → [`docs/STUDY_CONTEXT.md`](docs/STUDY_CONTEXT.md)
(what's done, what's in progress) → [`docs/CODE_MAP.md`](docs/CODE_MAP.md) (what each file does)
→ [`results/RUN_LOG.md`](results/RUN_LOG.md) (the diary).

---

## 4. Getting set up

You need Python 3.11 or newer. Everything runs on a normal laptop CPU — no GPU needed.

**Install:**

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-lock.txt
```

**Check it worked** by running the test suite. This takes about 15 seconds and needs no data:

```bash
.venv/Scripts/python -m pytest tests/ -q
```

You should see `15 passed`. If you do, your environment is fine.

### Downloading the data

The data files are big, so they aren't in git. You download them once:

| What | Where it goes | How to get it |
|---|---|---|
| GRACE satellite measurements | repo root, `CSR_..._Mascons_....nc` | [CSR mascon page](http://www2.csr.utexas.edu/grace/RL0603_mascons.html) |
| River basin boundaries | repo root, `HydroShed+Mascon_Basins_L3.nc` | ships with the project |
| ERA5 weather data | `data/raw/era5/` | run `scripts/download_era5.py` (needs a free [CDS account](https://cds.climate.copernicus.eu/)) |

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
land at r ≈ 0. The paper is careful about this wording, and you should be too.

### How the neural network models use the filter

The best-performing system isn't a replacement for the Kalman filter — it's built on top of it.
Its final prediction is literally three things added together:

```
prediction  =  kalman forecast  +  LSTM correction  +  neighbour correction
```

- The **Kalman forecast** is the baseline guess described above.
- The **LSTM** (a neural network for sequences) looks at 12 months of the basin's *filtered*
  history plus 11 weather variables, and predicts *how wrong the Kalman guess will be*.
- The **neighbour correction** takes the filtered state of the most-connected neighbouring
  basin and predicts how wrong the answer *still* is.

Every model in this study that works is shaped like "good baseline + small learned correction."
Models that tried to predict water storage from scratch, ignoring the filter, did worse. That's
Finding 3 restated: the architecture that delivers information as a correction wins.

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

### Placebo tests

This is the one we're proudest of. To claim "neighbouring basins carry useful information," it
isn't enough to show the model improved when we added neighbours — maybe *any* extra input would
have helped.

So we rerun the whole thing with **fake neighbours**: random basins, wired up in a graph with the
exact same shape (same number of connections per basin), using the exact same random seed for
the model itself. The *only* difference is which basins are connected to which.

If the real graph doesn't beat the fake ones, the effect isn't about geography. Our headline
neighbour result beats **20 out of 20** fake graphs in all 12 test cells.

We also run **IAAFT surrogates**, a second independent check: scramble the data so each basin
keeps its own statistical character but loses its timing relationship with other basins. If the
result survives that too, the signal really is about basins moving together.

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

Run the whole thing:

```bash
.venv/Scripts/python scripts/run_chain.py
```

Heads up: a full run takes roughly **30 hours** on a laptop. The neural network stages dominate.
Run just part of it with `--steps name1 name2`. Each stage writes its own log to
`results/chain_<name>.log`.

Building from completely raw data the first time, the order is:
`build_basin_series.py` → `download_era5.py` → `build_era5_basin_table.py` →
`download_indices.py` → `run_phase2_baselines.py` → `run_kalman_baseline.py` → then
`run_chain.py` for everything after that.

**Figures:** `scripts/make_figures.py` builds the paper's charts. It deliberately *crashes* if
any plotted value disagrees with the recorded numbers in `paper/notes/REWRITE_LEDGER.md`. That's
a feature — it means the figures and the manuscript can't silently drift apart.

---

## 8. Rules for working here

A few conventions. Please don't break them — each one exists because something went wrong before.

- **`results/RUN_LOG.md` is append-only.** It's a diary of what was true when each batch ran.
  Never edit an old entry to match a newer result. Add a new entry instead.
- **`paper/notes/REWRITE_LEDGER.md` is the only authoritative source for numbers in the paper.**
  If you change a result, update the ledger, and the figures will re-verify against it.
- **Don't touch `archive/`.** It's a frozen snapshot of pre-audit results, checksummed. It exists
  so we can always show what changed and when.
- **Big result files aren't in git.** They regenerate from the code plus raw data.
  `scripts/make_manifest.py` checksums them; `--check` verifies them later.
- **The notebooks have their outputs deliberately cleared**, so old numbers sitting in a saved
  cell can't be mistaken for current ones.
- **Run the tests before you commit.** `pytest tests/ -q`, 15 seconds.

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
| **LSTM** | A type of neural network built for sequences — it reads a run of months in order. |
| **Placebo test** | Rerunning with deliberately fake (random) neighbours, to check a result isn't just "more inputs help." |
| **Surrogate (IAAFT)** | Scrambled data that keeps each basin's own statistics but destroys cross-basin timing. A second null check. |
| **Fold** | One train/test round. We use 5, each testing on a later time period. |
| **Skill** | Percent improvement in error over a baseline. Higher is better. |
| **p-value** | Roughly, the chance of seeing a result this good if there were really no effect. Smaller is stronger. |
| **DM test** | Diebold–Mariano — the standard test for whether one forecast genuinely beats another. |
| **FDR** | A correction applied when testing many basins at once, so random flukes don't get counted as findings. |
| **OOF** | Out-of-fold — training a correction on predictions the model hasn't seen, so it can't cheat. |
