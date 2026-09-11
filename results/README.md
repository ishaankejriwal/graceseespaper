# results/

Every answer the pipeline has produced. Nothing in here is written by hand, it all comes out of
the scripts.

## Which file do I open?

Start with the headline files behind the three claims (see `docs/STUDY_CONTEXT.md`):

| File | What it holds |
|---|---|
| `paper_baseline_ladder.csv` | Every baseline and both retained models, leads 1 to 6, on matched rows, with skill against the stronger damped-persistence variant at each lead. |
| `paper_baseline_contrasts.csv` | The pairwise tests behind the ladder: skill, bootstrap CI, DM statistic and p for each named contrast. |
| `flat12_ridge_summary.csv` | The flat-12 ridge arms scored against the Kalman reference and the own-basin ridge. |
| `phase6_li_comparison_headline.csv` | Our models against Li and Kusche's GRACE-FCast, both orientations, on `all_matched`, `coverage_ge_0.5` and the strict `joint_full_cells` subset. |
| `kalman_mission_summary.csv` | The tested-and-rejected mission-split filter, plus the fitting diagnostics for both variants. |
| `r0_ablation_summary.csv` | Which half of the Kalman filter earns the win. |
| `conventional_metrics_summary.csv` | The retained systems restated in the literature's metrics: per-basin RMSE in cm, CC, NSE, anomaly and full signal. |
| `jpl/` | The same headline and summary tables from the collaborator's JPL run, most importantly `jpl/paper_baseline_ladder.csv`, `jpl/phase6_li_comparison_headline.csv` and `jpl/phase3b_summary.csv`. Row-level JPL files are not here; see `jpl/RUN_PROVENANCE.md`. |

Then the general patterns:

| Pattern | What it is |
|---|---|
| `*_headline.csv` | The one-line version of a summary: the number that ends up in the paper. |
| `*_summary.csv` | The scores: how well each model did, at each lead. |
| `*_predictions.csv` | Every individual forecast, one row each. Large, and not stored in git. |
| `*_placebo_*.csv` | The same thing for the fake-neighbour control runs, from the extended steps. Also large, also not in git. |
| `*_analysis.md`, `*_audit.md` | Written notes on what a phase found and whether it held up. |
| `RUN_LOG.md` | The diary. Which command produced which file, when, and what we concluded. |
| `chain_*.log` | Raw console output from each pipeline stage. Transient, regenerated every run. |

If you only read one file, read `RUN_LOG.md`.

## Rules

- **`RUN_LOG.md` is append-only.** It records what was true when each batch ran. Never edit an
  old entry so it agrees with a newer result, add a new entry instead. The whole point is being
  able to see what changed.
- **If a document disagrees with a CSV in here, the CSV wins.** Everything in this folder was
  produced by the current code. Superseded analyses are frozen under `archive/`, with their own
  checksums, and `paper/main.tex` has not yet been rewritten for the 2026-09-10 reframe.
- **Big files aren't in git.** Anything over about 5 MB regenerates from the code plus the raw
  data. `scripts/make_manifest.py` writes their checksums to `SHA256_MANIFEST_LIVE.csv`, and
  `make_manifest.py --check` verifies them later.
- **Don't hand-edit a CSV.** If a number looks wrong, fix the code and rerun the stage. A
  hand-patched result is invisible to everyone downstream.
