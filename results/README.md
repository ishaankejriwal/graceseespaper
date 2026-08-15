# results/

Every answer the pipeline has produced. Nothing in here is written by hand — it all comes out
of the scripts.

## Which file do I open?

| Pattern | What it is |
|---|---|
| `*_summary.csv` | **Start here.** The scores: how well each model did, at each lead. |
| `*_headline.csv` | The one-line version of a summary — the number that ends up in the paper. |
| `*_predictions.csv` | Every individual forecast, one row each. Large, and not stored in git. |
| `*_placebo_*.csv` | The same thing for the fake-neighbour control runs. Also large, also not in git. |
| `*_analysis.md`, `*_audit.md` | Written notes on what a phase found and whether it held up. |
| `RUN_LOG.md` | The diary. Which command produced which file, when, and what we concluded. |
| `chain_*.log` | Raw console output from each pipeline stage. Transient — regenerated every run. |

If you only read one file, read `RUN_LOG.md`.

## Rules

- **`RUN_LOG.md` is append-only.** It records what was true when each batch ran. Never edit an
  old entry so it agrees with a newer result — add a new entry instead. The whole point is being
  able to see what changed.
- **If an archived document disagrees with a CSV in here, the CSV wins.** Everything in this
  folder was produced by the current code on the corrected protocol. Older analyses, some with
  numbers that later flipped sign, are frozen in `archive/superseded_preaudit_docs/`, and the
  pre-audit output snapshot is in `archive/pre_audit_2026-08-13/` with its own checksums.
- **Big files aren't in git** — anything over about 5 MB regenerates from the code plus the raw
  data. `scripts/make_manifest.py` writes their checksums to `SHA256_MANIFEST_LIVE.csv`, and
  `make_manifest.py --check` verifies them later.
- **Don't hand-edit a CSV.** If a number looks wrong, fix the code and rerun the stage. A
  hand-patched result is invisible to everyone downstream, which is exactly how the problems the
  2026-08-13 audit found got in.
