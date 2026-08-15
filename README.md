# GRACE TWSA Forecasting Study

Global basin-scale forecasting of deseasonalized GRACE/GRACE-FO terrestrial water storage
anomalies: benchmark design (state-space filtering), a head-to-head crossing against a
published system, and controlled cross-basin experiments. Target venue: HESS.

## Orientation (read in this order)

1. [`context_global_study.md`](context_global_study.md) — what the study is, current status, the three findings.
2. [`AUDIT_REPORT.md`](AUDIT_REPORT.md) — the 2026-08-13 repository audit; its P0 findings were
   independently verified and drive the current rerun.
3. [`code_structure.md`](code_structure.md) — code layout.
4. [`results/RUN_LOG.md`](results/RUN_LOG.md) — append-only run journal (provenance for every result batch).
5. [`paper/`](paper/) — manuscript (`main.tex`), decisions, figure plan.

## Layout

- `src/gracefc/` — engine (data build, folds, models, stats). `scripts/` — phase runners.
- `data/processed/` — derived tables. Raw inputs at repo root (CSR NetCDF, basin mask) and `data/raw/`.
- `results/` — current (post-audit) outputs + the markdown journal.
- `archive/pre_audit_2026-08-13/` — frozen pre-audit result artifacts + SHA256 manifest. Do not overwrite.

## Environment

Windows, Python venv at `.venv`. Pinned snapshot: `requirements-lock.txt` (pip freeze,
2026-08-13; includes torch CPU). No test suite yet; leakage assertions live in
`src/gracefc/evaluate.py` and data-build assertions in `src/gracefc/basins.py`.

## Reproducing the paper pipeline

Ordered runners (each writes to `results/` and appends context to `RUN_LOG.md`):
`build_basin_series.py` → `run_phase2_baselines.py` → `run_kalman_baseline.py` →
`run_phase3b_kalman_neighbors.py` → `run_phase6_era5.py` → phase 7 runners (torch) →
`run_phase8_lstm_combined.py` → `run_phase8b_merge.py` → `build_paper_ladder.py`.

All evaluation is issue-date fold membership (model frozen at fold start, used forward);
see `evaluate.split_fold` for the protocol and the audit note explaining why.
