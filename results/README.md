# results/

Live outputs of the corrected pipeline (post 2026-08-13 repository audit, post
2026-08-15 external-audit repairs). Every CSV here was produced by the committed code
on the corrected date/fold protocol.

Provenance rules:

- **Pre-audit numbers are archived, not here.** Analyses and audits written against the
  pre-2026-08-13 pipeline (stale numbers, some sign-flipped) live in
  `archive/superseded_preaudit_docs/`; the frozen pre-audit artifact snapshot with its
  SHA256 manifest is `archive/pre_audit_2026-08-13/`. If a document in the archive
  disagrees with a CSV here, the CSV wins.
- `RUN_LOG.md` is the chronological experiment log and the authority on which command
  produced which file.
- Current-era analysis documents kept here: `phase7_corrected_analysis.md`,
  `phase8_corrected_audit.md`, `post_rerun_audit.md`, `phase8_related_benchmarks.md`
  (literature survey).
- Files larger than 5 MB (predictions, per-basin placebo tables) are excluded from git;
  their hashes are pinned in `SHA256_MANIFEST_LIVE.csv` once the corrected rerun chain
  completes.
- `chain_*.log` files are written by `scripts/run_chain.py` (fail-fast rerun chain).
