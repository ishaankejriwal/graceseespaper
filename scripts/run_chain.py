"""Fail-fast rerun chain with declared dependencies (audit repair, 2026-08-15).

The 2026-08-15 morning chain was an inline PowerShell one-liner: steps joined by ';'
with no dependency checks, so basin analysis launched before the coupled-filter output
existed, crashed, and the chain sailed on. This runner declares each step's input and
output files, verifies inputs BEFORE launching, verifies outputs (existence + fresh
mtime) after, stops at the first failure, and logs each step to results/chain_<name>.log.

Usage:
  python scripts/run_chain.py             # default step list, in order
  python scripts/run_chain.py --steps a b # explicit subset, in the order given
  python scripts/run_chain.py --list      # show steps and their dependencies

phase7_gnn is defined but not in the default list: its real arms (the only quoted
numbers — the GNN never beats ridge) are unaffected by the placebo seed repair, and its
full-batch training is by far the most expensive step. Run it explicitly if the GNN
placebo ranks are ever quoted.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# (name, script args, input files, output files)
STEPS: list[tuple[str, list[str], list[Path], list[Path]]] = [
    ("predlag",
     ["scripts/run_phase3b_kalman_neighbors.py", "--cells", "pred_lag1:1,2",
      "--seeds", "50", "--tag", "phase5_predlag"],
     [DATA / "basin_month_twsa_global.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase5_predlag_predictions.csv", RESULTS / "phase5_predlag_summary.csv"]),

    ("conditioned",
     ["scripts/run_phase3b_kalman_neighbors.py", "--cells", "corr:1", "corr_min300:1",
      "--seeds", "50", "--tag", "phase4_conditioned", "--condition-indices", "nino34", "dmi"],
     [DATA / "indices.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase4_conditioned_predictions.csv", RESULTS / "phase4_conditioned_summary.csv"]),

    ("surrogates",
     ["scripts/run_phase4_surrogates.py"],
     [RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase4_surrogate_summary.csv"]),

    ("r0_ablation",
     ["scripts/run_r0_ablation.py"],
     [RESULTS / "kalman_predictions.csv", RESULTS / "phase2_baseline_predictions.csv"],
     [RESULTS / "kalman_r0_predictions.csv", RESULTS / "r0_ablation_summary.csv"]),

    ("phase5_nonlinear",
     ["scripts/run_phase5_nonlinear.py"],
     [DATA / "basin_month_twsa_global.csv"],
     [RESULTS / "phase5_nonlinear_predictions.csv"]),

    ("phase5_stats",
     ["scripts/run_phase5_stats.py"],
     [RESULTS / "phase3b_predictions.csv", RESULTS / "phase5_predlag_predictions.csv",
      RESULTS / "phase5_fusion_predictions.csv", RESULTS / "phase5_coupled_predictions.csv",
      RESULTS / "phase5_nonlinear_predictions.csv"],
     [RESULTS / "phase5_headline_table.csv", RESULTS / "phase5_perbasin_fdr_h1.csv"]),

    ("basin_analysis",
     ["scripts/run_phase6_basin_analysis.py"],
     [RESULTS / "phase5_coupled_coupling.csv", RESULTS / "phase4_conditioned_predictions.csv",
      RESULTS / "phase2_strata.csv", RESULTS / "phase6_li_comparison_perbasin.csv",
      DATA / "era5_basin_coverage.csv"],
     [RESULTS / "phase6_basin_analysis_summary.csv"]),

    ("phase6_era5",
     ["scripts/run_phase6_era5.py"],
     [DATA / "era5_basin_month.csv"],
     [RESULTS / "phase6_era5_headline.csv"]),

    ("phase8_h13",
     ["scripts/run_phase8_lstm_combined.py", "--horizons", "1-3"],
     [DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase8_lstm_combined_summary.csv"]),

    ("phase8_h46",
     ["scripts/run_phase8_lstm_combined.py", "--horizons", "4-6", "--tag", "phase8b_lstm_h46"],
     [DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase8b_lstm_h46_summary.csv"]),

    ("phase8b_merge",
     ["scripts/run_phase8b_merge.py"],
     [RESULTS / "phase8_lstm_combined_predictions.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv"],
     [RESULTS / "phase8b_h16_headline.csv", RESULTS / "phase8b_h16_ensemble_headline.csv"]),

    ("phase8_strat",
     ["scripts/run_phase8_stratification.py"],
     [RESULTS / "resolution_diagnostics.csv", RESULTS / "phase8_lstm_combined_predictions.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv"],
     [RESULTS / "phase8_stratification.csv"]),

    ("phase7_resmlp",
     ["scripts/run_phase7_resmlp.py"],
     [DATA / "era5_basin_month.csv"],
     [RESULTS / "phase7_resmlp_summary.csv"]),

    ("phase7_lstm",
     ["scripts/run_phase7_lstm.py"],
     [DATA / "era5_basin_month.csv"],
     [RESULTS / "phase7_lstm_summary.csv"]),

    ("phase7_gnn",
     ["scripts/run_phase7_gnn.py"],
     [DATA / "era5_basin_month.csv"],
     [RESULTS / "phase7_gnn_summary.csv"]),
]
DEFAULT = [s[0] for s in STEPS if s[0] != "phase7_gnn"]


def run_step(name: str, args: list[str], inputs: list[Path], outputs: list[Path]) -> None:
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"[{name}] BLOCKED — missing inputs:\n  " + "\n  ".join(missing))
    log = RESULTS / f"chain_{name}.log"
    t0 = time.time()
    print(f"[{name}] start -> {log.name}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([str(PY), *args], cwd=ROOT, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t0) / 60
    if rc != 0:
        raise SystemExit(f"[{name}] FAILED (exit {rc}, {mins:.1f} min) — see {log}")
    stale = [str(p) for p in outputs
             if not p.exists() or p.stat().st_mtime < t0 - 1]
    if stale:
        raise SystemExit(f"[{name}] FAILED — outputs missing or not refreshed:\n  "
                         + "\n  ".join(stale))
    print(f"[{name}] done ({mins:.1f} min)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    by_name = {s[0]: s for s in STEPS}
    if args.list:
        for name, cmd, inputs, outputs in STEPS:
            flag = "" if name in DEFAULT else "  [not in default list]"
            print(f"{name}{flag}\n  cmd: {' '.join(cmd)}")
        return
    chosen = args.steps or DEFAULT
    unknown = [s for s in chosen if s not in by_name]
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}; use --list")
    print(f"chain: {' -> '.join(chosen)}", flush=True)
    for name in chosen:
        run_step(*by_name[name])
    print("chain complete", flush=True)


if __name__ == "__main__":
    main()
