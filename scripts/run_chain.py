"""Fail-fast pipeline chain with declared dependencies (audit repair, 2026-08-15;
completed to the full pipeline after the 2026-08-17 external audit — the earlier
version consumed phase3b/fusion/coupled/Li outputs it never generated, and ran
basin_analysis before the phase6_era5 predictions it reads existed).

The 2026-08-15 morning chain was an inline PowerShell one-liner: steps joined by ';'
with no dependency checks, so basin analysis launched before the coupled-filter output
existed, crashed, and the chain sailed on. This runner declares each step's input and
output files, verifies inputs BEFORE launching, verifies outputs (existence + fresh
mtime) after, stops at the first failure, and logs each step to results/chain_<name>.log.

The default list is the whole study from processed-and-raw data to figures and the
checksum manifest. What stays manual is only what needs a network or credentials:
downloading the CSR mascon + ancillary files, the basin mask, ERA5
(scripts/download_era5.py), the Li 2026 archive, and the climate indices
(scripts/download_indices.py). See README "Getting set up".

Usage:
  python scripts/run_chain.py             # default step list, in order
  python scripts/run_chain.py --steps a b # explicit subset, in the order given
  python scripts/run_chain.py --list      # show steps and their dependencies

phase7_gnn is defined but not in the default list: its real arms (the only quoted
numbers — the GNN never beats ridge) are unaffected by the placebo seed repair, and its
full-batch training is by far the most expensive step. Run it explicitly if the GNN
placebo ranks are ever quoted.

kalman_fold_params.pkl is deliberately absent from every step's OUTPUT list: it is a
content-addressed cache (src/gracefc/cache.py) that phase3b creates when missing but
legitimately leaves untouched when its fingerprint still matches, which would fail the
fresh-mtime output check. Steps that need it declare it as an INPUT.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "figures"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

MASCON_NC = ROOT / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
MASK_NC = ROOT / "HydroShed+Mascon_Basins_L3.nc"

# (name, script args, input files, output files)
STEPS: list[tuple[str, list[str], list[Path], list[Path]]] = [
    ("build_basin",
     ["scripts/build_basin_series.py"],
     [MASCON_NC, MASK_NC],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv"]),

    ("build_era5",
     ["scripts/build_era5_basin_table.py"],
     [RAW / "era5", MASK_NC],
     [DATA / "era5_basin_month.csv", DATA / "era5_basin_coverage.csv"]),

    ("build_li",
     ["scripts/build_li_basin_series.py"],
     [RAW / "li2026" / "CSR-FCast" / "global_gridded", MASK_NC, DATA / "basin_meta.csv"],
     [DATA / "li2026_csr_basin_forecasts.csv", DATA / "li2026_basin_coverage.csv"]),

    ("phase2",
     ["scripts/run_phase2_baselines.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv", DATA / "indices.csv"],
     [RESULTS / "phase2_baseline_predictions.csv", RESULTS / "phase2_strata.csv",
      RESULTS / "phase2_baseline_summary.csv"]),

    ("kalman",
     ["scripts/run_kalman_baseline.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "phase2_baseline_predictions.csv"],
     [RESULTS / "kalman_predictions.csv"]),

    ("phase3b",
     ["scripts/run_phase3b_kalman_neighbors.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv"],
     [RESULTS / "phase3b_predictions.csv", RESULTS / "phase3b_summary.csv",
      RESULTS / "phase3b_placebo_basin.csv"]),

    ("jump_screen",
     ["scripts/run_jump_screen.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "phase3b_predictions.csv", RESULTS / "phase3b_placebo_basin.csv"],
     [RESULTS / "phase4_jump_screen.csv"]),

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
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase4_surrogate_summary.csv"]),

    ("r0_ablation",
     ["scripts/run_r0_ablation.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_predictions.csv", RESULTS / "phase2_baseline_predictions.csv"],
     [RESULTS / "kalman_r0_predictions.csv", RESULTS / "r0_ablation_summary.csv"]),

    ("phase5_fusion",
     ["scripts/run_phase5_fusion.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase5_fusion_predictions.csv", RESULTS / "phase5_fusion_summary.csv"]),

    ("phase5_coupled",
     ["scripts/run_phase5_coupled.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase5_coupled_predictions.csv", RESULTS / "phase5_coupled_coupling.csv",
      RESULTS / "phase5_coupled_summary.csv"]),

    ("phase5_nonlinear",
     ["scripts/run_phase5_nonlinear.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase5_nonlinear_predictions.csv"]),

    ("phase5_stats",
     ["scripts/run_phase5_stats.py"],
     [RESULTS / "phase3b_predictions.csv", RESULTS / "phase5_predlag_predictions.csv",
      RESULTS / "phase5_fusion_predictions.csv", RESULTS / "phase5_coupled_predictions.csv",
      RESULTS / "phase5_nonlinear_predictions.csv"],
     [RESULTS / "phase5_headline_table.csv", RESULTS / "phase5_perbasin_fdr_h1.csv"]),

    ("li_comparison",
     ["scripts/run_phase6_li_comparison.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "li2026_csr_basin_forecasts.csv", DATA / "li2026_basin_coverage.csv",
      RESULTS / "phase3b_predictions.csv", RESULTS / "phase2_baseline_predictions.csv"],
     [RESULTS / "phase6_li_comparison_predictions.csv", RESULTS / "phase6_li_comparison_summary.csv",
      RESULTS / "phase6_li_comparison_headline.csv", RESULTS / "phase6_li_comparison_perbasin.csv"]),

    ("phase6_era5",
     ["scripts/run_phase6_era5.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase6_era5_headline.csv", RESULTS / "phase6_era5_predictions.csv"]),

    ("era5_attribution",
     ["scripts/run_phase6_era5_attribution.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "phase6_era5_predictions.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase6_era5_attribution.csv", RESULTS / "phase6_era5_attribution_folds.csv",
      RESULTS / "phase6_era5_attribution_continent.csv", RESULTS / "phase6_era5_attribution_fdr.csv"]),

    ("hybrid",
     ["scripts/run_phase6_hybrid.py"],
     [RESULTS / "phase6_li_comparison_predictions.csv", RESULTS / "phase6_era5_predictions.csv"],
     [RESULTS / "phase6_hybrid_summary.csv", RESULTS / "phase6_hybrid_headline.csv"]),

    ("basin_analysis",
     ["scripts/run_phase6_basin_analysis.py"],
     [RESULTS / "phase5_coupled_coupling.csv", RESULTS / "phase4_conditioned_predictions.csv",
      RESULTS / "phase2_strata.csv", RESULTS / "phase6_li_comparison_perbasin.csv",
      RESULTS / "phase3b_predictions.csv", RESULTS / "phase6_era5_predictions.csv",
      RESULTS / "kalman_fold_params.pkl", DATA / "era5_basin_coverage.csv"],
     [RESULTS / "phase6_basin_analysis_summary.csv"]),

    ("resolution",
     ["scripts/run_resolution_sensitivity.py"],
     [MASCON_NC, MASK_NC,
      RAW / "csr_ancillary" / "CSR_GRACE_GRACE-FO_RL0603_mascons_mapping_file.nc",
      RAW / "csr_ancillary" / "CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc",
      RESULTS / "phase3b_predictions.csv"],
     [RESULTS / "resolution_diagnostics.csv", RESULTS / "resolution_cross_2x2.csv",
      RESULTS / "resolution_cross_2x2_200k.csv"]),

    ("phase7_resmlp",
     ["scripts/run_phase7_resmlp.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase7_resmlp_summary.csv", RESULTS / "phase7_resmlp_predictions.csv"]),

    ("phase7_lstm",
     ["scripts/run_phase7_lstm.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase7_lstm_summary.csv", RESULTS / "phase7_lstm_predictions.csv"]),

    ("phase8_h13",
     ["scripts/run_phase8_lstm_combined.py", "--horizons", "1-3"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase8_lstm_combined_summary.csv",
      RESULTS / "phase8_lstm_combined_predictions.csv"]),

    ("phase8_h46",
     ["scripts/run_phase8_lstm_combined.py", "--horizons", "4-6", "--tag", "phase8b_lstm_h46"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase8b_lstm_h46_summary.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv"]),

    ("phase8b_merge",
     ["scripts/run_phase8b_merge.py"],
     [RESULTS / "phase8_lstm_combined_predictions.csv",
      RESULTS / "phase8_lstm_combined_summary.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv",
      RESULTS / "phase8b_lstm_h46_summary.csv",
      RESULTS / "phase6_li_comparison_predictions.csv",
      DATA / "li2026_basin_coverage.csv"],
     [RESULTS / "phase8b_h16_headline.csv", RESULTS / "phase8b_h16_ensemble_headline.csv",
      RESULTS / "phase8b_li_comparison_headline.csv", RESULTS / "phase8b_li_comparison_perbasin.csv"]),

    ("phase8_strat",
     ["scripts/run_phase8_stratification.py"],
     [RESULTS / "resolution_diagnostics.csv", RESULTS / "phase8_lstm_combined_predictions.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv",
      RESULTS / "phase7_resmlp_predictions.csv"],
     [RESULTS / "phase8_stratification.csv"]),

    ("phase7_gnn",
     ["scripts/run_phase7_gnn.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "phase7_gnn_summary.csv"]),

    ("flat12_train85",
     ["scripts/run_flat12_train85_sensitivity.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv", RESULTS / "phase7_lstm_predictions.csv",
      RESULTS / "kalman_fold_params.pkl"],
     [RESULTS / "flat12_train85_sensitivity.csv"]),

    ("ladder",
     ["scripts/build_paper_ladder.py"],
     [RESULTS / "phase2_baseline_predictions.csv", RESULTS / "phase3b_predictions.csv"],
     [RESULTS / "paper_baseline_ladder.csv", RESULTS / "paper_baseline_contrasts.csv"]),

    ("conventional_metrics",
     ["scripts/compute_conventional_metrics.py"],
     [DATA / "basin_month_twsa_global.csv", RESULTS / "phase2_baseline_predictions.csv",
      RESULTS / "kalman_predictions.csv", RESULTS / "phase8_lstm_combined_predictions.csv",
      RESULTS / "phase8b_lstm_h46_predictions.csv"],
     [RESULTS / "conventional_metrics_perbasin.csv", RESULTS / "conventional_metrics_summary.csv"]),

    ("figures",
     ["scripts/make_figures.py"],
     [MASK_NC, DATA / "basin_meta.csv",
      RESULTS / "paper_baseline_ladder.csv", RESULTS / "paper_baseline_contrasts.csv",
      RESULTS / "phase8b_li_comparison_headline.csv", RESULTS / "phase8b_li_comparison_perbasin.csv",
      RESULTS / "phase8b_h16_ensemble_headline.csv", RESULTS / "phase8b_h16_headline.csv",
      RESULTS / "phase8_stratification.csv", RESULTS / "phase3b_summary.csv",
      RESULTS / "phase3b_placebo_monthly.csv",
      RESULTS / "phase4_surrogate_summary.csv", RESULTS / "phase5_perbasin_fdr_h1.csv",
      RESULTS / "phase6_era5_headline.csv",
      RESULTS / "phase6_era5_predictions.csv", RESULTS / "phase4_conditioned_predictions.csv"],
     [FIGURES / f"{stem}.pdf" for stem in
      ("fig01_benchmark_ladder", "fig02_crossing", "fig03_neighbor_map", "fig04_controls",
       "fig05_delivery", "fig06_complementarity", "fig08_stratification")]),

    ("manifest",
     ["scripts/make_manifest.py"],
     [],
     [RESULTS / "SHA256_MANIFEST_LIVE.csv"]),
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
            for label, paths in (("in", inputs), ("out", outputs)):
                for p in paths:
                    print(f"  {label}:  {p.relative_to(ROOT)}")
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
