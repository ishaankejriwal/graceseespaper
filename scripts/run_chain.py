"""Fail-fast pipeline chain with declared dependencies (audit repair, 2026-08-15;
completed to the full pipeline after the 2026-08-17 external audit — the earlier
version consumed phase3b/fusion/coupled/Li outputs it never generated, and ran
basin_analysis before the phase6_era5 predictions it reads existed).

The 2026-08-15 morning chain was an inline PowerShell one-liner: steps joined by ';'
with no dependency checks, so basin analysis launched before the coupled-filter output
existed, crashed, and the chain sailed on. This runner declares each step's input and
output files, verifies inputs BEFORE launching, verifies outputs (existence + fresh
mtime) after, stops at the first failure, and logs each step to results/chain_<name>.log.

The default list is the paper spine: processed-and-raw data, the baselines, the
Kalman reference forecast and its flat-12 ridge correction, the mission-split
sensitivity, the ladder, the Li comparison, figures and the checksum manifest. It
needs neither torch nor any neighbor experiment. Everything neighbor-only or
torch-heavy sits in the EXTENDED list, run with --extended or named in --steps; no
script was deleted when the spine was narrowed. What stays manual is only what needs
a network or credentials: downloading the CSR mascon + ancillary files, the basin
mask, ERA5 (scripts/download_era5.py), the Li 2026 archive, and the climate indices
(scripts/download_indices.py). See README "Getting set up".

Usage:
  python scripts/run_chain.py             # default step list, in order
  python scripts/run_chain.py --extended  # the neighbor and torch steps
  python scripts/run_chain.py --source jpl # same experiments, isolated JPL outputs
  python scripts/run_chain.py --steps a b # explicit subset, in the order given
  python scripts/run_chain.py --list      # show steps and their dependencies

The figures step reads several extended outputs (phase 3b, phase 5, phase 6 ERA5,
phase 8), so a machine that has only ever run the default list is blocked there with
the missing files named. That is the dependency check working, not a defect.

kalman_fold_params.pkl is deliberately absent from every step's OUTPUT list: it is a
content-addressed cache (src/gracefc/cache.py) that flat12_ridge (default) and phase3b
(extended) create when missing but legitimately leave untouched when the fingerprint
still matches, which would fail the fresh-mtime output check. Steps that need it
declare it as an INPUT; the two steps that can build it do not.
"""
import argparse
import os
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

    ("flat12_ridge",
     ["scripts/run_flat12_ridge.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      DATA / "era5_basin_month.csv"],
     [RESULTS / "flat12_ridge_predictions.csv", RESULTS / "flat12_ridge_summary.csv"]),

    ("kalman_mission",
     ["scripts/run_kalman_mission_sensitivity.py"],
     [DATA / "basin_month_twsa_global.csv", DATA / "basin_meta.csv",
      RESULTS / "kalman_predictions.csv", RESULTS / "phase2_baseline_predictions.csv"],
     [RESULTS / "kalman_mission_predictions.csv", RESULTS / "kalman_mission_params.csv",
      RESULTS / "kalman_mission_summary.csv", RESULTS / "kalman_mission_perbasin_h1.csv"]),

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
      RESULTS / "kalman_predictions.csv", RESULTS / "flat12_ridge_predictions.csv",
      RESULTS / "phase2_baseline_predictions.csv"],
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
      DATA / "li2026_basin_coverage.csv", DATA / "basin_meta.csv"],
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
     [RESULTS / "phase2_baseline_predictions.csv", RESULTS / "kalman_predictions.csv",
      RESULTS / "flat12_ridge_predictions.csv"],
     [RESULTS / "paper_baseline_ladder.csv", RESULTS / "paper_baseline_contrasts.csv"]),

    ("conventional_metrics",
     ["scripts/compute_conventional_metrics.py"],
     [DATA / "basin_month_twsa_global.csv", RESULTS / "phase2_baseline_predictions.csv",
      RESULTS / "kalman_predictions.csv"],
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
# The default chain is the reframed paper spine: processed tables, the baselines, the
# Kalman reference forecast and its flat-12 ridge correction, the mission-split
# sensitivity, the ladder, the cross-product comparison, figures and the manifest.
# It needs no torch and touches no neighbor experiment.
DEFAULT = [
    "build_basin", "build_era5", "build_li", "phase2", "kalman", "flat12_ridge",
    "kalman_mission", "r0_ablation", "ladder", "li_comparison",
    "conventional_metrics", "figures", "manifest",
]
# Everything neighbor-only or torch-heavy. Run with --extended, or name in --steps.
EXTENDED = [s[0] for s in STEPS if s[0] not in DEFAULT]

SHARED_DATA_FILES = {
    "indices.csv", "era5_basin_month.csv", "era5_basin_coverage.csv",
}
JPL_UNAVAILABLE = {"figures"}


def steps_for_source(source: str, mascon_file: Path | None = None,
                     no_scale_factors: bool = False):
    """Rewrite declared paths/commands into an isolated product namespace."""
    if source == "csr":
        return STEPS, DEFAULT, RESULTS
    data_jpl = DATA / source
    results_jpl = RESULTS / source
    figures_jpl = FIGURES / source
    jpl_mascon = mascon_file or (
        RAW / "GRCTellus.JPL.200204_202604.GLO.RL06.3M.MSCNv04.nc"
    )
    converted = []
    for name, cmd, inputs, outputs in STEPS:
        cmd = list(cmd)
        if name in {"build_basin", "build_li"}:
            cmd += ["--source", source]
        if name == "build_basin" and mascon_file is not None:
            cmd += ["--mascon-file", str(jpl_mascon)]
        if name == "build_basin" and no_scale_factors:
            cmd += ["--no-scale-factors"]

        def remap(path: Path) -> Path:
            if path == MASCON_NC:
                return jpl_mascon
            if path == RAW / "li2026" / "CSR-FCast" / "global_gridded":
                return RAW / "li2026" / "JPL-FCast" / "global_gridded"
            if path.is_relative_to(DATA):
                if path.name in SHARED_DATA_FILES:
                    return path
                relative = path.relative_to(DATA)
                if relative.name == "li2026_csr_basin_forecasts.csv":
                    relative = relative.with_name("li2026_jpl_basin_forecasts.csv")
                return data_jpl / relative
            if path.is_relative_to(RESULTS):
                return results_jpl / path.relative_to(RESULTS)
            if path.is_relative_to(FIGURES):
                return figures_jpl / path.relative_to(FIGURES)
            return path

        mapped_inputs = [remap(p) for p in inputs]
        if name == "resolution":
            mapped_inputs = [jpl_mascon, MASK_NC,
                             results_jpl / "phase3b_predictions.csv"]
        converted.append((name, cmd, mapped_inputs, [remap(p) for p in outputs]))
    default = [name for name in DEFAULT if name not in JPL_UNAVAILABLE]
    return converted, default, results_jpl


def run_step(name: str, args: list[str], inputs: list[Path], outputs: list[Path],
             results: Path) -> None:
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit(f"[{name}] BLOCKED — missing inputs:\n  " + "\n  ".join(missing))
    results.mkdir(parents=True, exist_ok=True)
    log = results / f"chain_{name}.log"
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
    ap.add_argument("--source", choices=("csr", "jpl"), default="csr")
    ap.add_argument("--mascon-file", type=Path,
                    help="JPL NetCDF path (useful for the recommended CRI product)")
    ap.add_argument("--no-scale-factors", action="store_true",
                    help="disable scale factors when building a JPL CRI target")
    ap.add_argument("--steps", nargs="+", default=None)
    ap.add_argument("--extended", action="store_true",
                    help="run the extended steps (neighbor experiments, torch models) "
                         "instead of the default list")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.mascon_file is not None and args.source != "jpl":
        ap.error("--mascon-file currently applies only to --source jpl")
    if args.no_scale_factors and args.source != "jpl":
        ap.error("--no-scale-factors applies only to --source jpl")
    mascon_file = args.mascon_file.resolve() if args.mascon_file is not None else None
    if args.steps and args.extended:
        ap.error("--steps and --extended are mutually exclusive")
    steps, default, results = steps_for_source(
        args.source, mascon_file, args.no_scale_factors
    )
    by_name = {s[0]: s for s in steps}
    if args.list:
        for name, cmd, inputs, outputs in steps:
            if args.source == "jpl" and name in JPL_UNAVAILABLE:
                flag = "  [unavailable for JPL]"
            else:
                flag = "" if name in default else "  [extended]"
            print(f"{name}{flag}\n  cmd: {' '.join(cmd)}")
            for label, paths in (("in", inputs), ("out", outputs)):
                for p in paths:
                    try:
                        shown = p.relative_to(ROOT)
                    except ValueError:
                        shown = p
                    print(f"  {label}:  {shown}")
        return
    chosen = args.steps or (list(EXTENDED) if args.extended else default)
    unknown = [s for s in chosen if s not in by_name]
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}; use --list")
    unavailable = sorted(set(chosen) & JPL_UNAVAILABLE) if args.source == "jpl" else []
    if unavailable:
        raise SystemExit(
            "steps unavailable for JPL: " + ", ".join(unavailable)
            + "; publication figures contain CSR publication-number assertions"
        )
    os.environ["GRACEFC_SOURCE"] = args.source
    if mascon_file is not None:
        os.environ["GRACEFC_MASCON_FILE"] = str(mascon_file)
    else:
        os.environ.pop("GRACEFC_MASCON_FILE", None)
    print(f"source: {args.source}; results: {results.relative_to(ROOT)}", flush=True)
    print(f"chain: {' -> '.join(chosen)}", flush=True)
    for name in chosen:
        run_step(*by_name[name], results)
    print("chain complete", flush=True)


if __name__ == "__main__":
    main()
