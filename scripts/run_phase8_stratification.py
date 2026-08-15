"""Stratify the phase-8 stacked neighbor correction by mascon contamination and basin area.

The resolution-sensitivity audit (RUN_LOG 2026-08-15) showed the h1 LINEAR neighbor
effect is confined to high-contamination basins regardless of size — the 90k km2
"sub-resolution" framing was a proxy for footprint sharing. That analysis covered only
the linear Kalman arm at h1. This script answers the open question it ranked highest:
does the phase-8 stacked correction (the paper's headline neighbor result) also
concentrate in high-contamination basins, or does it survive where the linear effect
dies? Strata definitions replicate run_resolution_sensitivity exactly: contamination
terciles over the 234 keep basins, area split at the below_resolution flag, and the
2x2 cross. Also extends the linear stratification to h1-6 via the phase-8 ridge twins
(same row set, so directly comparable).

Horizon is filtered inside pooled_monthly_dm / block_bootstrap_skill_ci BEFORE the
(name, target_date) pairing merge, so the multi-horizon shared-target-date trap the
resolution audit hit cannot occur here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gracefc.stats import (  # noqa: E402
    _paired_losses, block_bootstrap_skill_ci, diebold_mariano, pooled_monthly_dm)

OUT = ROOT / "results"
KEY = ["name", "target_date", "horizon"]


def add_seed_ensembles(pred: pd.DataFrame, fams: list[str]) -> pd.DataFrame:
    """Mean-of-seeds ensemble rows (model f"{fam}_ens") for each family, any seed count."""
    parts = [pred]
    for fam in fams:
        seeds = sorted(m for m in pred["model"].unique()
                       if m.startswith(f"{fam}_s") and m[len(fam) + 2:].isdigit())
        if len(seeds) < 2:
            raise AssertionError(f"{fam}: found seeds {seeds}")
        base = pred[pred["model"] == seeds[0]].copy()
        n0 = len(base)
        for i, s in enumerate(seeds[1:], start=1):
            nxt = pred[pred["model"] == s][KEY + ["pred"]]
            base = base.merge(nxt, on=KEY, suffixes=("", f"_{i}"), validate="one_to_one")
            # inner join: a partial row-set mismatch SHRINKS the frame rather than
            # producing NaNs, so length against seed-0 is the real guard (audit 2026-08-15)
            if len(base) != n0:
                raise AssertionError(f"{fam}: seed row sets differ at {s} ({len(base)} vs {n0})")
        pcols = ["pred"] + [f"pred_{i}" for i in range(1, len(seeds))]
        base["pred"] = base[pcols].mean(axis=1)
        base = base.drop(columns=pcols[1:])
        base["model"] = f"{fam}_ens"
        parts.append(base)
    return pd.concat(parts, ignore_index=True)


def build_strata(diag: pd.DataFrame) -> dict[str, set[str]]:
    keep = diag[diag["exclude_reason"] == "keep"].copy()
    keep["cont_tercile"] = pd.qcut(keep["contamination"], 3, labels=["low", "mid", "high"])
    sub = keep["below_resolution"].astype(bool)
    high = keep["cont_tercile"] == "high"
    strata = {
        "all": keep,
        "area_resolved": keep[~sub],
        "area_sub_resolution": keep[sub],
        "cont_tercile_low": keep[keep["cont_tercile"] == "low"],
        "cont_tercile_mid": keep[keep["cont_tercile"] == "mid"],
        "cont_tercile_high": keep[high],
        "resolved_x_cont_high": keep[~sub & high],
        "resolved_x_cont_lowmid": keep[~sub & ~high],
        "sub_x_cont_high": keep[sub & high],
        "sub_x_cont_lowmid": keep[sub & ~high],
    }
    return {k: set(v["name"]) for k, v in strata.items()}


def run_contrasts(pred: pd.DataFrame, contrasts: list[tuple[str, str]],
                  strata: dict[str, set[str]], horizons: list[int], source: str) -> list[dict]:
    rows = []
    for a, b in contrasts:
        for label, names in strata.items():
            sub = pred[pred["name"].isin(names)]
            for h in horizons:
                point, lo, hi = block_bootstrap_skill_ci(sub, a, b, h)
                stat, p = pooled_monthly_dm(sub, a, b, h)
                sh = sub[(sub["horizon"] == h) & (sub["model"] == a)]
                rows.append({
                    "source": source, "challenger": a, "reference": b,
                    "stratum": label, "horizon": h,
                    "n_basins": sh["name"].nunique(), "n_rows": len(sh),
                    "skill_pct": 100 * point, "ci_lo_pct": 100 * lo,
                    "ci_hi_pct": 100 * hi, "dm_stat": stat, "dm_p": p,
                })
    return rows


def _stratum_monthly(pred: pd.DataFrame, a: str, b: str, names: set[str], h: int):
    """Monthly cross-basin mean loss differential (la−lb) and the reference loss series."""
    sub = pred[(pred["horizon"] == h) & pred["name"].isin(names)]
    j = _paired_losses(sub, a, b)
    la = j.groupby("target_date")["loss_a"].mean().sort_index()
    lb = j.groupby("target_date")["loss_b"].mean().sort_index()
    return la - lb, lb


def interaction_tests(pred: pd.DataFrame, strata: dict[str, set[str]],
                      a: str = "lstmres_corr_top1_ens",
                      b: str = "lstm_own_era5_ens") -> pd.DataFrame:
    """Formal stratum-interaction DM tests (method from the 2026-08-15 batch audit).

    Each stratum's monthly loss differential is normalized by that stratum's time-mean
    reference MSE (units: fractions of reference MSE), then the DM test runs on the
    difference of the two normalized series. Sign convention: mean_diff_pp < 0 means the
    FIRST-listed stratum gains MORE from the correction; > 0 means it gains less.
    """
    axes = [("cont_high_vs_low", "cont_tercile_high", "cont_tercile_low"),
            ("sub_vs_resolved", "area_sub_resolution", "area_resolved")]
    rows = []
    for axis, s1, s2 in axes:
        for h in range(1, 7):
            d1, r1 = _stratum_monthly(pred, a, b, strata[s1], h)
            d2, r2 = _stratum_monthly(pred, a, b, strata[s2], h)
            if not d1.index.equals(d2.index):
                raise AssertionError(f"{axis} h{h}: month sets differ between strata")
            x = d1.values / r1.mean() - d2.values / r2.mean()
            stat, p = diebold_mariano(x, np.zeros_like(x), horizon=h)
            rows.append({"axis": axis, "challenger": a, "reference": b, "horizon": h,
                         "n_months": len(x), "mean_diff_pp": 100 * x.mean(),
                         "dm_stat": stat, "dm_p": p})
    return pd.DataFrame(rows)


def main() -> None:
    diag = pd.read_csv(OUT / "resolution_diagnostics.csv")
    strata = build_strata(diag)
    for k, v in strata.items():
        print(f"{k}: {len(v)} basins")

    p8 = pd.concat([pd.read_csv(OUT / f"{t}_predictions.csv",
                                parse_dates=["issue_date", "target_date"])
                    for t in ("phase8_lstm_combined", "phase8b_lstm_h46")], ignore_index=True)
    p8 = add_seed_ensembles(p8, ["lstmres_corr_top1", "lstm_own_era5"])
    p8_contrasts = [
        ("lstmres_corr_top1_ens", "lstm_own_era5_ens"),   # the headline stacked correction
        ("lstmres_corr_top1_s0", "lstm_own_era5_s0"),
        ("lstmres_corr_top1_s1", "lstm_own_era5_s1"),
        ("ridge_corr_top1_era5", "ridge_own_era5"),       # linear extension to h1-6
    ]
    rows = run_contrasts(p8, p8_contrasts, strata, list(range(1, 7)), "phase8")

    p7 = pd.read_csv(OUT / "phase7_resmlp_predictions.csv",
                     parse_dates=["issue_date", "target_date"])
    p7 = add_seed_ensembles(p7, ["resmlp_corr_top1_era5", "resmlp_own_era5"])
    rows += run_contrasts(p7, [("resmlp_corr_top1_era5_ens", "resmlp_own_era5_ens")],
                          strata, [1, 2, 3], "phase7_resmlp")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase8_stratification.csv", index=False)

    inter = interaction_tests(p8, strata)
    inter.to_csv(OUT / "phase8_stratification_interactions.csv", index=False)
    print("\n=== stratum interaction tests (negative = first stratum gains more) ===")
    print(inter.round(4).to_string(index=False))

    hl = df[(df["challenger"] == "lstmres_corr_top1_ens")]
    for stratum in ["all", "cont_tercile_high", "cont_tercile_mid", "cont_tercile_low",
                    "resolved_x_cont_lowmid", "resolved_x_cont_high"]:
        s = hl[hl["stratum"] == stratum]
        print(f"\n=== lstmres_corr_top1_ens vs lstm_own_era5_ens | {stratum} "
              f"(n={s['n_basins'].iloc[0]}) ===")
        print(s[["horizon", "skill_pct", "ci_lo_pct", "ci_hi_pct", "dm_p"]]
              .round(4).to_string(index=False))

    lin = df[df["challenger"] == "ridge_corr_top1_era5"]
    print("\n=== linear ridge twin, cont_tercile_high vs rest (h1-6) ===")
    for stratum in ["cont_tercile_high", "cont_tercile_mid", "cont_tercile_low"]:
        s = lin[lin["stratum"] == stratum]
        print(f"{stratum}: " + " | ".join(
            f"h{int(r.horizon)} {r.skill_pct:+.2f}% p={r.dm_p:.3g}" for r in s.itertuples()))


if __name__ == "__main__":
    main()
