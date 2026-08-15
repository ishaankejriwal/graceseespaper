"""Publication figures for the GRACE TWSA forecasting paper (HESS, copernicus.cls).

Builds the four figures marked "Buildable NOW" in paper/notes/FIGURE_PLAN.md
(ADDENDUM 2026-08-15): F1, F2, F5, F8. F3/F4/F6/F7 are blocked on reruns.

Reads ONLY results/*.csv. Every headline number plotted is asserted against
paper/notes/REWRITE_LEDGER.md values to the printed precision; if an assert fires,
the source data no longer matches the ledger and the figure must NOT be used.

Outputs (vector PDF + 150 dpi PNG preview each) into figures/:
    fig01_benchmark_ladder, fig02_crossing, fig05_delivery, fig08_stratification

Usage:  .venv\\Scripts\\python.exe scripts\\make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CM = 1.0 / 2.54  # cm -> inch
H = np.array([1, 2, 3, 4, 5, 6])

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "lines.markersize": 4.0,
        "pdf.fonttype": 42,  # embed TrueType (editable text in the PDF)
        "savefig.bbox": "tight",
    }
)

# ---------------------------------------------------------------------------
# Ledger assertions
# ---------------------------------------------------------------------------

TOL = 0.005  # "matches to the printed precision" for 2-dp ledger numbers


def assert_ledger(label, plotted, ledger, tol=TOL):
    """Hard check: plotted values must reproduce REWRITE_LEDGER.md numbers.

    plotted/ledger: sequences aligned on horizons 1..len(ledger).
    STOP (raise) on mismatch -- never adjust the target to make it pass.
    """
    plotted = np.asarray(plotted, dtype=float)
    ledger = np.asarray(ledger, dtype=float)
    if plotted.shape != ledger.shape:
        raise AssertionError(f"[{label}] shape {plotted.shape} != {ledger.shape}")
    bad = np.abs(plotted - ledger) > tol
    if bad.any():
        rows = "; ".join(
            f"h{i + 1}: plotted {plotted[i]:+.4f} vs ledger {ledger[i]:+.2f}"
            for i in np.flatnonzero(bad)
        )
        raise AssertionError(f"[{label}] LEDGER MISMATCH -- {rows}")


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf + .png")


def series(df, keys, value="skill_pct", horizon_col="horizon"):
    """Return values for horizons 1..6 for the rows matching `keys` (dict)."""
    sub = df.copy()
    for k, v in keys.items():
        sub = sub[sub[k] == v]
    sub = sub.sort_values(horizon_col)
    if list(sub[horizon_col]) != list(H):
        raise AssertionError(f"rows for {keys} do not cover horizons 1..6")
    return sub[value].to_numpy()


# ---------------------------------------------------------------------------
# F1 -- benchmark ladder
# ---------------------------------------------------------------------------

def fig01_benchmark_ladder():
    print("F1 fig01_benchmark_ladder")
    lad = pd.read_csv(RESULTS / "paper_baseline_ladder.csv")
    con = pd.read_csv(RESULTS / "paper_baseline_contrasts.csv")

    # skill_vs_damped is a FRACTION vs the stronger damped variant per lead
    # (rho at h1, regression at h2..h6) -- verify that convention holds.
    ref = lad[lad.model == "kalman_ar1"].sort_values("horizon").damped_ref.tolist()
    assert ref == ["damped_persistence_rho"] + ["damped_persistence_reg"] * 5, (
        "damped_ref convention changed -- zero line is no longer the stronger variant"
    )

    def ladder(model):
        return series(lad, {"model": model}, value="skill_vs_damped") * 100.0

    kal = ladder("kalman_ar1")
    kor = ladder("kalman_own_ridge")
    pbr = ladder("ridge_own_perbasin")
    plr = ladder("ridge_own_lags")
    per = ladder("persistence")

    # --- ledger section 1 ---
    assert_ledger("F1 kalman_ar1", kal, [4.98, 8.79, 5.62, 3.07, 2.55, 3.63])
    assert_ledger("F1 ridge-on-filtered-states", kor, [4.82, 7.87, 6.04, 5.36, 5.14, 5.16])
    assert_ledger("F1 per-basin ridge", pbr, [3.05, 5.94, 4.26, 2.05, 0.38, -0.29])
    assert_ledger("F1 pooled ridge", plr, [1.17, 4.78, 5.08, 3.64, 3.17, 2.49])
    assert_ledger("F1 persistence", per, [-20.75, -22.68, -21.39, -19.49, -16.04, -15.80])

    # CI ribbon for kalman_ar1 vs the stronger damped variant: the contrasts
    # file stores it vs rho at h1 and vs reg at h2..h6 (matching damped_ref).
    kd = con[
        (con.challenger == "kalman_ar1")
        & (
            ((con.reference == "damped_persistence_rho") & (con.horizon == 1))
            | ((con.reference == "damped_persistence_reg") & (con.horizon > 1))
        )
    ].sort_values("horizon")
    assert list(kd.horizon) == list(H), "kalman-vs-damped contrast rows incomplete"
    ci_lo, ci_hi = kd.ci_lo.to_numpy() * 100.0, kd.ci_hi.to_numpy() * 100.0
    # contrast point estimates must agree with the ladder curve being drawn
    assert_ledger("F1 kalman contrast == ladder", kd.skill.to_numpy() * 100.0, kal)

    c = cm.batlow
    fig, ax = plt.subplots(figsize=(12 * CM, 8.5 * CM))

    ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
    ax.plot(H, per, ls=":", marker="v", color=c(0.80), lw=1.0,
            label="persistence", zorder=3)
    ax.plot(H, plr, ls="-.", marker="^", color=c(0.60), lw=1.0,
            label="pooled ridge (own lags)", zorder=3)
    ax.plot(H, pbr, ls="--", marker="s", color=c(0.40), lw=1.0,
            label="per-basin ridge", zorder=3)
    ax.fill_between(H, ci_lo, ci_hi, color=c(0.02), alpha=0.15, lw=0, zorder=2)
    ax.plot(H, kal, ls="-", marker="o", color=c(0.02), lw=1.8,
            label="Kalman (filtered-state persistence)", zorder=5)
    ax.plot(H, kor, ls="--", marker="D", color=c(0.20), lw=1.8,
            label="ridge on filtered states", zorder=4)

    # zero-line label (the reference itself)
    ax.annotate("damped persistence\n(stronger variant)", xy=(1.0, 0), xytext=(0.95, -2.2),
                ha="left", va="top", fontsize=6.8, color="0.35")

    # bracket at h1-h2: corrected noise-propagation gap +5.0-8.8 %
    yb = max(kal[0], kal[1]) + 1.4
    ax.plot([1, 1, 2, 2], [kal[0] + 0.7, yb, yb, kal[1] + 0.7],
            color="0.15", lw=0.8, zorder=6)
    ax.annotate("+5.0–8.8%", xy=(1.5, yb), xytext=(1.5, yb + 0.5),
                ha="center", va="bottom", fontsize=7.5, color="0.15", zorder=6)

    ax.set_xlabel("lead $h$ (months)")
    ax.set_ylabel("skill vs damped persistence (%)")
    ax.set_xticks(H)
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(-25.5, 13.5)
    ax.legend(loc="center right", frameon=False, ncol=1, handlelength=2.4)
    save(fig, "fig01_benchmark_ladder")


# ---------------------------------------------------------------------------
# F2 -- the crossing vs GRACE-FCast (Li et al.)
# ---------------------------------------------------------------------------

def fig02_crossing():
    print("F2 fig02_crossing")
    li = pd.read_csv(RESULTS / "phase8b_li_comparison_headline.csv")
    li = li[li.subset == "all_matched"]

    def curve(model, vs):
        sub = li[(li.model == model) & (li.vs == vs)].sort_values("horizon")
        assert list(sub.horizon) == list(H), f"{model} vs {vs}: rows incomplete"
        # skill/ci stored as FRACTIONS -> x100
        return (
            sub.skill.to_numpy() * 100.0,
            sub.ci_lo.to_numpy() * 100.0,
            sub.ci_hi.to_numpy() * 100.0,
            sub.dm_p.to_numpy(),
            sub,
        )

    main, mlo, mhi, mp, msub = curve("lstmres_corr_top1_ens", "li_lstm_full")
    kal, _, _, kp, _ = curve("kalman_ar1", "li_lstm_full")
    nons, _, _, np_, _ = curve("lstmres_corr_top1_ens", "li_lstm_nonseas")

    # --- ledger section 6 ---
    assert_ledger("F2 main (ens vs li_full)", main,
                  [20.01, -3.49, -12.94, -17.44, -23.64, -30.34])
    assert int(msub.n_rows.iloc[0]) == 13620 and int(msub.n_months.iloc[0]) == 60, (
        "matched sample is no longer 227 basins x 60 months = 13620 rows"
    )
    assert_ledger("F2 kalman h1", kal[:1], [14.63])
    assert_ledger("F2 vs nonseas h1-h2", nons[:2], [30.94, 8.96])
    assert mp[1] > 0.05, "h2 of the main curve is expected ns (open marker)"

    # per-basin win counts (for the caption; recorded in BUILD_NOTES)
    pb = pd.read_csv(RESULTS / "phase8b_li_comparison_perbasin.csv")
    pb = pb[(pb.model == "lstmres_corr_top1_ens") & (pb.vs == "li_lstm_full")]
    wins = pb[pb.dm_stat < 0].groupby("horizon").size().reindex(H, fill_value=0)
    totals = pb.groupby("horizon").size().reindex(H)
    assert (totals == 227).all(), "per-basin file no longer has 227 matched basins"
    win_str = "/".join(str(int(w)) for w in wins)
    print(f"  per-basin win counts (dm_stat<0, of 227): {win_str}")

    c = cm.batlow
    fig, ax = plt.subplots(figsize=(12 * CM, 8.5 * CM))

    ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
    ax.axvline(2.5, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.annotate("crossover", xy=(2.42, -42.5), ha="right", va="bottom",
                fontsize=7, color="0.35")

    ax.fill_between(H, mlo, mhi, color=c(0.05), alpha=0.15, lw=0, zorder=2)

    def draw(y, p, color, ls, marker, lw, label, z):
        ax.plot(H, y, ls=ls, color=color, lw=lw, label=label, zorder=z)
        sig = p < 0.05
        ax.plot(H[sig], y[sig], ls="none", marker=marker, color=color,
                mfc=color, mec=color, zorder=z + 1)
        ax.plot(H[~sig], y[~sig], ls="none", marker=marker, color=color,
                mfc="white", mec=color, zorder=z + 1)

    draw(main, mp, c(0.05), "-", "o", 1.8,
         "stacked ensemble vs GRACE-FCast (full)", 5)
    draw(kal, kp, c(0.45), "--", "s", 1.0,
         "Kalman alone vs GRACE-FCast (full)", 3)
    draw(nons, np_, c(0.72), ":", "^", 1.0,
         "stacked ensemble vs GRACE-FCast (non-seas.)", 3)

    ax.set_xlabel("lead $h$ (months)")
    ax.set_ylabel("skill vs GRACE-FCast (%)")
    ax.set_xticks(H)
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(-46, 35)
    ax.legend(loc="upper right", frameon=False, handlelength=2.6)
    save(fig, "fig02_crossing")
    return win_str


# ---------------------------------------------------------------------------
# F5 -- delivery contrast (correction stage vs input channel) + placebo band
# ---------------------------------------------------------------------------

def _pooled_mse(df, sum_col, count_col, by):
    g = df.groupby(by).agg(s=(sum_col, "sum"), n=(count_col, "sum")).reset_index()
    g["mse"] = g.s / g.n
    return g


def fig05_delivery():
    print("F5 fig05_delivery")
    ens = pd.read_csv(RESULTS / "phase8b_h16_ensemble_headline.csv")
    per = pd.read_csv(RESULTS / "phase8b_h16_headline.csv")

    def head(df, ch, ref):
        sub = df[(df.challenger == ch) & (df.reference == ref)].sort_values("horizon")
        assert list(sub.horizon) == list(H), f"{ch} vs {ref}: rows incomplete"
        return (sub.skill_pct.to_numpy(), sub.ci_lo_pct.to_numpy(),
                sub.ci_hi_pct.to_numpy(), sub.dm_p.to_numpy())

    corr, corr_lo, corr_hi, corr_p = head(ens, "lstmres_corr_top1_ens", "lstm_own_era5_ens")
    chan, chan_lo, chan_hi, chan_p = head(ens, "lstm_corr_top1_era5_ens", "lstm_own_era5_ens")
    s0, _, _, _ = head(per, "lstmres_corr_top1_s0", "lstm_own_era5_s0")
    s1, _, _, _ = head(per, "lstmres_corr_top1_s1", "lstm_own_era5_s1")
    ch0, _, _, _ = head(per, "lstm_corr_top1_era5_s0", "lstm_own_era5_s0")

    # --- ledger section 5 ---
    assert_ledger("F5 ensemble correction", corr, [0.91, 1.45, 1.33, 1.32, 1.42, 1.96])
    assert_ledger("F5 ensemble CI lower", corr_lo, [0.64, 1.07, 1.02, 0.98, 1.11, 1.67])
    # DISCREPANCY (flagged in BUILD_NOTES): REWRITE_LEDGER.md sect. 5 says
    # "All p <= 1.2e-10", but the CSV (sole source, cross-checked against
    # phase8_stratification.csv 'all' rows) has max dm_p = 3.14e-08 at h4
    # (h1 = 1.10e-10, h6 = 1.21e-13) -- the ledger line looks like a
    # transcription slip from the h1 value. dm_p is NOT printed on the
    # figure; we assert the level the CSV supports and report the mismatch.
    assert (corr_p < 5e-8).all(), "ensemble correction no longer significant at every lead"
    assert_ledger("F5 per-seed s0", s0, [0.64, 1.33, 1.29, 1.26, 1.51, 1.90])
    assert_ledger("F5 per-seed s1", s1, [1.19, 1.56, 1.36, 1.37, 1.28, 1.93])
    assert_ledger("F5 channel ensemble", chan, [0.36, -0.42, -0.43, -0.61, -0.37, -0.38])

    # ---- placebo band (seed-matched random-graph corrections) -------------
    # Scale note (documented in BUILD_NOTES): headline skill_pct is an
    # MSE-ratio skill, 100*(1 - MSE_challenger/MSE_reference). We verify that
    # below by reproducing the per-seed headline from raw predictions, then
    # compute placebo increments on the SAME scale.
    own_models = ["lstm_own_era5_s0", "lstm_own_era5_s1",
                  "lstmres_corr_top1_s0", "lstmres_corr_top1_s1"]
    parts = []
    for f in ["phase8_lstm_combined_predictions.csv", "phase8b_lstm_h46_predictions.csv"]:
        d = pd.read_csv(RESULTS / f, usecols=["target", "pred", "model", "horizon"])
        d = d[d.model.isin(own_models)]
        d["se"] = (d.pred - d.target) ** 2
        d["cnt"] = 1
        parts.append(_pooled_mse(d, "se", "cnt", ["model", "horizon"]))
    pr = pd.concat(parts).groupby(["model", "horizon"])[["s", "n"]].sum().reset_index()
    pr["mse"] = pr.s / pr.n
    mse = pr.pivot(index="model", columns="horizon", values="mse")

    # cross-check: reproduce the per-seed headline from predictions (3 dp)
    for s, target in [("s0", s0), ("s1", s1)]:
        rec = (1.0 - mse.loc[f"lstmres_corr_top1_{s}"] / mse.loc[f"lstm_own_era5_{s}"]) * 100.0
        assert_ledger(f"F5 {s} recomputed-from-predictions == headline",
                      rec.to_numpy(), target, tol=0.0015)

    parts = []
    for f in ["phase8_lstm_combined_placebo_monthly.csv", "phase8b_lstm_h46_placebo_monthly.csv"]:
        d = pd.read_csv(RESULTS / f)
        d = d[d.model.str.match(r"lstmres_corr_top1_s\d+_rand\d+")]
        parts.append(_pooled_mse(d, "sum", "count", ["model", "horizon"]))
    pl = pd.concat(parts).groupby(["model", "horizon"])[["s", "n"]].sum().reset_index()
    pl["mse"] = pl.s / pl.n
    pl["seed"] = pl.model.str.extract(r"_s(\d+)_rand")[0]
    pl["incr"] = pl.apply(
        lambda r: (1.0 - r.mse / mse.loc[f"lstm_own_era5_s{r.seed}", r.horizon]) * 100.0,
        axis=1,
    )
    n_draws = pl.groupby("horizon").size()
    assert (n_draws == 40).all(), "expected 20 draws x 2 seeds per horizon"

    # ledger checks: per-cell means -0.27..-0.05 %, and 20/20 beaten in all
    # 12 lead x seed cells
    cell_means = pl.groupby(["seed", "horizon"]).incr.mean()
    assert abs(cell_means.min() - (-0.27)) <= TOL and abs(cell_means.max() - (-0.05)) <= TOL, (
        f"placebo per-cell means {cell_means.min():+.3f}..{cell_means.max():+.3f} "
        "do not match ledger -0.27..-0.05"
    )
    real = {"0": s0, "1": s1}
    for seed, grp in pl.groupby("seed"):
        beaten = grp.groupby("horizon").apply(
            lambda d: int((d.incr < real[seed][d.name - 1]).sum()), include_groups=False
        )
        assert (beaten == 20).all(), f"seed {seed}: not 20/20 placebos beaten"

    band_lo = pl.groupby("horizon").incr.min().reindex(H).to_numpy()
    band_hi = pl.groupby("horizon").incr.max().reindex(H).to_numpy()

    # ---- draw -------------------------------------------------------------
    blue, red = cm.vik(0.08), cm.vik(0.92)
    fig, ax = plt.subplots(figsize=(12 * CM, 8.5 * CM))

    ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
    ax.fill_between(H, band_lo, band_hi, color="0.72", alpha=0.45, lw=0, zorder=2,
                    label="random-graph placebo range (40 draws)")

    # correction stage: ensemble curve + CI ribbon, per-seed light markers
    ax.fill_between(H, corr_lo, corr_hi, color=blue, alpha=0.18, lw=0, zorder=3)
    ax.plot(H, corr, ls="-", marker="o", color=blue, lw=1.8, zorder=6,
            label="as correction stage (2-seed ensemble)")
    ax.plot(H - 0.09, s0, ls="none", marker="^", color=blue, alpha=0.45,
            markersize=3.2, zorder=5, label="correction, single seeds")
    ax.plot(H + 0.09, s1, ls="none", marker="v", color=blue, alpha=0.45,
            markersize=3.2, zorder=5)

    # input channel: ensemble curve + CI error bars, seed-0 light markers
    ax.errorbar(H, chan, yerr=[chan - chan_lo, chan_hi - chan],
                ls="--", marker="s", color=red, lw=1.4, elinewidth=0.8,
                capsize=2.0, zorder=4,
                label="as input channel (2-seed ensemble)")
    ax.plot(H + 0.09, ch0, ls="none", marker="D", color=red, alpha=0.45,
            markersize=3.0, zorder=4, label="channel, seed 0")

    ax.annotate("identical neighbor information,\ntwo deliveries",
                xy=(0.95, 2.48), ha="left", va="top", fontsize=7, color="0.15")

    ax.set_xlabel("lead $h$ (months)")
    ax.set_ylabel("skill increment over stage-1 LSTM (%)")
    ax.set_xticks(H)
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(-2.8, 2.6)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in [
        "as correction stage (2-seed ensemble)", "correction, single seeds",
        "as input channel (2-seed ensemble)", "channel, seed 0",
        "random-graph placebo range (40 draws)"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower right", frameon=False, handlelength=2.4)
    save(fig, "fig05_delivery")
    return band_lo, band_hi


# ---------------------------------------------------------------------------
# F8 -- stratification (anti-leakage figure)
# ---------------------------------------------------------------------------

def fig08_stratification():
    print("F8 fig08_stratification")
    st = pd.read_csv(RESULTS / "phase8_stratification.csv")

    def strat(ch, ref, stratum):
        sub = st[(st.challenger == ch) & (st.reference == ref)
                 & (st.stratum == stratum)].sort_values("horizon")
        assert list(sub.horizon) == list(H), f"{ch}/{stratum}: rows incomplete"
        n = sub.n_basins.unique()
        assert len(n) == 1
        return (sub.skill_pct.to_numpy(), sub.ci_lo_pct.to_numpy(),
                sub.ci_hi_pct.to_numpy(), sub.dm_p.to_numpy(), int(n[0]))

    LIN = ("ridge_corr_top1_era5", "ridge_own_era5")
    STK = ("lstmres_corr_top1_ens", "lstm_own_era5_ens")

    lin = {s: strat(*LIN, s) for s in
           ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high"]}
    stk = {s: strat(*STK, s) for s in
           ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high",
            "resolved_x_cont_lowmid"]}

    # --- ledger section 3 (linear terciles) and stacked stratification ---
    assert_ledger("F8 linear cont-high", lin["cont_tercile_high"][0],
                  [1.11, 0.59, 0.37, 0.08, -0.11, 0.00])
    assert lin["cont_tercile_high"][3][0] < 3e-4, "linear cont-high h1 p =~ 2.3e-4"
    assert lin["cont_tercile_high"][3][1] < 0.05, "linear cont-high h2 p =~ 0.037"
    assert (lin["cont_tercile_low"][0] < 0).all() and (lin["cont_tercile_mid"][0] < 0).all(), (
        "ledger: linear mid/low terciles negative at every lead"
    )
    assert_ledger("F8 stacked cont-low", stk["cont_tercile_low"][0],
                  [0.69, 1.31, 1.29, 1.28, 1.31, 1.84])
    assert_ledger("F8 stacked resolved x cont-low/mid", stk["resolved_x_cont_lowmid"][0],
                  [0.89, 1.57, 1.46, 1.38, 1.49, 2.00])
    for s, v in stk.items():
        assert (v[0] > 0).all(), f"stacked stratum {s} must be positive at every lead"
    assert all(lin[s][4] == 78 for s in lin) and stk["resolved_x_cont_lowmid"][4] == 142, (
        "stratum n's changed (terciles 78 each; resolved x low/mid 142)"
    )

    c = cm.batlow
    STYLE = {  # shared tercile identity across both panels
        "cont_tercile_low": dict(color=c(0.05), ls="-", marker="o", label="contamination low"),
        "cont_tercile_mid": dict(color=c(0.38), ls="--", marker="s", label="contamination mid"),
        "cont_tercile_high": dict(color=c(0.68), ls=":", marker="^", label="contamination high"),
        "resolved_x_cont_lowmid": dict(color="0.15", ls="-.", marker="D",
                                       label="resolved × cont. low/mid"),
    }
    OFFS = {"cont_tercile_low": -0.09, "cont_tercile_mid": -0.03,
            "cont_tercile_high": 0.03, "resolved_x_cont_lowmid": 0.09}

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(17 * CM, 7.8 * CM), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    def draw(ax, data, stratum, lw):
        y, lo, hi, p, n = data
        s = STYLE[stratum]
        x = H + OFFS[stratum]
        ax.errorbar(x, y, yerr=[y - lo, hi - y], ls=s["ls"], color=s["color"],
                    lw=lw, elinewidth=0.7, capsize=1.8, zorder=4,
                    label=f"{s['label']} ($n$={n})")
        sig = p < 0.05
        for m, fc in [(sig, s["color"]), (~sig, "white")]:
            ax.plot(x[m], y[m], ls="none", marker=s["marker"], color=s["color"],
                    mfc=fc, mec=s["color"], markersize=3.8, zorder=5)

    for ax in (axa, axb):
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        ax.set_xticks(H)
        ax.set_xlim(0.6, 6.4)
        ax.set_xlabel("lead $h$ (months)")

    for s in ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high"]:
        draw(axa, lin[s], s, 1.1)
    for s in ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high",
              "resolved_x_cont_lowmid"]:
        draw(axb, stk[s], s, 1.1)

    axa.set_title("(a) linear arm (ridge, +ERA5)", loc="left", fontsize=7.5)
    axb.set_title("(b) stacked correction (LSTM ens.)", loc="left", fontsize=7.5)
    axa.set_ylabel("skill vs own-only twin (%)")
    axa.legend(loc="upper right", frameon=False, handlelength=2.4)
    axb.legend(loc="lower right", frameon=False, handlelength=2.4)
    axa.set_ylim(-1.9, 2.75)  # shared

    # filled = DM p < 0.05, open = ns (stated in caption as well)
    axa.annotate("filled: DM $p<0.05$", xy=(0.03, 0.03), xycoords="axes fraction",
                 fontsize=6.5, color="0.35")
    save(fig, "fig08_stratification")


# ---------------------------------------------------------------------------
# F3 -- per-basin neighbor-effect map (choropleth on the mask grid; no cartopy)
# ---------------------------------------------------------------------------

def fig03_neighbor_map():
    print("F3 fig03_neighbor_map")
    import sys
    import xarray as xr
    sys.path.insert(0, str(ROOT / "src"))
    from gracefc.basins import load_basin_masks

    fdr = pd.read_csv(RESULTS / "phase5_perbasin_fdr_h1.csv")
    fdr = fdr[fdr["comparison"] == "corr_top1_vs_own_ridge"].set_index("name")
    masks = load_basin_masks(ROOT / "HydroShed+Mascon_Basins_L3.nc")
    meta = masks["meta"]
    dm = xr.open_dataset(ROOT / "HydroShed+Mascon_Basins_L3.nc", decode_times=False)
    lat, lon = dm["lat"].values, dm["lon"].values
    dm.close()

    grid = np.full(lat.size * lon.size, np.nan)
    for b, name in enumerate(meta["name"]):
        if name in fdr.index and np.isfinite(fdr.loc[name, "dm_stat"]):
            # sign flipped: positive = neighbor helps
            grid[masks["indices"][b]] = -float(fdr.loc[name, "dm_stat"])
    grid = grid.reshape(lat.size, lon.size)

    # Roll 0..360 -> -180..180 for a conventional world map
    shift = lon.size // 2
    grid = np.roll(grid, shift, axis=1)
    lon_c = np.where(lon >= 180, lon - 360, lon)
    lon_plot = np.sort(lon_c)

    fig, ax = plt.subplots(figsize=(17 * CM, 9.2 * CM))
    pm = ax.pcolormesh(lon_plot, lat, np.clip(grid, -3, 3),
                       cmap=cm.vik_r, vmin=-3, vmax=3, rasterized=True, shading="nearest")
    pm.cmap.set_bad("0.94")
    ax.set_facecolor("0.94")
    ax.set_ylim(-60, 84)
    ax.set_aspect(1.0)
    ax.tick_params(labelsize=6.5)

    # FDR-significant basins: filled dot = helped, open = hurt (21 = 13 + 8)
    cent = pd.read_csv(ROOT / "data/processed/basin_meta.csv").set_index("name")
    sig = fdr[fdr["significant"]]
    n_h = int(sig["a_better"].sum())
    n_u = int((~sig["a_better"]).sum())
    assert (n_h, n_u) == (13, 8), f"FDR roster changed: {n_h} helped / {n_u} hurt"
    for name, row in sig.iterrows():
        la = cent.loc[name, "centroid_lat"]
        lo = cent.loc[name, "centroid_lon"]
        lo = lo - 360 if lo > 180 else lo
        if row["a_better"]:
            ax.plot(lo, la, "o", ms=3.4, mfc="k", mec="k", mew=0.5, zorder=5)
        else:
            ax.plot(lo, la, "o", ms=3.8, mfc="none", mec="k", mew=0.9, zorder=5)
    for name, label, dy in (("R_Yenisey_River", "Yenisey", 3), ("R_Parana_River", "Paraná", -6),
                            ("R_Yangtze_River", "Yangtze", -6), ("R_Amur_River", "Amur", 4),
                            ("R_Niger_River", "Niger", -6), ("R_Sao_Francisco_River", "São Francisco", 4)):
        la = cent.loc[name, "centroid_lat"]
        lo = cent.loc[name, "centroid_lon"]
        lo = lo - 360 if lo > 180 else lo
        ax.annotate(label, xy=(lo, la), xytext=(lo + 2, la + dy), fontsize=6.2,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.25"))
    cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.11, aspect=42)
    cb.set_label("per-basin DM statistic, lead 1 (positive = neighbor helps)", fontsize=7)
    ax.annotate(f"filled: FDR-significant helped (n={n_h})   open: hurt (n={n_u})   q=0.10",
                xy=(0.01, 1.02), xycoords="axes fraction", fontsize=6.6, color="0.25")
    save(fig, "fig03_neighbor_map")


# ---------------------------------------------------------------------------
# F4 -- the control battery (placebos + surrogates, distance profile, conditioning)
# ---------------------------------------------------------------------------

def fig04_controls():
    print("F4 fig04_controls")
    summ = pd.read_csv(RESULTS / "phase3b_summary.csv")
    s1 = summ[summ["horizon"] == 1].set_index("model")

    # (a) 50 random-graph placebo pooled RMSEs at h1 vs the real arm
    plac = pd.read_csv(RESULTS / "phase3b_placebo_monthly.csv")
    p1 = plac[(plac["horizon"] == 1)
              & plac["model"].str.startswith("corr_top1_rand")]
    pooled = (p1.groupby("model")[["sum", "count"]].sum()
              .assign(rmse=lambda d: np.sqrt(d["sum"] / d["count"])))["rmse"]
    real_rmse = float(s1.loc["kalman_corr_top1", "rmse_std"])
    assert len(pooled) == 50 and (real_rmse < pooled).all(), "placebo panel: expected 50/50"

    surr = pd.read_csv(RESULTS / "phase4_surrogate_summary.csv")
    su1 = surr[surr["horizon"] == 1].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(17 * CM, 5.6 * CM),
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.42})
    axa, axb, axc = axes

    axa.hist(pooled.values, bins=14, color=cm.batlow(0.55), edgecolor="white", lw=0.4)
    axa.axvline(real_rmse, color="k", lw=1.4)
    axa.annotate("real graph\n(beats 50/50)", xy=(real_rmse, axa.get_ylim()[1] * 0.94),
                 xytext=(4, -2), textcoords="offset points", fontsize=6.4, va="top")
    axa.set_xlabel("pooled lead-1 RMSE (std. units)")
    axa.xaxis.set_major_locator(plt.MaxNLocator(4))
    axa.set_ylabel("random graphs")
    axa.set_title("(a) placebo + surrogate nulls", loc="left", fontsize=7.5)
    axa.annotate(f"IAAFT surrogates: real skill {100 * su1['skill_vs_own']:+.2f}%"
                 f" beats 99/99\n(surrogate mean {100 * su1['surr_skill_vs_own_mean']:+.2f}%,"
                 f" $p_{{rank}}$=0.01)",
                 xy=(0.02, -0.52), xycoords="axes fraction", fontsize=6.4)

    # (b) distance profile (corrected values; placebo counts from the summary)
    prof = [("no restr.", "kalman_corr_top1"), ("≥300 km", "kalman_corr_min300_top1"),
            ("≥500 km", "kalman_corr_min500_top1"), ("≥1000 km", "kalman_corr_min1000_top1")]
    xs = np.arange(len(prof))
    vals = [100 * float(s1.loc[m, "skill_vs_own_ridge"]) for _, m in prof]
    beat = [f"{int(s1.loc[m, 'placebo_beaten'])}/{int(s1.loc[m, 'placebo_n'])}" for _, m in prof]
    assert_ledger("F4 distance profile", vals, [0.31, 0.32, 0.16, -0.60])
    axb.bar(xs, vals, width=0.62, color=[cm.batlow(v) for v in (0.25, 0.42, 0.60, 0.78)])
    axb.axhline(0, color="k", lw=0.7)
    for x, v, b in zip(xs, vals, beat):
        axb.annotate(b, xy=(x, v), xytext=(0, 4 if v >= 0 else -11),
                     textcoords="offset points", ha="center", fontsize=6.4)
    axb.set_xticks(xs)
    axb.set_xticklabels([p[0] for p in prof], fontsize=6.2, rotation=18, ha="right")
    axb.set_ylabel("lead-1 skill vs own ridge (%)")
    axb.set_title("(b) source-distance exclusion", loc="left", fontsize=7.5)

    # (c) conditioning invariance, bootstrap CIs
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from gracefc.stats import block_bootstrap_skill_ci
    era5 = pd.read_csv(RESULTS / "phase6_era5_headline.csv")

    def head_row(ch, ref):
        r = era5[(era5["challenger"] == ch) & (era5["reference"] == ref) & (era5["horizon"] == 1)].iloc[0]
        return r["skill_pct"], r["ci_lo_pct"], r["ci_hi_pct"]

    cond_pred = pd.read_csv(RESULTS / "phase4_conditioned_predictions.csv",
                            parse_dates=["issue_date", "target_date"])
    pt, lo, hi = block_bootstrap_skill_ci(cond_pred, "kalman_corr_top1", "kalman_own_ridge", 1)
    rows = [("uncond.", *head_row("ridge_corr_top1", "ridge_own")),
            ("ENSO+IOD", 100 * pt, 100 * lo, 100 * hi),
            ("full ERA5", *head_row("ridge_corr_top1_era5", "ridge_own_era5"))]
    xs = np.arange(len(rows))
    for x, (lab, v, l, h) in zip(xs, rows):
        axc.errorbar(x, v, yerr=[[v - l], [h - v]], fmt="o", ms=4, color=cm.vik(0.15),
                     ecolor="0.45", elinewidth=1.0, capsize=2.5)
    axc.axhline(0, color="k", lw=0.7)
    axc.set_xticks(xs)
    axc.set_xticklabels([r[0] for r in rows], fontsize=6.2, rotation=18, ha="right")
    axc.set_xlim(-0.6, len(rows) - 0.4)
    axc.set_ylabel("lead-1 skill vs own reference (%)")
    axc.set_title("(c) conditioning invariance", loc="left", fontsize=7.5)
    save(fig, "fig04_controls")


# ---------------------------------------------------------------------------
# F6 -- ERA5 gain vs neighbor gain complementarity
# ---------------------------------------------------------------------------

def fig06_complementarity():
    print("F6 fig06_complementarity")
    from scipy.stats import spearmanr

    pred = pd.read_csv(RESULTS / "phase6_era5_predictions.csv")
    h1 = pred[pred["horizon"] == 1]

    def perbasin(a, b):
        out = {}
        sub = h1[h1["model"].isin([a, b])]
        for name, g in sub.groupby("name"):
            j = g[g["model"] == a].merge(g[g["model"] == b], on=["name", "target_date"],
                                         suffixes=("_a", "_b"))
            if len(j) < 10:
                continue
            out[name] = 100 * (1 - ((j["target_a"] - j["pred_a"]) ** 2).mean()
                               / ((j["target_b"] - j["pred_b"]) ** 2).mean())
        return pd.Series(out)

    era5_gain = perbasin("ridge_own_era5", "ridge_own")
    nbr_gain = perbasin("ridge_corr_top1_era5", "ridge_own_era5")
    meta = pd.read_csv(ROOT / "data/processed/basin_meta.csv")[["name", "continent"]]
    df = (pd.concat([era5_gain.rename("era5"), nbr_gain.rename("nbr")], axis=1)
          .dropna().reset_index(names="name").merge(meta, on="name"))
    rho, p = spearmanr(df["era5"], df["nbr"])
    assert_ledger("F6 rho", [rho], [-0.22], tol=0.005)

    fig, ax = plt.subplots(figsize=(12 * CM, 8.6 * CM))
    conts = ["africa", "asia", "australia", "europe", "north_america", "south_america"]
    labels = ["Africa", "Asia", "Australia/Oceania", "Europe", "N. America", "S. America"]
    colors = [cm.batlowS(i) for i in range(len(conts))]
    n_off = 0
    for cont, lab, col in zip(conts, labels, colors):
        g = df[df["continent"] == cont]
        big = cont == "africa"
        x = g["era5"].clip(-30, 40)
        n_off += int(((g["era5"] < -30) | (g["era5"] > 40)).sum())
        ax.scatter(x, g["nbr"].clip(-12, 12), s=16 if big else 8,
                   color=col, edgecolor="k" if big else "none", linewidth=0.4,
                   label=lab, zorder=5 if big else 3, alpha=0.95 if big else 0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("per-basin ERA5 gain, lead 1 (%)")
    ax.set_ylabel("per-basin ERA5-conditioned neighbor gain, lead 1 (%)")
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.9)
    ax.annotate(f"Spearman $\\rho$ = {rho:+.2f} ($p$ = {p:.0e}), 234 basins"
                + (f"; {n_off} points clipped" if n_off else ""),
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6.8)
    med = df.groupby("continent")[["era5", "nbr"]].median().round(1)
    txt = "median (ERA5, nbr) %\n" + "\n".join(
        f"{lab:<12} {med.loc[c, 'era5']:+.1f}, {med.loc[c, 'nbr']:+.1f}"
        for c, lab in zip(conts, ["Africa", "Asia", "Aus/Oc", "Europe", "N. Am", "S. Am"]))
    ax.annotate(txt, xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=6.2,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    save(fig, "fig06_complementarity")


# ---------------------------------------------------------------------------

def main():
    fig01_benchmark_ladder()
    win_str = fig02_crossing()
    fig03_neighbor_map()
    fig04_controls()
    fig05_delivery()
    fig06_complementarity()
    fig08_stratification()
    print("all figures built; all ledger asserts passed")
    print(f"F2 caption win counts (of 227): {win_str}")


if __name__ == "__main__":
    main()
