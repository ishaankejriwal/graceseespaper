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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.patheffects import withStroke

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CM = 1.0 / 2.54  # cm -> inch
H = np.array([1, 2, 3, 4, 5, 6])

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "sky": "#56B4E9", "purple": "#CC79A7",
      "yellow": "#F0E442", "black": "#000000"}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.linewidth": 0.4, "grid.color": "0.88",
    "lines.linewidth": 1.4, "lines.markersize": 4.0,
    "legend.frameon": False, "legend.handlelength": 2.2,
    "errorbar.capsize": 0,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})

# fixed color semantics (see BUILD_NOTES): Kalman family = OI blue;
# stacked ensemble / correction stage = near-black; ridge-on-filtered-states =
# OI vermillion dashed; per-basin / pooled ridge = OI green / OI sky;
# neighbor-as-input-channel = OI orange dashed; persistence / zero lines =
# gray 0.45; placebo nulls = gray 0.75; Li nonseasonal = OI purple dotted.
NEARBLACK = "0.1"
REFGRAY = "0.45"
NULLGRAY = "0.75"


def tint(color, f):
    """Blend a color toward white by fraction f (0 = unchanged, 1 = white)."""
    c = np.asarray(mcolors.to_rgb(color))
    return tuple(c + (1.0 - c) * f)


def shade(color, f):
    """Blend a color toward black by fraction f."""
    c = np.asarray(mcolors.to_rgb(color))
    return tuple(c * (1.0 - f))


def sig_markers(ax, x, y, p, marker, color, ms=4.0, z=6):
    """Significance convention: filled marker = DM p<0.05, open = ns."""
    x, y, p = np.asarray(x), np.asarray(y), np.asarray(p)
    sig = p < 0.05
    ax.plot(x[sig], y[sig], ls="none", marker=marker, mfc=color, mec=color,
            markersize=ms, zorder=z)
    ax.plot(x[~sig], y[~sig], ls="none", marker=marker, mfc="white", mec=color,
            markersize=ms, zorder=z)


def dm_note(ax, x=0.02, y=0.03):
    ax.annotate("filled: DM $p<0.05$", xy=(x, y), xycoords="axes fraction",
                fontsize=7, color=REFGRAY, zorder=7)


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

    # broken axis: raw persistence lives at -16..-23 %, far below the field
    fig = plt.figure(figsize=(12 * CM, 8.5 * CM))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06,
                          left=0.11, right=0.72, top=0.97, bottom=0.11)
    axt = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=axt)

    axt.axhline(0.0, color=REFGRAY, lw=0.8, zorder=1)
    axt.fill_between(H, ci_lo, ci_hi, color=OI["blue"], alpha=0.15, lw=0, zorder=2)
    axt.plot(H, plr, ls="-.", marker="^", color=OI["sky"], lw=1.2, zorder=3)
    axt.plot(H, pbr, ls="--", marker="s", color=OI["green"], lw=1.2, zorder=3)
    axt.plot(H, kal, ls="-", marker="o", color=OI["blue"], lw=1.5, zorder=5)
    axt.plot(H, kor, ls="--", marker="D", color=OI["vermillion"], lw=1.5, zorder=4)
    axb.plot(H, per, ls=":", marker="v", color=REFGRAY, lw=1.0, zorder=3)

    # direct labels at the right edge (no floating legend)
    for txt, y, col in [
        ("ridge on filtered states", kor[-1], OI["vermillion"]),
        ("Kalman (filtered-state\npersistence)", kal[-1], OI["blue"]),
        ("pooled ridge (own lags)", plr[-1], OI["sky"]),
        ("per-basin ridge", pbr[-1], OI["green"]),
    ]:
        axt.annotate(txt, xy=(6.15, y), ha="left", va="center", fontsize=7,
                     color=col, annotation_clip=False)
    axb.annotate("persistence", xy=(6.15, per[-1]), ha="left", va="center",
                 fontsize=7, color=REFGRAY, annotation_clip=False)

    # zero-line label (the reference itself)
    axt.annotate("damped persistence (stronger variant)", xy=(1.0, 0),
                 xytext=(0.98, -0.35), ha="left", va="top", fontsize=6.5,
                 color=REFGRAY)

    # bracket at h1-h2: corrected noise-propagation gap +5.0-8.8 %
    yb = max(kal[0], kal[1]) + 1.0
    axt.plot([1, 1, 2, 2], [kal[0] + 0.6, yb, yb, kal[1] + 0.6],
             color="0.15", lw=0.8, zorder=6)
    axt.annotate("+5.0–8.8%", xy=(1.5, yb), xytext=(1.5, yb + 0.25),
                 ha="center", va="bottom", fontsize=7, color="0.15", zorder=6)

    # broken-axis cosmetics
    axt.set_ylim(-1.5, 11.0)
    axb.set_ylim(-24.0, -14.0)
    axt.spines["bottom"].set_visible(False)
    axt.tick_params(bottom=False, labelbottom=False)
    d = 0.5  # diagonal break marks at the left spine
    mark = dict(marker=[(-1, -d), (1, d)], markersize=6, linestyle="none",
                color="k", mec="k", mew=0.6, clip_on=False)
    axt.plot([0], [0], transform=axt.transAxes, **mark)
    axb.plot([0], [1], transform=axb.transAxes, **mark)

    axb.set_xlabel("lead $h$ (months)")
    axt.set_ylabel("skill vs damped persistence (%)")
    axt.yaxis.set_label_coords(-0.085, 0.38)
    axb.set_yticks([-22, -18, -14])
    axb.set_xticks(H)
    axb.set_xlim(0.75, 6.25)
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

    fig, ax = plt.subplots(figsize=(12 * CM, 8.5 * CM))

    ax.axhline(0.0, color=REFGRAY, lw=0.8, zorder=1)
    ax.axvline(2.5, color=REFGRAY, lw=0.7, ls="--", zorder=1)
    ax.annotate("crossover", xy=(2.55, 2.0), ha="left", va="center",
                fontsize=7, color=REFGRAY)

    ax.fill_between(H, mlo, mhi, color=NEARBLACK, alpha=0.12, lw=0, zorder=2)

    def draw(y, p, color, ls, marker, lw, label, z):
        ax.plot(H, y, ls=ls, color=color, lw=lw, label=label, zorder=z)
        sig_markers(ax, H, y, p, marker, color, z=z + 1)

    draw(main, mp, NEARBLACK, "-", "o", 1.6,
         "stacked ensemble vs GRACE-FCast (full)", 5)
    draw(kal, kp, OI["blue"], "--", "s", 1.3,
         "Kalman alone vs GRACE-FCast (full)", 3)
    draw(nons, np_, OI["purple"], ":", "^", 1.3,
         "stacked ensemble vs GRACE-FCast (non-seas.)", 3)

    dm_note(ax)
    ax.set_xlabel("lead $h$ (months)")
    ax.set_ylabel("skill vs GRACE-FCast (%)")
    ax.set_xticks(H)
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(-42, 34)
    ax.legend(loc="upper right", handlelength=2.6)
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
    # NOTE (styling decision, 2026-08-17): the input-channel route has no
    # seed-1 contrast in phase8b_h16_headline.csv (only lstm_corr_top1_era5_s0),
    # so single-seed markers are dropped from BOTH routes to keep the seed
    # treatment symmetric; the ensembles carry the figure. s0/s1/ch0 are still
    # loaded because the ledger and placebo cross-checks below depend on them.

    # --- ledger section 5 ---
    assert_ledger("F5 ensemble correction", corr, [0.91, 1.45, 1.33, 1.32, 1.42, 1.96])
    assert_ledger("F5 ensemble CI lower", corr_lo, [0.64, 1.07, 1.02, 0.96, 1.06, 1.66])
    # Max dm_p = 3.14e-08 at h4 (h1 = 1.10e-10, h6 = 1.21e-13). The ledger's
    # earlier "All p <= 1.2e-10" was a transcription slip from the h1 value,
    # corrected in the ledger on 2026-08-15; both now agree with the CSV.
    # dm_p is NOT printed on the figure; we assert the level the CSV supports.
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

    # ledger checks: per-cell means -0.26..-0.07 %, and 20/20 beaten in all
    # 12 lead x seed cells. Bounds moved from -0.27..-0.05 on 2026-08-16: draws are
    # now redrawn per (fold, horizon) rather than reused, so the placebo distribution
    # is resampled rather than one shared set re-scored.
    cell_means = pl.groupby(["seed", "horizon"]).incr.mean()
    assert abs(cell_means.min() - (-0.26)) <= TOL and abs(cell_means.max() - (-0.07)) <= TOL, (
        f"placebo per-cell means {cell_means.min():+.3f}..{cell_means.max():+.3f} "
        "do not match ledger -0.26..-0.07"
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
    fig, ax = plt.subplots(figsize=(12 * CM, 8.2 * CM))

    ax.axhline(0.0, color=REFGRAY, lw=0.8, zorder=1)
    ax.fill_between(H, band_lo, band_hi, color=NULLGRAY, alpha=0.35, lw=0, zorder=2,
                    label="random-graph placebo range")

    # correction stage: ensemble curve + CI ribbon
    ax.fill_between(H, corr_lo, corr_hi, color=NEARBLACK, alpha=0.12, lw=0, zorder=3)
    ax.plot(H, corr, ls="-", color=NEARBLACK, lw=1.6, zorder=6,
            label="as correction stage (2-seed ensemble)")
    sig_markers(ax, H, corr, corr_p, "o", NEARBLACK, z=7)

    # input channel: ensemble curve + CI error bars
    ax.errorbar(H, chan, yerr=[chan - chan_lo, chan_hi - chan],
                ls="--", color=OI["orange"], lw=1.4, elinewidth=0.8,
                ecolor="0.6", zorder=4,
                label="as input channel (2-seed ensemble)")
    sig_markers(ax, H, chan, chan_p, "s", OI["orange"], z=5)

    ax.annotate("identical neighbor information, two deliveries",
                xy=(0.98, 2.42), ha="left", va="top", fontsize=7, color="0.15")
    dm_note(ax)

    ax.set_xlabel("lead $h$ (months)")
    ax.set_ylabel("skill increment over stage-1 LSTM (%)")
    ax.set_xticks(H)
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(-1.8, 2.6)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in [
        "as correction stage (2-seed ensemble)",
        "as input channel (2-seed ensemble)",
        "random-graph placebo range"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
              handlelength=2.0, columnspacing=1.2, borderaxespad=0.0)
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

    # one hue at three lightness steps; DARKEST = high tercile (carries the finding)
    STYLE = {  # shared tercile identity across both panels
        "cont_tercile_low": dict(color=tint(OI["blue"], 0.62), ls=":", marker="o",
                                 label="low"),
        "cont_tercile_mid": dict(color=tint(OI["blue"], 0.28), ls="--", marker="s",
                                 label="mid"),
        "cont_tercile_high": dict(color=shade(OI["blue"], 0.25), ls="-", marker="^",
                                  label="high"),
        "resolved_x_cont_lowmid": dict(color=NEARBLACK, ls="-.", marker="D",
                                       label="resolved × low/mid sharing"),
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
                    lw=lw, elinewidth=0.5, ecolor="0.6", zorder=4,
                    label=s["label"])
        sig = p < 0.05
        for m, fc in [(sig, s["color"]), (~sig, "white")]:
            ax.plot(x[m], y[m], ls="none", marker=s["marker"], color=s["color"],
                    mfc=fc, mec=s["color"], markersize=3.8, zorder=5)

    for ax in (axa, axb):
        ax.axhline(0.0, color=REFGRAY, lw=0.8, zorder=1)
        ax.set_xticks(H)
        ax.set_xlim(0.6, 6.4)
        ax.set_xlabel("lead $h$ (months)")

    for s in ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high"]:
        draw(axa, lin[s], s, 1.1)
    for s in ["cont_tercile_low", "cont_tercile_mid", "cont_tercile_high",
              "resolved_x_cont_lowmid"]:
        draw(axb, stk[s], s, 1.1)

    # panel letters inside the axes; legends above the data region
    axa.text(0.02, 0.975, "(a)", transform=axa.transAxes, va="top",
             fontweight="bold", fontsize=8)
    axa.text(0.085, 0.975, "linear tier (ridge, +ERA5)", transform=axa.transAxes,
             va="top", fontsize=8)
    axb.text(0.02, 0.975, "(b)", transform=axb.transAxes, va="top",
             fontweight="bold", fontsize=8)
    axb.text(0.085, 0.975, "stacked correction (LSTM ensemble)", transform=axb.transAxes,
             va="top", fontsize=8)
    axa.set_ylabel("skill vs own-only twin (%)")
    ha, la = axa.get_legend_handles_labels()
    la = ["footprint sharing low" if l == "low" else l for l in la]
    axa.legend(ha, la, loc="lower left",
               bbox_to_anchor=(0.0, 1.02), ncol=3, columnspacing=1.0,
               handlelength=1.8, borderaxespad=0.0)
    hb, lb = axb.get_legend_handles_labels()
    keep = [i for i, l in enumerate(lb) if l.startswith("resolved")]
    axb.legend([hb[i] for i in keep], [lb[i] for i in keep], loc="lower left",
               bbox_to_anchor=(0.0, 1.02), handlelength=2.0, borderaxespad=0.0)
    axa.set_ylim(-1.9, 2.75)  # shared

    # filled = DM p < 0.05, open = ns (stated in caption as well)
    dm_note(axa, 0.03, 0.03)
    save(fig, "fig08_stratification")


# ---------------------------------------------------------------------------
# F3 -- per-basin neighbor-effect map (choropleth on the mask grid)
# ---------------------------------------------------------------------------

def fig03_neighbor_map():
    print("F3 fig03_neighbor_map")
    import sys
    import cartopy.crs as ccrs
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

    sig = fdr[fdr["significant"]]
    sig_names = set(sig.index)

    # choropleth values plus a significance mask (used for muting only)
    grid = np.full(lat.size * lon.size, np.nan)
    sig_mask = np.zeros(lat.size * lon.size, dtype=bool)
    for b, name in enumerate(meta["name"]):
        if name in fdr.index and np.isfinite(fdr.loc[name, "dm_stat"]):
            # sign flipped: positive = neighbor helps
            grid[masks["indices"][b]] = -float(fdr.loc[name, "dm_stat"])
            if name in sig_names:
                sig_mask[masks["indices"][b]] = True
    grid = grid.reshape(lat.size, lon.size)
    sig_mask = sig_mask.reshape(lat.size, lon.size)

    # Roll 0..360 -> -180..180 for a conventional world map
    shift = lon.size // 2
    grid = np.roll(grid, shift, axis=1)
    sig_mask = np.roll(sig_mask, shift, axis=1)
    lon_c = np.where(lon >= 180, lon - 360, lon)
    lon_plot = np.sort(lon_c)

    pc = ccrs.PlateCarree()
    fig = plt.figure(figsize=(17 * CM, 9.6 * CM))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -60, 84], crs=pc)
    ax.set_facecolor("0.94")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor("0.3")

    # single mesh with precomputed RGBA: non-FDR-significant basins are
    # white-blended (muted) so no alpha compositing stipple appears
    norm = mcolors.Normalize(vmin=-3, vmax=3)
    rgba = cm.vik_r(norm(np.clip(grid, -3, 3)))  # NaN cells -> transparent
    ns = np.isfinite(grid) & ~sig_mask
    rgba[ns, :3] = 1.0 - 0.45 * (1.0 - rgba[ns, :3])
    ax.pcolormesh(lon_plot, lat, rgba, rasterized=True, shading="nearest",
                  transform=pc)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm.vik_r)
    ax.coastlines(lw=0.3, color="0.4")

    # FDR-significant basins: filled dot = helped, open = hurt (21 = 13 + 8)
    cent = pd.read_csv(ROOT / "data/processed/basin_meta.csv").set_index("name")
    n_h = int(sig["a_better"].sum())
    n_u = int((~sig["a_better"]).sum())
    assert (n_h, n_u) == (13, 8), f"FDR roster changed: {n_h} helped / {n_u} hurt"
    for name, row in sig.iterrows():
        la = cent.loc[name, "centroid_lat"]
        lo = cent.loc[name, "centroid_lon"]
        lo = lo - 360 if lo > 180 else lo
        if row["a_better"]:
            ax.plot(lo, la, "o", ms=3.4, mfc="k", mec="k", mew=0.5, zorder=5,
                    transform=pc)
        else:
            ax.plot(lo, la, "o", ms=3.8, mfc="none", mec="k", mew=0.9, zorder=5,
                    transform=pc)
    tr = pc._as_mpl_transform(ax)
    halo = [withStroke(linewidth=1.5, foreground="white")]
    for name, label, dy in (("R_Yenisey_River", "Yenisey", 3), ("R_Parana_River", "Paraná", -6),
                            ("R_Yangtze_River", "Yangtze", -6), ("R_Amur_River", "Amur", 4),
                            ("R_Niger_River", "Niger", -6), ("R_Sao_Francisco_River", "São Francisco", 4)):
        la = cent.loc[name, "centroid_lat"]
        lo = cent.loc[name, "centroid_lon"]
        lo = lo - 360 if lo > 180 else lo
        ax.annotate(label, xy=(lo, la), xycoords=tr, xytext=(lo + 2, la + dy),
                    textcoords=tr, fontsize=6.5, zorder=6, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.25"))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.7,
                      pad=0.04, aspect=42, extend="both")
    cb.set_label("per-basin DM statistic, lead 1 (positive = neighbor helps)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.annotate(f"filled: FDR-significant helped (n={n_h})   open: hurt (n={n_u})   "
                f"q=0.10   muted fill: not significant",
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

    # (a) two null-distribution strips sharing the pooled-RMSE axis.
    # Per-surrogate RMSEs are not stored (phase4_surrogate_summary.csv keeps
    # only mean/min/p_rank), so the IAAFT row shows the summary marks.
    axa.grid(False, axis="y")
    axa.grid(True, axis="x")
    rng = np.random.default_rng(7)
    yj = 1.0 + rng.uniform(-0.15, 0.15, size=len(pooled))
    axa.scatter(pooled.values, yj, s=6, color=NULLGRAY, edgecolor="none", zorder=3)
    axa.plot([real_rmse] * 2, [0.70, 1.30], color=OI["blue"], lw=1.4, zorder=4)
    axa.annotate("real graph\n(beats 50/50)", xy=(real_rmse, 1.32),
                 xytext=(1, 2), textcoords="offset points", ha="left",
                 va="bottom", fontsize=6.5, color=OI["blue"], zorder=5)
    axa.plot([su1["real_rmse"]] * 2, [-0.30, 0.30], color=OI["blue"], lw=1.4, zorder=4)
    axa.annotate("real\n(beats 99/99, $p_{rank}$=0.01)", xy=(su1["real_rmse"], -0.32),
                 xytext=(1, -2), textcoords="offset points", ha="left",
                 va="top", fontsize=6.5, color=OI["blue"], zorder=5)
    axa.scatter([su1["surr_mean"]], [0.0], s=6, color=NULLGRAY,
                edgecolor="none", zorder=3)
    axa.scatter([su1["surr_min"]], [0.0], s=10, facecolor="white",
                edgecolor="0.55", linewidth=0.6, zorder=3)
    axa.annotate("mean of 99", xy=(su1["surr_mean"], -0.10), ha="right",
                 va="top", fontsize=6.5, color="0.45")
    axa.annotate("best of 99", xy=(su1["surr_min"], 0.08), ha="right",
                 va="bottom", fontsize=6.5, color="0.45")
    axa.set_yticks([1.0, 0.0])
    axa.set_yticklabels(["random\ngraphs", "IAAFT\nsurrogates"], fontsize=7)
    axa.set_ylim(-0.85, 1.75)
    pad = 0.00025
    axa.set_xlim(min(real_rmse, su1["real_rmse"]) - pad,
                 max(pooled.max(), su1["surr_mean"]) + pad)
    axa.set_xlabel("pooled lead-1 RMSE (std. units)")
    axa.xaxis.set_major_locator(plt.MaxNLocator(4))
    axa.set_title("(a) placebo + surrogate nulls", loc="left")

    # (b) distance profile (corrected values; placebo counts from the summary)
    prof = [("no restr.", "kalman_corr_top1"), ("≥300 km", "kalman_corr_min300_top1"),
            ("≥500 km", "kalman_corr_min500_top1"), ("≥1000 km", "kalman_corr_min1000_top1")]
    xs = np.arange(len(prof))
    vals = [100 * float(s1.loc[m, "skill_vs_own_ridge"]) for _, m in prof]
    beat = [f"{int(s1.loc[m, 'placebo_beaten'])}/{int(s1.loc[m, 'placebo_n'])}" for _, m in prof]
    assert_ledger("F4 distance profile", vals, [0.31, 0.32, 0.16, -0.60])
    for x, v in zip(xs, vals):
        if v >= 0:
            axb.bar(x, v, width=0.62, color=OI["blue"], edgecolor="none", zorder=3)
        else:
            axb.bar(x, v, width=0.62, facecolor="0.85", edgecolor="0.5",
                    linewidth=0.5, hatch="///", zorder=3)
    axb.axhline(0, color=REFGRAY, lw=0.7, zorder=2)
    for x, v, b in zip(xs, vals, beat):
        axb.annotate(b, xy=(x, v), xytext=(0, 3 if v >= 0 else -9),
                     textcoords="offset points", ha="center", fontsize=6.5,
                     clip_on=False, zorder=5)
    axb.set_xticks(xs)
    axb.set_xticklabels([p[0] for p in prof], fontsize=6.5, rotation=18, ha="right")
    axb.set_ylim(-0.85, 0.48)
    axb.set_ylabel("lead-1 skill vs own ridge (%)")
    axb.set_title("(b) source-distance exclusion", loc="left")

    # (c) conditioning invariance, bootstrap CIs, as a horizontal caterpillar
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
    axc.grid(False, axis="y")
    axc.grid(True, axis="x")
    ys = np.array([2, 1, 0])
    axc.axvline(0, color=REFGRAY, lw=0.8, zorder=2)
    axc.axvline(np.mean([r[1] for r in rows]), color="0.8", lw=0.6, ls=":", zorder=1)
    for y, (lab, v, l, h) in zip(ys, rows):
        axc.errorbar(v, y, xerr=[[v - l], [h - v]], fmt="o", ms=4, color=OI["blue"],
                     ecolor="0.45", elinewidth=1.0, zorder=4)
    axc.set_yticks(ys)
    axc.set_yticklabels([r[0] for r in rows], fontsize=7)
    axc.set_ylim(-0.6, 2.6)
    axc.set_xlabel("lead-1 skill vs own reference (%)")
    axc.set_title("(c) conditioning invariance", loc="left")
    save(fig, "fig04_controls")


# ---------------------------------------------------------------------------
# F6 -- ERA5 gain vs neighbor gain complementarity
# ---------------------------------------------------------------------------

def fig06_complementarity():
    print("F6 fig06_complementarity")
    from scipy.stats import spearmanr, theilslopes

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

    XL, YL = (-30.0, 40.0), (-12.0, 12.0)
    x = df["era5"].to_numpy()
    y = df["nbr"].to_numpy()
    clipped = (x < XL[0]) | (x > XL[1]) | (y < YL[0]) | (y > YL[1])
    n_off = int(clipped.sum())

    fig, ax = plt.subplots(figsize=(12 * CM, 8.6 * CM))
    ax.grid(False)
    ax.axhline(0, color=REFGRAY, lw=0.6, zorder=1)
    ax.axvline(0, color=REFGRAY, lw=0.6, zorder=1)

    # all 234 basins, de-emphasized; clipped points pinned as open triangles
    ax.scatter(x[~clipped], y[~clipped], s=7, color=NULLGRAY,
               edgecolor="none", zorder=3)
    for xi, yi in zip(x[clipped], y[clipped]):
        if xi < XL[0]:
            m = "<"
        elif xi > XL[1]:
            m = ">"
        elif yi < YL[0]:
            m = "v"
        else:
            m = "^"
        ax.scatter(np.clip(xi, *XL), np.clip(yi, *YL), s=14, marker=m,
                   facecolor="none", edgecolor="0.55", linewidth=0.6, zorder=3)

    # highlighted continents with direct labels (white halo)
    halo = [withStroke(linewidth=1.5, foreground="white")]
    for cont, lab, col, dx, dy, ha in [
        ("africa", "Africa", OI["blue"], -2.0, 4.2, "right"),
        ("europe", "Europe", OI["vermillion"], 2.5, -3.6, "left"),
    ]:
        g = df[df["continent"] == cont]
        ax.scatter(g["era5"].clip(*XL), g["nbr"].clip(*YL), s=14, color=col,
                   edgecolor="none", zorder=5)
        mx, my = g["era5"].median(), g["nbr"].median()
        ax.text(mx + dx, my + dy, lab, color=col, fontsize=8, ha=ha,
                va="center", zorder=6, path_effects=halo)

    # Theil-Sen fit with the rank correlation attached to the line's end
    slope, intercept, _, _ = theilslopes(df["nbr"], df["era5"])
    xx = np.array(XL)
    ax.plot(xx, intercept + slope * xx, color="k", lw=1.0, zorder=4)
    ax.annotate(f"Spearman $\\rho$ = {rho:+.2f} ($p$ = {p:.4f})",
                xy=(XL[1] - 0.8, intercept + slope * (XL[1] - 0.8)),
                xytext=(0, -7), textcoords="offset points",
                ha="right", va="top", fontsize=7, color="k",
                path_effects=halo, zorder=6)

    # the two off-diagonal quadrants are each other's dead zones
    ax.text(XL[0] + 1.2, YL[1] - 0.6, "ERA5 dead zone:\nneighbors cover it",
            fontsize=6.5, color="0.45", ha="left", va="top", zorder=2)
    ax.text(XL[1] - 1.2, YL[0] + 0.6, "neighbor dead zone:\nERA5 covers it",
            fontsize=6.5, color="0.45", ha="right", va="bottom", zorder=2)

    ax.annotate(f"234 basins; {n_off} clipped to range (open triangles)",
                xy=(0.02, 0.045), xycoords="axes fraction", fontsize=6.5,
                color="0.45")
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    ax.set_xlabel("per-basin ERA5 gain, lead 1 (%)")
    ax.set_ylabel("per-basin ERA5-conditioned neighbor gain, lead 1 (%)")
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
