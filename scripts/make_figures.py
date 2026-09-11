"""Publication figures for the GRACE TWSA forecasting paper (HESS, copernicus.cls).

Six figures, each a single file, vector PDF plus a 150 dpi PNG preview, into
``figures/``:

    fig01_basins            the sample and the two comparison subsets
    fig02_benchmark_ladder  where the filter sits on the conventional ladder
    fig03_crossing          the crossover against the published product
    fig04_filter_mechanism  which half of the filter does the work
    fig05_era5_where        where the ERA5 window pays
    fig06_sequence_models   the sequence models against the flat-12 ridge

Every number plotted is read from ``results/*.csv`` or ``data/processed/*.csv``;
nothing is typed in by hand. Every headline value is asserted against its source
to the precision the manuscript prints, so a rerun that moves a number breaks the
build instead of silently redrawing it. The asserts are pinned to the CURRENT
result files, not to paper/notes/REWRITE_LEDGER.md, which describes the earlier
three-finding structure and is no longer authoritative.

Sign convention throughout: skill = 1 - MSE(first)/MSE(second), so a positive
number means the first-named model is better. Where a source file lists the
published product first, the figure plots the inverse comparison
1 - MSE(ours)/MSE(theirs), which is NOT the negated skill -- see ours_over on the
flip helper below.

Usage:  .venv\\Scripts\\python.exe scripts\\make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
JPL = RESULTS / "jpl"
PROC = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
MASK_NC = ROOT / "HydroShed+Mascon_Basins_L3.nc"

sys.path.insert(0, str(ROOT / "src"))

CM = 1.0 / 2.54  # cm -> inch
H = np.array([1, 2, 3, 4, 5, 6])

# HESS column widths
W1 = 8.3 * CM   # single column
W2 = 17.0 * CM  # double column

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.linewidth": 0.4, "grid.color": "0.90",
    "lines.linewidth": 1.4, "lines.markersize": 3.6,
    "legend.frameon": False, "legend.handlelength": 2.4,
    "errorbar.capsize": 0,
    "hatch.linewidth": 0.45,
    "pdf.fonttype": 42, "savefig.bbox": "standard", "savefig.dpi": 300,
})

# Categorical colours are cmcrameri batlow samples, capped at a luminance that
# still reads as a 1.4 pt line on white. batlow is perceptually uniform and
# colour-vision-deficiency safe, and no two series here form a red-green pair.
BATLOW = {
    "navy":   "#011959",  # L 0.10
    "blue":   "#114360",  # L 0.23
    "teal":   "#226061",  # L 0.33
    "green":  "#4d734d",  # L 0.41
    "olive":  "#828231",  # L 0.49
    "gold":   "#c09036",  # L 0.58
    "orange": "#f19d6b",  # L 0.67
    "pink":   "#d7999b",  # L 0.65 -- batlow's #fdb4b6 darkened 15 % so a 1.4 pt
                          # line still reads on white; hue unchanged
}
REFGRAY = "0.45"
LANDGRAY = "0.93"
COASTGRAY = "0.55"

# One name and one look per model, reused by every figure. Plain English only:
# no code identifiers ever reach a legend.
STYLE = {
    "kalman_ar1":            dict(label="Kalman reference",     c=BATLOW["navy"],   ls="-",  m="o", lw=1.7),
    "kalman_own_ridge":      dict(label="own-state ridge",      c=BATLOW["teal"],   ls="-",  m="^", lw=1.3),
    "ridge_own_flat12":      dict(label="flat-12 ridge",        c=BATLOW["green"],  ls="-",  m="s", lw=1.3),
    "ridge_own_era5_flat12": dict(label="flat-12 ERA5 ridge",   c=BATLOW["gold"],   ls="-",  m="D", lw=1.5),
    "ridge_own_perbasin":    dict(label="per-basin ridge",      c=BATLOW["blue"],   ls="--", m="v", lw=1.2),
    "ridge_own_lags":        dict(label="lag-feature ridge",    c=BATLOW["olive"],  ls="--", m="<", lw=1.2),
    "ridge_own_era5":        dict(label="lag-feature ERA5 ridge", c=BATLOW["olive"], ls="--", m="<", lw=1.2),
    "persistence":           dict(label="persistence",          c=BATLOW["orange"], ls="-.", m=">", lw=1.2),
    "climatology_zero":      dict(label="climatology",          c=BATLOW["pink"],   ls=":",  m="x", lw=1.4),
    "damped_persistence":    dict(label="damped persistence",   c=REFGRAY,          ls="-",  m=None, lw=1.2),
    "mlp_own_era5_flat12":   dict(label="residual MLP (flat-12 window)", c=BATLOW["orange"], ls="--", m="P", lw=1.2),
    "lstm_own_era5_ens":     dict(label="LSTM ensemble (own state + ERA5)", c=BATLOW["blue"], ls="-", m="h", lw=1.3),
    "li_lstm_full":          dict(label="published product (full)", c=BATLOW["navy"], ls="-", m="o", lw=1.5),
    "li_lstm_nonseas":       dict(label="published product (non-seasonal)", c=BATLOW["gold"], ls="--", m="D", lw=1.5),
}

CAPTION_NOTES = []


def note(fig_stem, text):
    """Record a caveat the caption must carry; printed at the end and in BUILD_NOTES."""
    CAPTION_NOTES.append((fig_stem, text))
    print(f"    NOTE [{fig_stem}] {text}")


# ---------------------------------------------------------------------------
# Assertions against the current source files
# ---------------------------------------------------------------------------

def assert_source(label, plotted, expected, tol=0.005):
    """Hard check: what is drawn must reproduce the source value to print precision.

    STOP (raise) on mismatch. Never adjust the expected value to make it pass --
    a mismatch means the results moved and the figure must not be used.
    """
    p = np.atleast_1d(np.asarray(plotted, dtype=float))
    e = np.atleast_1d(np.asarray(expected, dtype=float))
    if p.shape != e.shape:
        raise AssertionError(f"[{label}] shape {p.shape} != {e.shape}")
    bad = np.abs(p - e) > tol
    if bad.any():
        rows = "; ".join(f"#{i}: plotted {p[i]:+.5f} vs source {e[i]:+.5f}"
                         for i in np.flatnonzero(bad))
        raise AssertionError(f"[{label}] SOURCE MISMATCH -- {rows}")
    print(f"    assert ok  {label}")


def save(fig, stem):
    """Vector PDF plus a 150 dpi PNG preview, at the exact figure size.

    bbox_inches is deliberately NOT "tight": a tight box crops to whatever the
    artists happen to occupy, which silently pushes the page past the 17 cm
    double-column limit and then gets scaled down in LaTeX, shrinking every
    label below the 7 pt floor. Margins are set per figure instead, and the
    width is checked here.
    """
    FIGURES.mkdir(exist_ok=True)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = fig.get_size_inches()
    over = [("left", -bb.x0), ("right", bb.x1 - fw),
            ("bottom", -bb.y0), ("top", bb.y1 - fh)]
    spill = [(s, v) for s, v in over if v > 0.005]
    if spill:
        raise AssertionError(
            f"[{stem}] content runs off the canvas: "
            + ", ".join(f"{s} by {v / CM:.2f} cm" for s, v in spill)
        )
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches=None,
                metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{stem}.png", dpi=150, bbox_inches=None)
    w_cm, h_cm = [v / CM for v in fig.get_size_inches()]
    if not (8.0 <= w_cm <= 17.05):
        raise AssertionError(
            f"[{stem}] width {w_cm:.2f} cm is outside the HESS range 8-17 cm"
        )
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf + .png  ({w_cm:.1f} x {h_cm:.1f} cm)")


def panel_label(ax, text, x=0.0, y=1.015, **kw):
    ax.text(x, y, text, transform=ax.transAxes, va="bottom", ha="left",
            fontweight="bold", fontsize=8.5, **kw)


def panel_head(ax, letter, text, y=1.015):
    """Bold panel letter plus its one-line description, offset in points so the
    description never collides with the letter whatever the panel width is."""
    panel_label(ax, letter, x=0.0, y=y)
    ax.annotate(text, xy=(0.0, y), xycoords="axes fraction", xytext=(20, 0),
                textcoords="offset points", va="bottom", ha="left", fontsize=8)


def lead_axis(ax, lo=0.7, hi=6.3, leads=H):
    ax.set_xticks(leads)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("forecast lead $h$ (months)")


def series(df, keys, value, horizon_col="horizon", leads=H):
    sub = df.copy()
    for k, v in keys.items():
        sub = sub[sub[k] == v]
    sub = sub.sort_values(horizon_col)
    if list(np.asarray(sub[horizon_col], dtype=int)) != list(leads):
        raise AssertionError(f"rows for {keys} do not cover leads {list(leads)}")
    return sub[value].to_numpy(dtype=float)


def ours_over(skill_theirs_first):
    """Invert a skill reported with the other model first.

    skill_theirs_first = 1 - MSE(theirs)/MSE(ours); the figure wants
    1 - MSE(ours)/MSE(theirs) = 1 - 1/(1 - skill_theirs_first). This is the
    reciprocal transform, not a sign flip: a sign flip would report +77.7 %
    where the RMSEs say +43.7 %. Verified against both directions of the CSR
    file and against rmse_std in the JPL summary (see fig03).
    """
    s = np.asarray(skill_theirs_first, dtype=float)
    return 1.0 - 1.0 / (1.0 - s)


def zero_crossing(y, leads=H):
    """Lead at which a skill curve crosses zero, by linear interpolation."""
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            return leads[i] + y[i] / (y[i] - y[i + 1])
    return None


# ---------------------------------------------------------------------------
# Shared map machinery
# ---------------------------------------------------------------------------

_MASK_CACHE = {}


def basin_grid():
    """Flat cell indices per basin plus the plotting lat/lon axes (-180..180)."""
    if "g" not in _MASK_CACHE:
        import xarray as xr
        from gracefc.basins import load_basin_masks

        masks = load_basin_masks(MASK_NC)
        dm = xr.open_dataset(MASK_NC, decode_times=False)
        lat, lon = dm["lat"].values, dm["lon"].values
        dm.close()
        lon_plot = np.sort(np.where(lon >= 180, lon - 360, lon))
        _MASK_CACHE["g"] = dict(masks=masks, lat=lat, lon=lon,
                                lon_plot=lon_plot, shift=lon.size // 2)
    return _MASK_CACHE["g"]


def paint(values_by_name, fill=np.nan):
    """Rasterize a per-basin scalar onto the mask grid, rolled to -180..180."""
    g = basin_grid()
    lat, lon = g["lat"], g["lon"]
    flat = np.full(lat.size * lon.size, fill, dtype=float)
    for b, name in enumerate(g["masks"]["meta"]["name"]):
        v = values_by_name.get(name)
        if v is not None and np.isfinite(v):
            flat[g["masks"]["indices"][b]] = v
    return np.roll(flat.reshape(lat.size, lon.size), g["shift"], axis=1)


def paint_flag(names):
    """Boolean grid: 1 inside any basin in `names`, 0 elsewhere."""
    g = basin_grid()
    lat, lon = g["lat"], g["lon"]
    flat = np.zeros(lat.size * lon.size, dtype=float)
    want = set(names)
    for b, name in enumerate(g["masks"]["meta"]["name"]):
        if name in want:
            flat[g["masks"]["indices"][b]] = 1.0
    return np.roll(flat.reshape(lat.size, lon.size), g["shift"], axis=1)


def world_axes(fig, spec):
    import cartopy.crs as ccrs

    ax = fig.add_subplot(spec, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, -58, 84], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(_land_feature(), facecolor=LANDGRAY, edgecolor="none", zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor("0.4")
    return ax


def _land_feature():
    import cartopy.feature as cfeature
    return cfeature.NaturalEarthFeature("physical", "land", "110m")


def draw_choropleth(ax, grid, norm, cmap):
    import cartopy.crs as ccrs
    g = basin_grid()
    ax.pcolormesh(g["lon_plot"], g["lat"], np.ma.masked_invalid(grid),
                  cmap=cmap, norm=norm, shading="nearest", rasterized=True,
                  transform=ccrs.PlateCarree(), zorder=2)
    ax.coastlines(lw=0.25, color=COASTGRAY, zorder=3)


def hatch_basins(ax, names, hatch="///", color="0.12"):
    """Hatch the BH-FDR significant basins (Wilks 2016: hatch the significant ones)."""
    import cartopy.crs as ccrs
    g = basin_grid()
    flag = paint_flag(names)
    cs = ax.contourf(g["lon_plot"], g["lat"], flag, levels=[0.5, 1.5],
                     colors="none", hatches=[hatch], transform=ccrs.PlateCarree(),
                     zorder=4)
    cs.set_edgecolor(color)
    cs.set_linewidth(0.0)
    return cs


def hcolorbar(fig, ax, norm, cmap, label, extend="both", shrink=0.72, pad=0.05):
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=shrink,
                      pad=pad, aspect=34, extend=extend)
    cb.set_label(label, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    return cb


# ===========================================================================
# Figure 1 -- the sample and the two comparison subsets
# ===========================================================================

def fig01_basins():
    stem = "fig01_basins"
    print(f"{stem}")
    import matplotlib.patches as mpatches

    meta = pd.read_csv(PROC / "basin_meta.csv")
    kept = meta[meta["exclude_reason"] == "keep"]
    assert len(kept) == 234, f"kept sample is {len(kept)}, not 234"

    cov = pd.read_csv(PROC / "li2026_basin_coverage.csv")
    assert set(cov["name"]) == set(kept["name"]), "coverage file is not the kept sample"

    strict = set(cov.loc[(cov["n_full_native_mascons"] >= 1)
                         & (cov["n_full_li_cells"] >= 1), "name"])
    matched = set(cov.loc[cov["li_coverage"] > 0, "name"])
    loose = matched - strict
    unmatched = set(cov["name"]) - matched

    # the strict roster must be exactly the basins the CSR comparison scored
    csr_pb = pd.read_csv(RESULTS / "phase6_li_comparison_perbasin.csv")
    assert set(csr_pb["name"]) == strict, "strict roster != scored CSR basins"
    csr_sum = pd.read_csv(RESULTS / "phase6_li_comparison_summary.csv")
    n_strict_csv = int(csr_sum.loc[csr_sum["subset"] == "joint_full_cells",
                                   "n_basins"].unique()[0])
    n_matched_csv = int(csr_sum.loc[csr_sum["subset"] == "all_matched",
                                    "n_basins"].unique()[0])

    jpl_pb = pd.read_csv(JPL / "phase6_li_comparison_perbasin.csv")
    jpl_strict = set(jpl_pb["name"])
    jpl_sum = pd.read_csv(JPL / "phase6_li_comparison_summary.csv")
    n_jpl_csv = int(jpl_sum.loc[jpl_sum["subset"] == "joint_full_cells",
                                "n_basins"].unique()[0])

    assert_source("F1 counts 209/18/7/67",
                  [len(strict), len(loose), len(unmatched), len(jpl_strict)],
                  [209, 18, 7, 67], tol=0.0)
    assert_source("F1 counts agree with the comparison summaries",
                  [n_strict_csv, n_matched_csv, n_jpl_csv],
                  [209, 227, 67], tol=0.0)
    assert jpl_strict <= set(kept["name"]), "JPL roster leaves the 234-basin sample"

    import cartopy.crs as ccrs

    C_STRICT, C_LOOSE, C_NONE = BATLOW["navy"], BATLOW["green"], BATLOW["orange"]
    C_OFF = "#9c9c9c"

    cat = {n: 0 for n in strict}
    cat.update({n: 1 for n in loose})
    cat.update({n: 2 for n in unmatched})
    grid_a = paint(cat)
    grid_b = paint({n: (0 if n in jpl_strict else 1) for n in kept["name"]})

    cmap_a = mcolors.ListedColormap([C_STRICT, C_LOOSE, C_NONE])
    cmap_b = mcolors.ListedColormap([C_STRICT, C_OFF])
    norm_a = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_a.N)
    norm_b = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap_b.N)

    fig = plt.figure(figsize=(W2, 15.6 * CM))
    gs = fig.add_gridspec(2, 1, left=0.012, right=0.988, top=0.950, bottom=0.075,
                          hspace=0.34)
    axa = world_axes(fig, gs[0])
    axb = world_axes(fig, gs[1])

    draw_choropleth(axa, grid_a, norm_a, cmap_a)
    draw_choropleth(axb, grid_b, norm_b, cmap_b)

    # The 18 and the 7 are small coastal and island units: at world scale their
    # fills are a few pixels, so each also carries a centroid marker.
    cent = meta.set_index("name")[["centroid_lat", "centroid_lon"]]
    for names, colour, mk in ((loose, C_LOOSE, "o"), (unmatched, C_NONE, "^")):
        la = cent.loc[sorted(names), "centroid_lat"].to_numpy()
        lo = cent.loc[sorted(names), "centroid_lon"].to_numpy()
        lo = np.where(lo > 180, lo - 360, lo)
        axa.plot(lo, la, ls="none", marker=mk, ms=4.6, mfc=colour, mec="white",
                 mew=0.7, zorder=6, transform=ccrs.PlateCarree())

    panel_head(axa, "(a)", "CSR mascons: the 234-basin sample and the strict "
               "comparison subset", y=1.02)
    panel_head(axb, "(b)", "JPL mascons: the strict comparison subset", y=1.02)

    leg_a = [mpatches.Patch(fc=C_STRICT, ec="none",
                            label=f"in the strict subset ($n$ = {len(strict)})"),
             plt.Line2D([], [], ls="none", marker="o", ms=4.6, mfc=C_LOOSE,
                        mec="white", mew=0.7,
                        label=f"matched, excluded by the strict rule "
                              f"($n$ = {len(loose)})"),
             plt.Line2D([], [], ls="none", marker="^", ms=4.6, mfc=C_NONE,
                        mec="white", mew=0.7,
                        label=f"no usable published forecast ($n$ = {len(unmatched)})")]
    axa.legend(handles=leg_a, loc="upper center", bbox_to_anchor=(0.5, -0.015),
               ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.3,
               handletextpad=0.5, borderaxespad=0.0, fontsize=7.2)
    axa.annotate("the last two categories are small coastal and island basins; "
                 "each also carries a marker at its centroid",
                 xy=(0.5, -0.105), xycoords="axes fraction", ha="center",
                 va="top", fontsize=7, color="0.35")

    leg_b = [mpatches.Patch(fc=C_STRICT, ec="none",
                            label=f"in the strict JPL subset ($n$ = {len(jpl_strict)})"),
             mpatches.Patch(fc=C_OFF, ec="none",
                            label=f"kept on CSR, outside the strict JPL subset "
                                  f"($n$ = {234 - len(jpl_strict)})"),
             mpatches.Patch(fc=LANDGRAY, ec="none",
                            label="land outside the basin sample")]
    axb.legend(handles=leg_b, loc="upper center", bbox_to_anchor=(0.5, -0.015),
               ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.3,
               handletextpad=0.5, borderaxespad=0.0, fontsize=7.2)
    axb.annotate("the JPL run's own basin roster is not held locally, so the "
                 "neutral shading is the CSR sample, not the 228 basins JPL kept",
                 xy=(0.5, -0.105), xycoords="axes fraction", ha="center",
                 va="top", fontsize=7, color="0.35")

    note(stem, "Panel (b) is not the JPL equivalent of panel (a). The JPL basin "
               "metadata is on the collaborator's machine, so only the 67 strict "
               "basins can be identified here; the remaining 167 basins are the "
               "234 the study keeps on CSR, shaded neutral, and are NOT claimed "
               "to be the 228 basins the JPL run kept.")
    note(stem, "The pale grey underlay is the Natural Earth 110 m land polygon, "
               "drawn only so land outside the basin sample reads as land rather "
               "than as a category of its own.")
    save(fig, stem)
    return dict(strict=len(strict), loose=len(loose), unmatched=len(unmatched),
                jpl=len(jpl_strict))


# ===========================================================================
# Figure 2 -- the conventional benchmark ladder
# ===========================================================================

def fig02_benchmark_ladder():
    stem = "fig02_benchmark_ladder"
    print(f"{stem}")
    lad = pd.read_csv(RESULTS / "paper_baseline_ladder.csv")
    con = pd.read_csv(RESULTS / "paper_baseline_contrasts.csv")
    conv = pd.read_csv(RESULTS / "conventional_metrics_summary.csv")

    # The zero line is the STRONGER damped variant at each lead: rho at h1,
    # regression at h2-h6. Pin that convention -- if it changes, the ladder is
    # no longer measured against what the caption says.
    ref = lad[lad["model"] == "kalman_ar1"].sort_values("horizon")["damped_ref"].tolist()
    assert ref == ["damped_persistence_rho"] + ["damped_persistence_reg"] * 5, (
        "damped_ref convention changed; the zero line is no longer the stronger variant"
    )

    def ladder(model):
        return series(lad, {"model": model}, "skill_vs_damped") * 100.0

    top_models = ["ridge_own_era5_flat12", "kalman_ar1", "ridge_own_flat12",
                  "kalman_own_ridge", "ridge_own_perbasin", "ridge_own_lags"]
    bot_models = ["persistence", "climatology_zero"]
    vals = {m: ladder(m) for m in top_models + bot_models}

    # headline asserts, straight off paper_baseline_ladder.csv
    assert_source("F2a Kalman reference lead 1 = +4.98 %", vals["kalman_ar1"][:1], [4.98])
    assert_source("F2a Kalman reference leads 1-6", vals["kalman_ar1"],
                  [4.98, 8.79, 5.62, 3.07, 2.55, 3.63])
    assert_source("F2a flat-12 ERA5 ridge leads 1-6", vals["ridge_own_era5_flat12"],
                  [12.24, 13.71, 9.61, 6.16, 4.58, 4.47])

    # CI ribbons where the contrasts file has them, always against the same
    # stronger damped variant the ladder used at that lead.
    def damped_ci(model):
        sub = con[(con["challenger"] == model)
                  & (((con["reference"] == "damped_persistence_rho") & (con["horizon"] == 1))
                     | ((con["reference"] == "damped_persistence_reg") & (con["horizon"] > 1)))
                  ].sort_values("horizon")
        if list(sub["horizon"]) != list(H):
            return None
        assert_source(f"F2a {STYLE[model]['label']} contrast == ladder curve",
                      sub["skill"].to_numpy() * 100.0, vals[model])
        return sub["ci_lo"].to_numpy() * 100.0, sub["ci_hi"].to_numpy() * 100.0

    cis = {m: damped_ci(m) for m in top_models}
    have_ci = [m for m in top_models if cis[m] is not None]
    print(f"    CI available for: {', '.join(STYLE[m]['label'] for m in have_ci)}")

    # panel (b): pooled raw-cm RMSE, stacked_ens excluded (an extended-chain arm)
    cm_models = ["damped_persistence", "kalman_ar1", "kalman_own_ridge",
                 "ridge_own_flat12", "ridge_own_era5_flat12"]
    pooled = {m: series(conv, {"model": m}, "pooled_rmse_cm") for m in cm_models}
    assert "stacked_ens" not in cm_models
    assert_source("F2b pooled lead-1 RMSE (cm): filter / flat-12 / flat-12 ERA5",
                  [pooled["kalman_ar1"][0], pooled["ridge_own_flat12"][0],
                   pooled["ridge_own_era5_flat12"][0]],
                  [5.128, 5.118, 5.231], tol=0.0005)

    fig = plt.figure(figsize=(W2, 10.0 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.10, 1.0], height_ratios=[2.55, 1.0],
                          left=0.070, right=0.988, top=0.940, bottom=0.215,
                          wspace=0.22, hspace=0.09)
    axt = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[1, 0], sharex=axt)
    axb = fig.add_subplot(gs[:, 1])

    axt.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    for m in have_ci:
        lo, hi = cis[m]
        axt.fill_between(H, lo, hi, color=STYLE[m]["c"], alpha=0.11, lw=0, zorder=3)
    for m in top_models:
        s = STYLE[m]
        axt.plot(H, vals[m], ls=s["ls"], marker=s["m"], color=s["c"], lw=s["lw"],
                 label=s["label"], zorder=5)
    for m in bot_models:
        s = STYLE[m]
        axm.plot(H, vals[m], ls=s["ls"], marker=s["m"], color=s["c"], lw=s["lw"],
                 label=s["label"], zorder=5)

    axt.annotate("zero line: damped persistence,\nstronger variant at each lead",
                 xy=(6.28, -0.85), ha="right", va="top", fontsize=7,
                 color=REFGRAY, zorder=6)

    axt.set_ylim(-2.5, 16.2)
    axt.set_yticks([0, 5, 10, 15])
    axm.set_ylim(-134, -10)
    axm.set_yticks([-120, -80, -40])
    axt.spines["bottom"].set_visible(False)
    axt.tick_params(bottom=False, labelbottom=False)
    d = 0.55
    brk = dict(marker=[(-1, -d), (1, d)], markersize=5.5, linestyle="none",
               color="0.25", mec="0.25", mew=0.7, clip_on=False)
    axt.plot([0], [0], transform=axt.transAxes, **brk)
    axm.plot([0], [1], transform=axm.transAxes, **brk)

    lead_axis(axm)
    axt.set_ylabel("skill vs damped persistence (%)")
    axt.yaxis.set_label_coords(-0.085, 0.30)
    panel_head(axt, "(a)", "skill against the conventional reference")

    for m in cm_models:
        s = STYLE[m]
        axb.plot(H, pooled[m], ls=s["ls"], marker=s["m"], color=s["c"], lw=s["lw"],
                 label=s["label"], zorder=5)
    lead_axis(axb)
    axb.set_ylabel("pooled RMSE (cm of equivalent water height)")
    axb.set_ylim(4.9, 7.15)
    panel_head(axb, "(b)", "the same models weighted by raw storage")

    order = ["kalman_ar1", "ridge_own_era5_flat12", "ridge_own_flat12",
             "kalman_own_ridge", "ridge_own_perbasin", "ridge_own_lags",
             "persistence", "climatology_zero", "damped_persistence"]
    handles = [plt.Line2D([], [], color=STYLE[m]["c"], ls=STYLE[m]["ls"],
                          marker=STYLE[m]["m"], lw=STYLE[m]["lw"],
                          markersize=3.6) for m in order]
    fig.legend(handles, [STYLE[m]["label"] for m in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=5, handlelength=2.2,
               columnspacing=1.3, fontsize=7.5)

    note(stem, "Panel (a) shading is the 95 % block-bootstrap CI from "
               "paper_baseline_contrasts.csv; persistence, climatology and the "
               "lag-feature ridge have no contrast rows in that file and are "
               "drawn without a band.")
    note(stem, "Panel (a) has a broken vertical axis: climatology and "
               "persistence sit far below the rest of the field.")
    note(stem, "Panel (b) excludes the stacked ensemble, which is an "
               "extended-chain arm and not part of the ladder.")
    save(fig, stem)


# ===========================================================================
# Figure 3 -- the crossover against the published product
# ===========================================================================

def _li_series(df, model, vs, subset="joint_full_cells"):
    sub = df[(df["subset"] == subset) & (df["model"] == model) & (df["vs"] == vs)]
    sub = sub.sort_values("horizon")
    if list(sub["horizon"]) != list(H):
        return None
    return (sub["skill"].to_numpy() * 100.0, sub["ci_lo"].to_numpy() * 100.0,
            sub["ci_hi"].to_numpy() * 100.0, sub["dm_p"].to_numpy(), sub)


def fig03_crossing():
    stem = "fig03_crossing"
    print(f"{stem}")
    csr = pd.read_csv(RESULTS / "phase6_li_comparison_headline.csv")
    jpl = pd.read_csv(JPL / "phase6_li_comparison_headline.csv")
    jpl_sum = pd.read_csv(JPL / "phase6_li_comparison_summary.csv")

    # --- CSR: our model is listed first, so the file value is already
    # "skill of ours over the published product". Verify against the
    # reverse-direction rows that the reciprocal identity holds.
    csr_pairs = [("kalman_ar1", "li_lstm_full"), ("kalman_ar1", "li_lstm_nonseas"),
                 ("ridge_own_era5_flat12", "li_lstm_full"),
                 ("ridge_own_era5_flat12", "li_lstm_nonseas")]
    A = {}
    for model, vs in csr_pairs:
        got = _li_series(csr, model, vs)
        assert got is not None, f"CSR rows missing for {model} vs {vs}"
        y, lo, hi, p, sub = got
        rev = _li_series(csr, vs, model)
        assert rev is not None, f"CSR reverse rows missing for {vs} vs {model}"
        assert_source(f"F3a reciprocal identity, {model} vs {vs}",
                      y, ours_over(rev[0] / 100.0) * 100.0, tol=0.002)
        A[(model, vs)] = (y, lo, hi, p)
        n_rows, n_months = int(sub["n_rows"].iloc[0]), int(sub["n_months"].iloc[0])
    assert (n_rows, n_months) == (12540, 60), (
        f"CSR strict sample is {n_rows} rows / {n_months} months, not 12540 / 60"
    )
    assert n_rows // n_months == 209, "CSR strict sample is not 209 basins"

    assert_source("F3a Kalman reference over the published product (full), lead 1 = +16.4 %",
                  A[("kalman_ar1", "li_lstm_full")][0][:1], [16.354], tol=0.01)
    assert_source("F3a flat-12 ERA5 ridge over the published product "
                  "(non-seasonal), leads 1-2",
                  A[("ridge_own_era5_flat12", "li_lstm_nonseas")][0][:2],
                  [32.781, 11.231], tol=0.01)

    # --- JPL: the collaborator's file lists the published product FIRST, so the
    # figure inverts it. Cross-check the inversion against rmse_std, which is
    # independent of how the skill column was signed.
    B = {}
    piv = jpl_sum[jpl_sum["subset"] == "joint_full_cells"].pivot(
        index="model", columns="horizon", values="rmse_std")
    for vs in ["li_lstm_full", "li_lstm_nonseas"]:
        got = _li_series(jpl, vs, "kalman_ar1")
        assert got is not None, f"JPL rows missing for {vs} vs kalman_ar1"
        y_theirs, lo_t, hi_t, p = got[:4]
        y = ours_over(y_theirs / 100.0) * 100.0
        # the transform is monotone decreasing, so the CI bounds swap
        lo = ours_over(hi_t / 100.0) * 100.0
        hi = ours_over(lo_t / 100.0) * 100.0
        from_rmse = (1.0 - (piv.loc["kalman_ar1"] / piv.loc[vs]) ** 2).to_numpy() * 100.0
        assert_source(f"F3b inversion == recompute from rmse_std, {vs}", y, from_rmse,
                      tol=0.002)
        B[vs] = (y, lo, hi, p)
    n_jpl = int(jpl[(jpl["subset"] == "joint_full_cells")]["n_rows"].iloc[0])
    n_jpl_m = int(jpl[(jpl["subset"] == "joint_full_cells")]["n_months"].iloc[0])
    assert (n_jpl, n_jpl_m) == (3953, 59), f"JPL strict sample is {n_jpl} / {n_jpl_m}"

    assert_source("F3b Kalman reference over the published product (full), "
                  "JPL lead 1 = +43.7 %", B["li_lstm_full"][0][:1], [43.718], tol=0.01)
    assert_source("F3b JPL source row (product first), lead 1 = -77.68 %",
                  [jpl[(jpl["subset"] == "joint_full_cells")
                       & (jpl["model"] == "li_lstm_full")
                       & (jpl["vs"] == "kalman_ar1")
                       & (jpl["horizon"] == 1)]["skill"].iloc[0] * 100.0],
                  [-77.678], tol=0.01)

    # --- draw ---------------------------------------------------------------
    def key(model, vs):
        c = STYLE[model]["c"]
        ls = "-" if vs == "li_lstm_full" else "--"
        mk = STYLE[model]["m"]
        lab = f"{STYLE[model]['label']} vs {STYLE[vs]['label']}"
        return c, ls, mk, lab

    fig = plt.figure(figsize=(W2, 9.4 * CM))
    gs = fig.add_gridspec(1, 2, left=0.070, right=0.988, top=0.930, bottom=0.255,
                          wspace=0.07)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharey=axa)

    OFF = {0: -0.13, 1: -0.045, 2: 0.045, 3: 0.13}

    from matplotlib.patheffects import withStroke
    halo = [withStroke(linewidth=1.8, foreground="white")]

    def draw(ax, x_off, y, lo, hi, p, c, ls, mk, lab):
        x = H + x_off
        ax.errorbar(x, y, yerr=[y - lo, hi - y], ls=ls, color=c, lw=1.4,
                    elinewidth=0.7, ecolor=c, alpha=1.0, zorder=5, label=lab)
        sig = np.asarray(p) < 0.05
        ax.plot(x[sig], y[sig], ls="none", marker=mk, mfc=c, mec=c, ms=4.0, zorder=6)
        ax.plot(x[~sig], y[~sig], ls="none", marker=mk, mfc="white", mec=c,
                mew=0.9, ms=4.0, zorder=6)
        xc = zero_crossing(y)
        if xc is not None:
            # The four CSR curves cross within a third of a lead of each other,
            # so only the marker goes on the axis; the values are listed in a
            # colour-keyed block in the corner.
            ax.plot([xc], [0], marker="v", ms=4.5, mfc=c, mec="white", mew=0.5,
                    zorder=8, clip_on=False)
        return xc

    def crossing_block(ax, rows, y0=0.245):
        ax.annotate("zero crossing (lead $h$)", xy=(0.025, y0), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=7, color="0.30",
                    path_effects=halo, zorder=9)
        for k, (colour, txt) in enumerate(rows):
            ax.annotate(txt, xy=(0.025, y0 - 0.062 * (k + 1)),
                        xycoords="axes fraction", ha="left", va="bottom",
                        fontsize=7, color=colour, path_effects=halo, zorder=9)

    crossings = {}
    for i, (model, vs) in enumerate(csr_pairs):
        c, ls, mk, lab = key(model, vs)
        y, lo, hi, p = A[(model, vs)]
        crossings[("CSR", model, vs)] = draw(axa, OFF[i], y, lo, hi, p, c, ls, mk, lab)
    for i, vs in enumerate(["li_lstm_full", "li_lstm_nonseas"]):
        c, ls, mk, lab = key("kalman_ar1", vs)
        y, lo, hi, p = B[vs]
        crossings[("JPL", "kalman_ar1", vs)] = draw(axb, OFF[i * 2], y, lo, hi, p,
                                                    c, ls, mk, lab)

    def row(tag, model):
        f = crossings[(tag, model, "li_lstm_full")]
        n = crossings[(tag, model, "li_lstm_nonseas")]
        return (STYLE[model]["c"],
                f"{STYLE[model]['label']}:  {f:.2f} vs full,  {n:.2f} vs non-seasonal")

    crossing_block(axa, [row("CSR", "kalman_ar1"),
                         row("CSR", "ridge_own_era5_flat12")])
    crossing_block(axb, [row("JPL", "kalman_ar1")])

    for ax in (axa, axb):
        ax.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
        lead_axis(ax)
    axa.set_ylim(-58, 64)
    axa.set_yticks([-50, -25, 0, 25, 50])
    axa.set_ylabel("skill of our model over the published product (%)")
    axb.tick_params(labelleft=False)

    panel_head(axa, "(a)", f"CSR mascons, 209 basins, {n_months} months")
    panel_head(axb, "(b)", f"JPL mascons, 67 basins, {n_jpl_m} months")
    for ax in (axa, axb):
        ax.annotate("we are ahead", xy=(0.985, 0.985), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7, color=REFGRAY)
        ax.annotate("the published product is ahead", xy=(0.985, 0.015),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=7, color=REFGRAY)

    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=2, handlelength=2.4, columnspacing=2.0, fontsize=7.5)
    fig.text(0.5, 0.115, "filled markers: Diebold-Mariano $p<0.05$;  open markers: "
                         "not significant;  triangles on the zero line mark the "
                         "interpolated lead at which each curve crosses",
             ha="center", va="bottom", fontsize=7, color="0.30")

    note(stem, "The JPL file reports the published product first. The figure "
               "inverts it as 1 - MSE(ours)/MSE(theirs), which is the reciprocal "
               "of the stored skill, not its negation; the script asserts the "
               "inverted values against rmse_std in the JPL summary.")
    note(stem, "Both panels use the joint_full_cells subset. The JPL run "
               "predates the flat-12 step, so panel (b) carries the Kalman "
               "reference only.")
    save(fig, stem)
    return crossings


# ===========================================================================
# Figure 4 -- which half of the filter does the work
# ===========================================================================

def fig04_filter_mechanism():
    stem = "fig04_filter_mechanism"
    print(f"{stem}")
    r0 = pd.read_csv(RESULTS / "r0_ablation_summary.csv")
    ms = pd.read_csv(RESULTS / "kalman_mission_summary.csv")
    cmet = pd.read_csv(RESULTS / "conventional_metrics_perbasin.csv")

    # (a) the same filter with and without the observation-noise term, both
    # scored against the regression damped variant at every lead.
    full = r0[r0["component"] == "full_margin"].sort_values("horizon")
    zero = r0[r0["component"] == "rho_estimation_term"].sort_values("horizon")
    gap = r0[r0["component"] == "noise_filtering_term"].sort_values("horizon")
    for d, nm in ((full, "full_margin"), (zero, "rho_estimation_term"),
                  (gap, "noise_filtering_term")):
        assert list(d["horizon"]) == list(H), f"{nm} rows incomplete"
    assert set(full["reference"]) == {"damped_persistence_reg"}, "r0 reference changed"

    y_full = full["skill_pct"].to_numpy()
    y_zero = zero["skill_pct"].to_numpy()
    y_gap = gap["skill_pct"].to_numpy()
    assert_source("F4a filter with observation noise, lead 1 (vs damped, reg)",
                  y_full[:1], [6.206], tol=0.001)
    assert_source("F4a filter with r forced to zero, lead 1", y_zero[:1], [0.900],
                  tol=0.001)
    assert_source("F4a skill given up when r = 0, lead 1", y_gap[:1], [5.354],
                  tol=0.001)
    assert_source("F4a skill given up when r = 0, leads 1-6", y_gap,
                  [5.354, 9.633, 11.667, 12.481, 12.653, 12.272], tol=0.001)

    # (b) two observation-noise variances (one per mission) against one
    split = ms[ms["component"] == "mission_split_term"].copy()
    split["horizon"] = split["horizon"].astype(int)
    split = split.sort_values("horizon")
    assert list(split["horizon"]) == list(H), "mission-split rows incomplete"
    y_ms = split["skill_pct"].to_numpy()
    lo_ms = split["ci_lo_pct"].to_numpy()
    hi_ms = split["ci_hi_pct"].to_numpy()
    assert_source("F4b mission-split filter over the one-variance filter, lead 1 "
                  "= -1.84 %", y_ms[:1], [-1.837], tol=0.005)
    assert (y_ms < 0).all() and (hi_ms < 0).all(), (
        "the mission split is no longer a loss at every lead"
    )

    # (c) per-basin lead-1 skill of the filter over damped persistence, in cm
    h1 = cmet[cmet["horizon"] == 1].pivot(index="name", columns="model",
                                          values="rmse_cm")
    assert len(h1) == 234, f"per-basin metrics cover {len(h1)} basins, not 234"
    sk = (1.0 - (h1["kalman_ar1"] / h1["damped_persistence"]) ** 2) * 100.0
    n_pos, n_neg = int((sk > 0).sum()), int((sk < 0).sum())
    assert n_pos + n_neg == 234
    print(f"    per-basin lead-1: filter better in {n_pos} of 234 basins")

    VLIM = 20.0
    fig = plt.figure(figsize=(W2, 14.6 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.30], left=0.068,
                          right=0.985, top=0.955, bottom=0.030, wspace=0.20,
                          hspace=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    axa.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    axa.fill_between(H, y_zero, y_full, color=BATLOW["navy"], alpha=0.10, lw=0,
                     zorder=3)
    axa.plot(H, y_full, ls="-", marker="o", color=BATLOW["navy"], lw=1.7,
             label="Kalman reference (observation noise estimated)", zorder=5)
    axa.plot(H, y_zero, ls="--", marker="s", color=BATLOW["gold"], lw=1.4,
             label="same filter, observation noise forced to zero", zorder=5)
    for h, a, b in zip(H, y_zero, y_full):
        axa.annotate("", xy=(h, b), xytext=(h, a),
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55"), zorder=4)
    mid = int(np.argmax(y_gap))
    from matplotlib.patheffects import withStroke
    axa.annotate("the shaded gap is the noise term:\n"
                 f"+{y_gap[0]:.1f} % at lead 1, +{y_gap[mid]:.1f} % at lead {H[mid]}",
                 xy=(0.025, 0.02), xycoords="axes fraction", ha="left",
                 va="bottom", fontsize=7, color="0.20", zorder=7,
                 path_effects=[withStroke(linewidth=2.0, foreground="white")])
    lead_axis(axa)
    axa.set_ylabel("skill vs damped persistence (%)")
    axa.set_ylim(-13.8, 15.5)
    axa.set_yticks([-10, -5, 0, 5, 10])
    axa.legend(loc="upper left", bbox_to_anchor=(-0.015, 1.01), handlelength=2.2,
               fontsize=7.2)
    panel_head(axa, "(a)", "the observation-noise term")

    axb.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    axb.fill_between(H, lo_ms, hi_ms, color=BATLOW["teal"], alpha=0.16, lw=0, zorder=3)
    axb.plot(H, y_ms, ls="-", marker="D", color=BATLOW["teal"], lw=1.5, zorder=5)
    axb.annotate("one variance per mission is worse at every lead,\n"
                 "and the CI never reaches zero",
                 xy=(0.975, 0.32), xycoords="axes fraction", ha="right",
                 va="top", fontsize=7, color="0.20")
    lead_axis(axb)
    axb.set_ylabel("skill of the two-variance filter\nover the one-variance filter (%)")
    axb.set_ylim(-3.0, 0.7)
    panel_head(axb, "(b)", "a separate GRACE-FO noise variance")

    axc = world_axes(fig, gs[1, :])
    norm = mcolors.TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)
    draw_choropleth(axc, paint(sk.to_dict()), norm, cm.vik)
    hcolorbar(fig, axc, norm, cm.vik,
              "per-basin lead-1 skill of the Kalman reference over damped "
              "persistence (%), from RMSE in cm\nred: the filter is better;  "
              "blue: damped persistence is better;  scale clipped at "
              f"$\\pm${VLIM:.0f} %", shrink=0.62, pad=0.04)
    panel_head(axc, "(c)", f"where the filter helps: better in {n_pos} of 234 "
               f"basins, worse in {n_neg}", y=1.02)

    note(stem, "Panels (a) and (b) are scored against the regression damped "
               "variant at every lead, which is the reference r0_ablation_summary.csv "
               "and kalman_mission_summary.csv use. Fig. 2 instead uses the "
               "stronger variant per lead, so the lead-1 Kalman value differs "
               "(+6.21 % here, +4.98 % there).")
    note(stem, "Panel (c) is a per-basin RMSE ratio in raw cm, not the pooled "
               "standardized skill, and the colour scale is clipped at "
               f"{VLIM:.0f} % (both ends extended).")
    save(fig, stem)
    return dict(n_pos=n_pos, n_neg=n_neg)


# ===========================================================================
# Figure 5 -- where the ERA5 window pays
# ===========================================================================

def fig05_era5_where():
    stem = "fig05_era5_where"
    print(f"{stem}")
    from scipy.stats import spearmanr
    from gracefc.stats import per_basin_dm_fdr
    from gracefc.features import pivot_wide
    from gracefc.evaluate import DEFAULT_FOLDS, deseasonalize_fold

    pred = pd.read_csv(RESULTS / "flat12_ridge_predictions.csv")
    h1 = pred[pred["horizon"] == 1]
    assert set(["kalman_ar1", "ridge_own_era5_flat12"]) <= set(h1["model"]), (
        "flat12 predictions missing a model"
    )

    # per-basin lead-1 skill of the flat-12 ERA5 ridge over the Kalman reference
    wide = h1.pivot_table(index=["name", "target_date"], columns="model",
                          values="pred")
    tgt = (h1[h1["model"] == "kalman_ar1"].set_index(["name", "target_date"])["target"]
           .loc[wide.index])
    loss_e = (tgt - wide["ridge_own_era5_flat12"]) ** 2
    loss_k = (tgt - wide["kalman_ar1"]) ** 2
    skill = (1.0 - loss_e.groupby("name").mean() / loss_k.groupby("name").mean()) * 100.0
    pooled = 100.0 * (1.0 - loss_e.mean() / loss_k.mean())
    assert len(skill) == 234, f"per-basin skill covers {len(skill)} basins"
    assert_source("F5 pooled lead-1 skill of the flat-12 ERA5 ridge over the "
                  "Kalman reference = +7.65 %", [pooled], [7.646], tol=0.005)

    # per-basin Diebold-Mariano with a HAC lag of h-1 (the repository helper),
    # then Benjamini-Hochberg at q = 0.10
    fdr = per_basin_dm_fdr(h1, "ridge_own_era5_flat12", "kalman_ar1", 1, q=0.10)
    assert len(fdr) == 234, f"DM ran on {len(fdr)} basins"
    sig = fdr[fdr["significant"]]
    sig_names = set(sig["name"])
    n_sig = len(sig_names)
    n_sig_better = int((sig["a_better"]).sum())
    n_sig_worse = n_sig - n_sig_better
    assert_source("F5 BH-FDR q=0.10: significant / better / worse",
                  [n_sig, n_sig_better, n_sig_worse], [45, 44, 1], tol=0.0)
    # the DM sign must agree with the plotted skill for every significant basin
    agree = [(skill[n] > 0) == bool(fdr.set_index("name").loc[n, "a_better"])
             for n in sig_names]
    assert all(agree), "DM sign and per-basin skill disagree on a significant basin"

    # training-window std of the deseasonalized target, in cm, exactly as the
    # pipeline computes it (evaluate.deseasonalize_fold), median over the folds
    meta = pd.read_csv(PROC / "basin_meta.csv")
    keep = set(meta.loc[meta["exclude_reason"] == "keep", "name"])
    long = pd.read_csv(PROC / "basin_month_twsa_global.csv", parse_dates=["date"])
    wide_t = pivot_wide(long[long["name"].isin(keep)])
    assert wide_t.shape[1] == 234, f"target matrix has {wide_t.shape[1]} basins"
    per_fold = pd.DataFrame({f.name: deseasonalize_fold(wide_t, f)[1]
                             for f in DEFAULT_FOLDS})
    train_std = per_fold.median(axis=1)
    assert train_std.notna().all() and (train_std > 0).all()

    common = skill.index.intersection(train_std.index)
    assert len(common) == 234
    x = train_std.loc[common].to_numpy()
    y = skill.loc[common].to_numpy()
    rho, p_rho = spearmanr(x, y)
    assert_source("F5b Spearman rho between training-window std and ERA5 gain",
                  [rho], [-0.1436], tol=0.0005)
    assert rho < 0, "the ERA5 gain no longer concentrates in low-variance basins"

    VLIM = 35.0
    fig = plt.figure(figsize=(W2, 8.6 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.0], left=0.030, right=0.978,
                          top=0.915, bottom=0.205, wspace=0.17)
    axa = world_axes(fig, gs[0])
    norm = mcolors.TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)
    draw_choropleth(axa, paint(skill.to_dict()), norm, cm.vik)
    hatch_basins(axa, sig_names)
    hcolorbar(fig, axa, norm, cm.vik,
              "lead-1 skill of the flat-12 ERA5 ridge over the Kalman reference (%)\n"
              f"hatched: {n_sig} of 234 basins pass BH-FDR at $q$ = 0.10 "
              f"({n_sig_better} for, {n_sig_worse} against)",
              shrink=0.88, pad=0.045)
    panel_head(axa, "(a)", "where the ERA5 window pays", y=1.02)

    axb = fig.add_subplot(gs[1])
    axb.grid(True, axis="both")
    axb.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    is_sig = np.array([n in sig_names for n in common])
    axb.scatter(x[~is_sig], y[~is_sig], s=9, facecolor="none", edgecolor="0.60",
                linewidth=0.55, zorder=4, label="not significant")
    axb.scatter(x[is_sig], y[is_sig], s=11, color=BATLOW["navy"], edgecolor="none",
                zorder=5, label=f"BH-FDR significant ($q$ = 0.10)")
    # the median is printed in the build notes; no guide line is drawn
    med = float(np.median(x))
    axb.set_xscale("log")
    axb.set_xlim(0.6, 60)
    YLO, YHI = -42.0, 54.0
    axb.set_ylim(YLO, YHI)
    axb.set_yticks([-40, -20, 0, 20, 40])
    axb.set_xticks([1, 3, 10, 30])
    axb.set_xticklabels(["1", "3", "10", "30"])
    # a median guide line was tried here and removed: it crossed the densest
    # part of the cloud and the Spearman coefficient already carries the claim
    axb.set_xlabel("training-window standard deviation\n"
                   "of the deseasonalized target (cm)")
    axb.set_ylabel("lead-1 skill over the Kalman reference (%)")
    n_off = int((y < YLO).sum())
    tail = f"\n{n_off} basin{'s' if n_off != 1 else ''} below the axis" if n_off else ""
    axb.annotate(f"Spearman $\\rho$ = {rho:+.2f} ($p$ = {p_rho:.3f}, $n$ = 234)"
                 f"{tail}", xy=(0.97, 0.035), xycoords="axes fraction",
                 ha="right", va="bottom", fontsize=7.5, color="0.15")
    axb.legend(loc="upper left", bbox_to_anchor=(-0.01, 1.005), fontsize=7,
               handletextpad=0.3, borderaxespad=0.0)
    panel_head(axb, "(b)", "the gain is a low-variance effect")

    note(stem, "Significance is a per-basin Diebold-Mariano test on the monthly "
               "squared-loss differential with a Newey-West HAC lag of max(h-1, 1) "
               "(gracefc.stats.per_basin_dm_fdr, the same routine the per-basin "
               "FDR scripts use), then Benjamini-Hochberg at q = 0.10.")
    note(stem, "The x axis of panel (b) is the median, across the five "
               "expanding-window folds, of the per-basin training-window standard "
               "deviation of the deseasonalized target, recomputed with "
               "gracefc.evaluate.deseasonalize_fold. The per-fold values move by a "
               "median of 13 % across folds, so the median is used as the single "
               "basin-level scale.")
    save(fig, stem)
    return dict(n_sig=n_sig, n_sig_better=n_sig_better, rho=rho, p=p_rho,
                pooled=pooled)


# ===========================================================================
# Figure 6 -- the sequence models against the flat-12 ridge
# ===========================================================================

def fig06_sequence_models():
    stem = "fig06_sequence_models"
    print(f"{stem}")
    from gracefc.stats import block_bootstrap_skill_ci

    summ = pd.read_csv(RESULTS / "phase7_lstm_summary.csv")
    pred = pd.read_csv(RESULTS / "phase7_lstm_predictions.csv")
    sens = pd.read_csv(RESULTS / "flat12_train85_sensitivity.csv")
    con = pd.read_csv(RESULTS / "paper_baseline_contrasts.csv")
    L3 = np.array([1, 2, 3])

    # the two-seed LSTM ensemble is the mean of the seed predictions; build it
    # and verify against the published lstm_own_era5_ens used by the
    # equalized-training-window sensitivity run.
    keys = ["name", "issue_date", "target_date", "horizon", "fold"]
    ens = (pred[pred["model"].isin(["lstm_own_era5_s0", "lstm_own_era5_s1"])]
           .groupby(keys, as_index=False)
           .agg(target=("target", "first"), pred=("pred", "mean")))
    assert (pred[pred["model"] == "lstm_own_era5_s0"].shape[0] == ens.shape[0]), (
        "the seed mean did not pair one-to-one"
    )
    ens["model"] = "lstm_own_era5_ens"
    allpred = pd.concat([pred, ens], ignore_index=True)

    def rmse_of(model, horizon):
        s = allpred[(allpred["model"] == model) & (allpred["horizon"] == horizon)]
        return float(np.sqrt(((s["target"] - s["pred"]) ** 2).mean()))

    r_ens = np.array([rmse_of("lstm_own_era5_ens", h) for h in L3])
    r_flat = np.array([rmse_of("ridge_own_era5_flat12", h) for h in L3])
    # ridge_own_era5_flat12_train85 has no prediction rows here; reconstruct its
    # RMSE from its stored skill against the full-window twin, then check that
    # the published ensemble comparison falls out of our reconstructed ensemble.
    s85 = series(sens[sens["reference"] == "ridge_own_era5_flat12"],
                 {"challenger": "ridge_own_era5_flat12_train85"}, "skill_pct",
                 leads=L3)
    r85 = r_flat * np.sqrt(1.0 - s85 / 100.0)
    recon = (1.0 - (r85 / r_ens) ** 2) * 100.0
    published = series(sens[sens["reference"] == "lstm_own_era5_ens"],
                       {"challenger": "ridge_own_era5_flat12_train85"}, "skill_pct",
                       leads=L3)
    assert_source("F6 two-seed ensemble reproduces the published "
                  "lstm_own_era5_ens comparison", recon, published, tol=0.002)
    assert_source("F6b equalized-window flat-12 ERA5 ridge over the LSTM "
                  "ensemble, leads 1-3", published, [1.02, 2.76, 2.27], tol=0.005)

    # panel (a): skill over the Kalman reference, with block-bootstrap CIs
    arms = ["ridge_own_era5", "ridge_own_flat12", "ridge_own_era5_flat12",
            "mlp_own_era5_flat12_s0", "mlp_own_era5_flat12_s1", "lstm_own_era5_ens"]
    pt, lo, hi = {}, {}, {}
    for m in arms:
        vals = [block_bootstrap_skill_ci(allpred, m, "kalman_ar1", int(h), seed=0)
                for h in L3]
        pt[m] = np.array([v[0] for v in vals]) * 100.0
        lo[m] = np.array([v[1] for v in vals]) * 100.0
        hi[m] = np.array([v[2] for v in vals]) * 100.0

    # the point estimates must reproduce the summary file's rmse_std ratios
    piv = summ.pivot(index="model", columns="horizon", values="rmse_std")
    for m in ["ridge_own_era5", "ridge_own_flat12", "ridge_own_era5_flat12",
              "mlp_own_era5_flat12_s0", "mlp_own_era5_flat12_s1"]:
        from_summary = (1.0 - (piv.loc[m, L3] / piv.loc["kalman_ar1", L3]) ** 2
                        ).to_numpy() * 100.0
        assert_source(f"F6a {m} == rmse_std in phase7_lstm_summary.csv", pt[m],
                      from_summary, tol=0.002)
    # and the flat-12 arms must match the published contrasts against the filter
    for m in ["ridge_own_era5_flat12", "ridge_own_flat12"]:
        c = series(con[(con["reference"] == "kalman_ar1") & (con["horizon"] <= 3)],
                   {"challenger": m}, "skill", leads=L3) * 100.0
        assert_source(f"F6a {STYLE[m]['label']} == paper_baseline_contrasts.csv",
                      pt[m], c, tol=0.01)
    assert_source("F6a flat-12 ERA5 ridge over the Kalman reference, lead 1 = +7.65 %",
                  pt["ridge_own_era5_flat12"][:1], [7.65], tol=0.006)
    assert_source("F6a LSTM ensemble over the Kalman reference, leads 1-3",
                  pt["lstm_own_era5_ens"], [6.534, 2.743, 2.186], tol=0.005)

    LOOK = {
        "ridge_own_era5":         dict(c=BATLOW["olive"],  m="<", label="lag-feature ERA5 ridge"),
        "ridge_own_flat12":       dict(c=BATLOW["green"],  m="s", label="flat-12 ridge"),
        "ridge_own_era5_flat12":  dict(c=BATLOW["gold"],   m="D", label="flat-12 ERA5 ridge"),
        "mlp_own_era5_flat12_s0": dict(c=BATLOW["orange"], m="P", label="residual MLP, flat-12 window (seed 0)"),
        "mlp_own_era5_flat12_s1": dict(c=BATLOW["pink"],   m="X", label="residual MLP, flat-12 window (seed 1)"),
        "lstm_own_era5_ens":      dict(c=BATLOW["blue"],   m="h", label="LSTM ensemble, own state + ERA5"),
    }
    OFF = np.linspace(-0.27, 0.27, len(arms))

    fig = plt.figure(figsize=(W2, 8.0 * CM))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], left=0.062, right=0.988,
                          top=0.920, bottom=0.265, wspace=0.24)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    axa.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    for m, off in zip(arms, OFF):
        k = LOOK[m]
        axa.errorbar(L3 + off, pt[m], yerr=[pt[m] - lo[m], hi[m] - pt[m]],
                     ls="none", marker=k["m"], color=k["c"], mfc=k["c"],
                     mec=k["c"], ms=4.6, elinewidth=1.1, capsize=0, zorder=5,
                     label=k["label"])
    axa.annotate("zero line: the Kalman reference", xy=(0.985, 0.03),
                 xycoords="axes fraction", ha="right", va="bottom", fontsize=7,
                 color=REFGRAY)
    axa.set_xticks(L3)
    axa.set_xlim(0.6, 3.45)
    axa.set_xlabel("forecast lead $h$ (months)")
    axa.set_ylabel("skill over the Kalman reference (%)")
    axa.set_ylim(-4.5, 11.5)
    panel_head(axa, "(a)", "every sequence model sits behind the flat-12 ERA5 ridge")

    axb.axhline(0.0, color=REFGRAY, lw=0.9, zorder=2)
    dmp = series(sens[sens["reference"] == "lstm_own_era5_ens"],
                 {"challenger": "ridge_own_era5_flat12_train85"}, "dm_p", leads=L3)
    bars = axb.bar(L3, published, width=0.56, color=BATLOW["gold"],
                   edgecolor="none", zorder=4)
    for h, v, pv in zip(L3, published, dmp):
        lab = f"+{v:.2f}%\n$p$ = {pv:.2g}" if pv >= 1e-4 else f"+{v:.2f}%\n$p$ < 1e-4"
        axb.annotate(lab, xy=(h, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7, color="0.15")
    axb.set_xticks(L3)
    axb.set_xlim(0.5, 3.5)
    axb.set_ylim(0, 4.4)
    axb.set_xlabel("forecast lead $h$ (months)")
    axb.set_ylabel("skill of the flat-12 ERA5 ridge\nover the LSTM ensemble (%)")
    panel_head(axb, "(b)", "training window equalized")

    handles, labels = axa.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, handlelength=1.2, columnspacing=1.6, handletextpad=0.5,
               fontsize=7.5)

    note(stem, "Leads 1-3 only: the LSTM and residual-MLP arms in "
               "phase7_lstm_summary.csv were never run past lead 3.")
    note(stem, "Error bars are 95 % moving-block bootstrap CIs recomputed from "
               "phase7_lstm_predictions.csv with gracefc.stats."
               "block_bootstrap_skill_ci; the summary file stores no CI. The "
               "point estimates are asserted against its rmse_std column.")
    note(stem, "The LSTM ensemble is the mean of the two seed predictions. That "
               "reconstruction reproduces the published lstm_own_era5_ens "
               "comparison in flat12_train85_sensitivity.csv to 0.002 %.")
    note(stem, "Panel (b) is the equalized-training-window run: the flat-12 ERA5 "
               "ridge refit on 85 % of the window the LSTM sees, so neither model "
               "has a history-length advantage.")
    save(fig, stem)
    return dict(published=published, dmp=dmp)


# ===========================================================================
# Build notes
# ===========================================================================

def write_build_notes(f1, f3x, f4, f5, f6):
    lines = []
    A = lines.append
    A("# Figure build notes")
    A("")
    A("Generated by `scripts/make_figures.py`. Every figure is one file: a vector")
    A("PDF for the manuscript and a 150 dpi PNG preview for review. Rasterized")
    A("map layers are written at 300 dpi inside the PDF (`savefig.dpi = 300`), so")
    A("the PDFs meet the HESS 300 dpi floor; the PNGs are previews only.")
    A("")
    A("Conventions, per `paper/notes/FIGURE_PLAN.md`: cmcrameri `vik` for diverging")
    A("maps, cmcrameri `batlow` samples for categorical series, BH-FDR significant")
    A("basins hatched rather than the insignificant ones stippled (Wilks 2016),")
    A("panel labels `(a)`/`(b)` lowercase in brackets, leads on integer ticks, a")
    A("zero line wherever skill is plotted, and no default matplotlib colours.")
    A("")
    A("Sign convention: `skill = 1 - MSE(first)/MSE(second)`, so positive means the")
    A("first-named model is better.")
    A("")
    A("Model names are fixed across figures: Kalman reference, own-state ridge,")
    A("flat-12 ridge, flat-12 ERA5 ridge, per-basin ridge, lag-feature ridge,")
    A("lag-feature ERA5 ridge, damped persistence, persistence, climatology,")
    A("residual MLP (flat-12 window), LSTM ensemble, published product (full),")
    A("published product (non-seasonal). No code identifier appears in a legend.")
    A("")

    def block(stem, width, shows, sources, asserts):
        A(f"## `{stem}`")
        A("")
        A(f"- **Print width**: {width}")
        A(f"- **Shows**: {shows}")
        A("- **Sources**:")
        for s in sources:
            A(f"  - `{s}`")
        A("- **Asserted against source**:")
        for a in asserts:
            A(f"  - {a}")
        caveats = [t for st, t in CAPTION_NOTES if st == stem]
        if caveats:
            A("- **Caveats the caption must carry**:")
            for c in caveats:
                A(f"  - {c}")
        A("")

    block("fig01_basins", "double column (17 cm), two stacked map panels",
          "the 234-basin sample, split by how each basin enters the comparison "
          "against the published product, on CSR (a) and JPL (b)",
          ["HydroShed+Mascon_Basins_L3.nc (via gracefc.basins.load_basin_masks)",
           "data/processed/basin_meta.csv",
           "data/processed/li2026_basin_coverage.csv",
           "results/phase6_li_comparison_perbasin.csv",
           "results/phase6_li_comparison_summary.csv",
           "results/jpl/phase6_li_comparison_perbasin.csv",
           "results/jpl/phase6_li_comparison_summary.csv"],
          [f"strict CSR subset n = {f1['strict']} (expected 209)",
           f"matched but excluded by the strict rule n = {f1['loose']} (expected 18)",
           f"no usable published forecast n = {f1['unmatched']} (expected 7)",
           f"strict JPL subset n = {f1['jpl']} (expected 67)",
           "the strict roster equals the set of basins actually scored in "
           "`phase6_li_comparison_perbasin.csv`",
           "209 / 227 / 67 also read back from the two comparison summary files"])

    block("fig02_benchmark_ladder", "double column (17 cm), two panels",
          "(a) skill against damped persistence by lead for the eight "
          "conventional-ladder models, with the filter family drawn solid; "
          "(b) the same models ranked by pooled RMSE in raw cm",
          ["results/paper_baseline_ladder.csv",
           "results/paper_baseline_contrasts.csv",
           "results/conventional_metrics_summary.csv"],
          ["Kalman reference lead-1 skill +4.98 % (and +8.79/+5.62/+3.07/+2.55/"
           "+3.63 % at leads 2-6)",
           "flat-12 ERA5 ridge +12.24/+13.71/+9.61/+6.16/+4.58/+4.47 %",
           "pooled lead-1 RMSE 5.128 cm (Kalman reference), 5.118 cm (flat-12 "
           "ridge), 5.231 cm (flat-12 ERA5 ridge)",
           "every CI ribbon's point estimate equals the ladder curve it wraps",
           "`damped_ref` is rho at lead 1 and the regression variant at leads 2-6"])

    cross = ", ".join(f"{k[0]} {STYLE[k[1]]['label']} vs {STYLE[k[2]]['label']}: "
                      f"h = {v:.2f}" for k, v in f3x.items() if v is not None)
    block("fig03_crossing", "double column (17 cm), two panels, shared y axis",
          "skill of our models over the published product by lead, with 95 % CIs, "
          "on CSR (a) and JPL (b); the crossover from ahead to behind sits between "
          "leads 2 and 3 on both products",
          ["results/phase6_li_comparison_headline.csv",
           "results/jpl/phase6_li_comparison_headline.csv",
           "results/jpl/phase6_li_comparison_summary.csv"],
          ["CSR Kalman reference over the published product (full), lead 1 = "
           "+16.354 %",
           "CSR flat-12 ERA5 ridge over the published product (non-seasonal), "
           "leads 1-2 = +32.781 / +11.231 %",
           "JPL Kalman reference over the published product (full), lead 1 = "
           "+43.718 %, inverted from the stored -77.678 % and cross-checked "
           "against rmse_std",
           "for all four CSR pairs, the value the file stores with our model "
           "first equals the reciprocal inversion of the reverse-direction row",
           "CSR strict sample 12540 rows / 60 months / 209 basins; JPL 3953 "
           "rows / 59 months",
           f"interpolated zero crossings: {cross}"])

    block("fig04_filter_mechanism", "double column (17 cm), three panels",
          "(a) the filter with and without the observation-noise term; (b) the "
          "two-variance mission-split filter against the one-variance filter; "
          "(c) per-basin lead-1 skill of the filter over damped persistence",
          ["results/r0_ablation_summary.csv",
           "results/kalman_mission_summary.csv",
           "results/conventional_metrics_perbasin.csv",
           "HydroShed+Mascon_Basins_L3.nc"],
          ["the noise term is worth +5.354 % at lead 1 and +9.633/+11.667/"
           "+12.481/+12.653/+12.272 % at leads 2-6",
           "filter with observation noise +6.206 % at lead 1, with r forced to "
           "zero +0.900 %, both against the regression damped variant",
           "mission-split filter -1.837 % at lead 1, and its CI stays below zero "
           "at every lead",
           f"per-basin map covers 234 basins; the filter is better in "
           f"{f4['n_pos']} and worse in {f4['n_neg']}"])

    block("fig05_era5_where", "double column (17 cm), two panels",
          "(a) per-basin lead-1 skill of the flat-12 ERA5 ridge over the Kalman "
          "reference, with BH-FDR significant basins hatched; (b) the same skill "
          "against each basin's training-window storage variability",
          ["results/flat12_ridge_predictions.csv",
           "data/processed/basin_month_twsa_global.csv",
           "data/processed/basin_meta.csv",
           "src/gracefc/stats.py (per_basin_dm_fdr)",
           "src/gracefc/evaluate.py (deseasonalize_fold, DEFAULT_FOLDS)",
           "HydroShed+Mascon_Basins_L3.nc"],
          [f"pooled lead-1 skill +{f5['pooled']:.3f} % (expected +7.646 %, the "
           "+7.65 % quoted in docs/STUDY_CONTEXT.md)",
           f"BH-FDR q = 0.10: {f5['n_sig']} of 234 basins significant, "
           f"{f5['n_sig_better']} of them in our favour",
           f"Spearman rho = {f5['rho']:+.4f} (p = {f5['p']:.4f}), pinned to -0.1436",
           "the DM sign agrees with the plotted per-basin skill for every "
           "significant basin"])

    pubs = " / ".join(f"+{v:.2f} %" for v in f6["published"])
    block("fig06_sequence_models", "double column (17 cm), two panels",
          "(a) leads 1-3 skill over the Kalman reference for the ridge, MLP and "
          "LSTM arms; (b) the same flat-12 ERA5 ridge against the LSTM ensemble "
          "once the training window is equalized",
          ["results/phase7_lstm_summary.csv",
           "results/phase7_lstm_predictions.csv",
           "results/flat12_train85_sensitivity.csv",
           "results/paper_baseline_contrasts.csv",
           "src/gracefc/stats.py (block_bootstrap_skill_ci)"],
          ["every point estimate reproduces the rmse_std ratio in "
           "`phase7_lstm_summary.csv` to 0.002 %",
           "the flat-12 arms also reproduce their rows in "
           "`paper_baseline_contrasts.csv` against `kalman_ar1`",
           "flat-12 ERA5 ridge over the Kalman reference, lead 1 = +7.65 %",
           "LSTM ensemble over the Kalman reference = +6.534/+2.743/+2.186 %",
           f"equalized-window flat-12 ERA5 ridge over the LSTM ensemble = {pubs} "
           "(expected +1.02 / +2.76 / +2.27 %)",
           "the seed-mean ensemble reproduces the published `lstm_own_era5_ens` "
           "comparison to 0.002 %"])

    A("## Things this build could not do as specified")
    A("")
    A("- The JPL panel of Fig. 1 cannot show the 228 basins the JPL run kept: the")
    A("  JPL `basin_meta.csv` is on the collaborator's machine and only the")
    A("  compact headline, summary and per-basin CSVs are versioned here. The")
    A("  panel therefore shades the 67 strict basins and leaves the other 167 of")
    A("  the 234 CSR-kept basins neutral, and says so on the figure.")
    A("- Fig. 6 covers leads 1-3 only. The LSTM and residual-MLP arms were never")
    A("  run past lead 3, so there is no lead 4-6 row to plot.")
    A("- `results/phase7_lstm_summary.csv` has no `skill_vs_kalman` column and no")
    A("  CI columns, so Fig. 6 recomputes both from")
    A("  `results/phase7_lstm_predictions.csv` and asserts the point estimates")
    A("  back against the summary file's `rmse_std`.")
    A("- The JPL skill is inverted with the reciprocal transform")
    A("  `1 - 1/(1 - s)`, not by negating the stored skill. Negation would report")
    A("  +77.7 % where the RMSEs say +43.7 %. The inversion is asserted against")
    A("  `rmse_std` in `results/jpl/phase6_li_comparison_summary.csv`.")
    A("")
    (FIGURES / "BUILD_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  wrote figures/BUILD_NOTES.md")


# ===========================================================================

def main():
    FIGURES.mkdir(exist_ok=True)
    f1 = fig01_basins()
    fig02_benchmark_ladder()
    f3x = fig03_crossing()
    f4 = fig04_filter_mechanism()
    f5 = fig05_era5_where()
    f6 = fig06_sequence_models()
    write_build_notes(f1, f3x, f4, f5, f6)
    print("\nall six figures built; every source assert passed")


if __name__ == "__main__":
    main()
