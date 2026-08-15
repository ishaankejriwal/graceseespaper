# Figure plan — GRACE TWSA noise-filtering benchmark paper

Seven figures, none generated yet. Global conventions:

- **Colormaps**: cmcrameri `vik` for all diverging maps/scatters (helped/hurt,
  win/lose), cmcrameri `batlow` for categorical/sequential series (model
  curves, distance bands). Never red-green pairs outside `vik`.
- **Significance on maps**: hatching (not stippling of insignificant cells) for
  BH-FDR q=0.10 significant basins, per Wilks (2016, BAMS) — hatch the
  *significant* basins, state the FDR level in the caption.
- **Multi-panel figures are single files** (one PDF each, panels composed in
  matplotlib; no LaTeX subfloat).
- Vector PDF output; grayscale-legible line styles (solid/dashed/dotted +
  markers) so no curve is distinguished by color alone.
- All skills in % vs the stated reference; leads on the x-axis are h = 1..6.
- Every panel must be reproducible from `results/*.csv` alone (notebook to be
  added as `notebooks/03_paper_figures.ipynb`).

---

## F1 — `fig01_benchmark_ladder.pdf` (Sect. 4.1, referenced as Fig. \ref{fig:ladder})

**Shows**: the benchmark ladder as skill-vs-lead curves; the visual argument
that the filtering class sits above every conventional reference at all leads.

- **Source**: `results/paper_baseline_ladder.csv`
  (columns: `model`, `horizon`, `skill_vs_damped`, `dm_p_vs_damped`) and
  `results/paper_baseline_contrasts.csv` (columns `challenger`, `reference`,
  `horizon`, `skill`, `ci_lo`, `ci_hi`) for the CI ribbons on the two
  filtering-class curves (`kalman_ar1 vs damped`, and per-basin-ridge contrast).
- **Plot**: line plot, x = lead (1–6), y = skill vs stronger damped variant
  (%). Curves: persistence, damped persistence (zero line), pooled ridge,
  per-basin ridge, kalman_ar1 (bold), kalman_own_ridge (bold, dashed).
  Climatology omitted (off-scale; note in caption). Shaded 95% CI ribbon on
  kalman_ar1 only (from `paper_baseline_contrasts.csv`, kalman_ar1 vs damped
  rows, `ci_lo`/`ci_hi`).
- **Layout**: single panel, ~12 cm wide.
- **Colors**: batlow discrete samples; kalman curves darkest.
- **Annotation**: bracket at h1–2 marking the 4.8–8.1% noise-propagation gap.

## F2 — `fig02_crossing.pdf` (Sect. 4.3, Fig. \ref{fig:crossing})

**Shows**: the initial-conditions/forcing crossover against GRACE-FCast, with
uncertainty, and its robustness to their seasonal/trend edge.

- **Source**: `results/phase8b_li_comparison_headline.csv`
  (rows `lstmres_corr_top1_ens` vs `li_lstm_full`: `skill_pct... ` — note this
  file stores skill as fraction in `skill`, CI in `ci_lo`/`ci_hi`; subset
  `all_matched`), same file rows vs `li_lstm_nonseas`, and rows `kalman_ar1`
  vs `li_lstm_full`. DM p per lead from `dm_p`.
- **Plot**: x = lead, y = skill of ours vs theirs (%). Main curve:
  `lstmres_corr_top1_ens` vs `li_lstm_full` with block-bootstrap 95% band
  (ci_lo/ci_hi). Secondary thin curves: (i) `kalman_ar1` vs `li_lstm_full`
  (the three-parameter filter alone), (ii) `lstmres_corr_top1_ens` vs
  `li_lstm_nonseas`. Horizontal zero line; vertical dashed line between h2
  and h3 labeled "crossover". Filled markers where DM p < .05, open where ns
  (h2 of the main curve is the one open marker).
- **Layout**: single panel, ~12 cm.
- **Colors**: batlow; main curve darkest.
- **Caption must carry**: 227 basins, 61 months, fold 5 excluded, per-basin
  win counts 133/81/69/65/57/57 (from `phase8b_li_comparison_perbasin.csv`).

## F3 — `fig03_neighbor_map.pdf` (Sect. 4.4, Fig. \ref{fig:map})

**Shows**: global choropleth of the per-basin lead-1 neighbor effect with FDR
hatching; the 9-helped/7-hurt two-sided geography.

- **Source**: `results/phase5_perbasin_fdr_h1.csv` (columns: `name`,
  `dm_stat`, `p`, `significant`) joined to basin polygons/centroids via
  `results/phase6_basin_analysis_summary.csv` (per-basin covariates incl.
  continent, area, and skill columns) and the basin mask geometry in
  `data/processed/` (basin_meta for centroids).
- **Plot**: world map (Robinson or Equal Earth), basins filled by per-basin
  DM statistic (sign flipped so positive = neighbor helps), diverging `vik`
  centered at 0, clipped at ±3. Hatch the 16 FDR-significant basins
  (`significant == True`); annotate by name the 7 hurt (Yenisey, Paraná,
  Yangtze, Amur, E. Barents Sea Coast, Aleutians/S. Alaska, Gulf of
  Oman/W. Arabian Sea) and the Africa winner cluster (Niger, Lake Rukwa,
  W. Nile Delta, SW Mediterranean Coast).
- **Layout**: single map panel + horizontal colorbar; ~17 cm (two-column).
- **Alternative considered**: plotting per-basin skill % instead of DM —
  rejected; per-basin skill on 84 months has heavy tails (to −30%), DM is
  the stable statistic (phase6_basin_analysis.md caveats).

## F4 — `fig04_controls.pdf` (Sect. 4.4, Fig. \ref{fig:controls})

**Shows**: the null-model battery in one figure — placebo distribution,
surrogate distribution, distance profile, conditioning invariance.

- **Panel (a) placebos + surrogates**: two stacked histogram strips.
  Top: 50 random-graph pooled RMSEs at h1 from
  `results/phase3b_placebo_monthly.csv` (aggregate the per-month placebo
  losses to pooled RMSE per seed; family `corr_top1`), with the real
  `kalman_corr_top1` RMSE (1.04268, from `phase3b_summary.csv`) as a vertical
  line. Bottom: 99 IAAFT surrogate skills at h1 from
  `results/phase4_surrogate_summary.csv` (that file stores the summary:
  surr_mean/surr_min and beats 99/99; if per-surrogate values are needed use
  the surrogate run artifacts; otherwise draw the summary as a box with
  min/mean and the real value marked).
- **Panel (b) distance profile**: bar/point plot of h1 skill vs own-ridge for
  no-restriction (+0.488%), ≥300 km (+0.502%), ≥500 km (+0.337%), ≥1000 km
  (−0.319%), from `results/phase3b_summary.csv` (`skill_vs_own_ridge`,
  models `kalman_corr_top1`, `kalman_corr_min{300,500,1000}_top1`), with
  placebo-beaten counts (50/50, 50/50, 50/50, 0/50) printed above bars.
- **Panel (c) conditioning invariance**: three points with bootstrap CIs:
  unconditioned +0.488% (phase3b/`phase6_era5_headline.csv` ridge_corr_top1
  vs ridge_own), ENSO/IOD-conditioned +0.497%
  (`phase4_conditioned_summary.csv`, kalman_corr_top1 h1
  `skill_vs_own_ridge`), ERA5-conditioned +0.485% with CI +0.15..+0.86
  (`phase6_era5_headline.csv`, ridge_corr_top1_era5 vs ridge_own_era5).
- **Layout**: 3 panels in one row (a wider), single file, ~17 cm.
- **Colors**: batlow for bars; real-effect line in near-black.

## F5 — `fig05_delivery.pdf` (Sect. 4.5, Fig. \ref{fig:delivery})

**Shows**: the delivery contrast — identical neighbor information as encoder
channel vs as residual-correction stage, across all six leads.

- **Source**: `results/phase8b_h16_headline.csv`:
  - correction increments: rows (`lstmres_corr_top1_s0`, `lstm_own_era5_s0`)
    and (`lstmres_corr_top1_s1`, `lstm_own_era5_s1`), `skill_pct`,
    `ci_lo_pct`, `ci_hi_pct`, h = 1..6;
  - in-encoder channel: rows (`lstm_corr_top1_era5_s0`, `lstm_own_era5_s0`),
    h = 1..6.
  - h1–3 ensemble increments (+0.93/+1.09/+1.27 with CIs) from
    `results/phase8_analysis.md` sect. 2 (recompute in the notebook from
    `phase8_lstm_combined_predictions.csv` for figure use).
- **Placebo band**: seed-matched placebo increments from
  `results/phase8_lstm_combined_placebo_monthly.csv` (h1–3) and
  `results/phase8b_lstm_h46_placebo_monthly.csv` (h4–6): per placebo seed,
  pooled skill relative to `lstm_own_era5_s0`; shade min–max of the 20 draws
  (mean ≈ 0). Caption must note h4–6 draws reuse the h1–3 per-fold graphs
  (not independent).
- **Plot**: x = lead; y = skill increment over same-seed stage-1 (%). Two
  curve families: correction stage (s0, s1 as paired markers, near-identical)
  and in-encoder channel (s0; include s1 as light markers to show
  seed-instability). Placebo band shaded around zero. Error bars = bootstrap
  CIs where available.
- **Layout**: single panel ~12 cm.
- **Colors**: vik extremes for the two delivery mechanisms (they are
  opposite-signed at h2–3), placebo band gray.

## F6 — `fig06_complementarity.pdf` (Sect. 5.3, Fig. \ref{fig:complementarity})

**Shows**: ERA5 gain and neighbor gain cover each other's dead zones
(geographic complementarity; the Africa argument).

- **Source**: `results/phase6_perbasin_era5_skill_h1.csv` (per-basin ERA5
  skill at h1) and `results/phase6_basin_analysis_summary.csv` (per-basin
  neighbor skill/DM, continent). Spearman ρ = −0.22 (p = 7e-4) from
  phase6_analysis.md sect. 7 (recompute in notebook). NOTE (audit fix): the
  −0.22 correlation is against the **ERA5-conditioned** neighbor gain — plot
  the conditioned gain on the y-axis to match the quoted ρ (the unconditioned
  correlation is only −0.121, p=.065; caption in main.tex now says
  "ERA5-conditioned neighbor gain").
- **Plot**: scatter, x = per-basin ERA5 gain (%), y = per-basin ERA5-conditioned
  neighbor gain (%), points colored by continent (batlow categorical), Africa emphasized
  (larger markers). Axes clipped to [−30, +40] with outlier count noted
  (Amazon −99% off-scale, arrow annotation). Marginal density strips.
  Inset table or side bar: continental pooled ERA5 gain (NA +7.8, EU +7.3,
  Asia +6.4, Africa +0.3 ns) vs continental neighbor medians (Africa +1.43,
  Europe −0.54) from phase6_analysis.md sect. 1 / phase6_basin_analysis.md Q3.
- **Layout**: main scatter + inset, single file, ~12 cm.

## F7 (optional, discussion/appendix) — `fig07_geography_concordance.pdf`

**Shows**: the stacked correction re-finds the linear-effect geography — the
strongest indirect control for the stacked arm (given the ≥300 km/conditioning
ports are not run for it).

- **Source**: per-basin DM of `lstmres_corr_top1` vs `lstm_own_era5`
  (recomputed in notebook from `phase8_lstm_combined_predictions.csv` +
  `phase8b_lstm_h46_predictions.csv`) against per-basin linear DM from
  `results/phase5_perbasin_fdr_h1.csv`; annotate Spearman +0.46/+0.51
  (phase8_analysis.md sect. 4).
- **Plot**: scatter with per-seed panels or overlay; Yenisey and the Africa
  winners annotated. Drop if figure count must stay at 6 — the correlation is
  already stated in text.

---

### Figure-to-claim mapping (for the audit)

| Fig | Claim it carries | Table twin |
|---|---|---|
| F1 | benchmark understated by 4.8–8.1% | Table 1 |
| F2 | crossover located at ~2 months | Table 2 |
| F3 | two-sided 9/7 geography, Yenisey | (text) |
| F4 | neighbor survives all nulls | Table 3 |
| F5 | delivery decides; ~+1% all leads | Table 4 |
| F6 | ERA5/neighbor complementarity | (text) |
| F7 | stacked arm re-finds controlled geography | (text) |

---

## ADDENDUM 2026-08-15 — post-audit status per figure (supersedes stale numbers above)

The plan above predates the corrected rerun and the resolution/stratification audits.
Authoritative numbers live in `paper/notes/REWRITE_LEDGER.md`; where this addendum and the
original spec disagree, the addendum wins.

**Buildable NOW (all inputs final):**
- **F1**: sources already corrected (`paper_baseline_ladder.csv` was rebuilt on the
  corrected pipeline). Annotation bracket is **+5.0–8.8%** (corrected +4.98/+8.79), not
  4.8–8.1. kalman vs per-basin ridge is significant at **5 of 6 leads — h4 ns (p=0.078)**:
  use an open marker at h4 on that contrast if drawn.
- **F2**: corrected crossing is **+20.0/−3.5 ns/−12.9/−17.4/−23.6/−30.3%**; matched sample
  is **227 basins × 60 months** (not 61). Per-basin win counts: read from the CURRENT
  `phase8b_li_comparison_perbasin.csv` (post-rerun) — do NOT hardcode 133/81/69/65/57/57.
- **F5**: headline curve is now the **2-seed ensemble** increment
  +0.91/+1.45/+1.33/+1.32/+1.42/+1.96% with CIs from `phase8b_h16_ensemble_headline.csv`
  (per-seed curves as light robustness marks). Placebos are now **seed-matched**
  (`{arm}_s{s}_rand{seed}` rows in the placebo monthly files); 20/20 in all 12 cells.
  Caption keeps the reused-draws caveat.
- **F8 (NEW, required — carries the anti-leakage claim, Sect. on stratification)**:
  `fig08_stratification.pdf`. Two panels from `results/phase8_stratification.csv`:
  (a) LINEAR ridge_corr_top1_era5 vs ridge_own_era5 skill by contamination tercile
  (low/mid/high), x = lead 1–6 — shows confinement to cont-high at h1–2, negative
  elsewhere; (b) STACKED lstmres_corr_top1_ens vs lstm_own_era5_ens for the same three
  terciles + the resolved×cont-low/mid stratum, x = lead 1–6, bootstrap CI ribbons —
  shows uniform positive effect everywhere. One shared y-axis (%), zero line. Claim in
  caption: "the linear effect carries a leakage signature (high-contamination, short
  lead); the stacked correction does not." Terciles n=78 each; resolved×low/mid n=142.

**BLOCKED until the control-phase reruns land (do not build from current CSVs):**
- **F3**: `phase5_perbasin_fdr_h1.csv` is PRE-AUDIT; the linear h1 effect is now ns
  overall (+0.308%, p=0.199) and the 16-FDR/9-helped/7-hurt roster must be recomputed
  from the corrected phase-5 rerun before this map is drawn.
- **F4**: panel (a) surrogates and panel (c) conditioning await the phase-4 reruns;
  NOTE: the F4 caption in main.tex now asserts the panel-(c) CIs cross zero — verified
  against phase6_era5_headline.csv (unconditioned [−0.088, +0.824], ERA5-conditioned
  [−0.041, +0.756]); keep the built panel consistent with that sentence.
  SURROGATES now FINAL (2026-08-15 rerun): h1 real beats 99/99 (p_rank=0.01); h2-6 real
  is beaten by surrogates (13/99, 1/99, 0/99, 0/99, 0/99) — panel (a) should show h1
  only, or the full lead profile as the honest version.
  panel (b) numbers changed (unrestricted +0.308 ns; use corrected `phase3b_summary.csv`,
  keep placebo counts from the corrected files). The +0.488/+0.502/+0.485 values above
  are pre-audit — banned.
- **F6**: awaits the `phase6_basin_analysis` rerun (its input file arrives with the
  phase5_coupled rerun); ρ values will shift.
- **F7**: linear per-basin reference must come from the corrected phase-5/3b rerun;
  Spearman +0.46/+0.51 was measured pre-audit — recompute.

Toolchain note: no LaTeX yet; figures are matplotlib → vector PDF into `figures/`
(`pip install cmcrameri` into `.venv` if missing; no cartopy needed for F1/F2/F5/F8).
