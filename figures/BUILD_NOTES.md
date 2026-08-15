# Figure build notes — 2026-08-15

Built by `scripts/make_figures.py` (run with `.venv\Scripts\python.exe scripts\make_figures.py`).
Reads only `results/*.csv`. Every headline value below is hard-asserted in the script
against `paper/REWRITE_LEDGER.md` (tolerance ±0.005 on 2-dp values); the script refuses
to build if any value drifts. F3/F4/F6/F7 were NOT built (inputs stale until the
control-phase reruns land, per FIGURE_PLAN.md addendum).

Conventions applied to all four: cmcrameri `batlow` for categorical/sequential curves,
`vik` extremes for the diverging F5 pair; solid/dashed/dotted/dashdot + distinct markers
(never color-only); skills in %; leads h = 1..6 on x; vector PDF + 150 dpi PNG preview;
PDF `CreationDate` stripped for byte-stable rebuilds.

---

## fig01_benchmark_ladder (12 cm, single panel)

**Sources**
- `results/paper_baseline_ladder.csv` — columns `model`, `horizon`, `skill_vs_damped`
  (fraction ×100), `damped_ref`. Curves drawn: `persistence`, `ridge_own_lags`
  (pooled ridge), `ridge_own_perbasin` (per-basin ridge), `kalman_ar1` (bold),
  `kalman_own_ridge` (= "ridge on filtered states", bold dashed). Climatology omitted
  (off-scale, −125.6% at h1 — say so in the caption).
- `results/paper_baseline_contrasts.csv` — columns `challenger`, `reference`, `horizon`,
  `skill`, `ci_lo`, `ci_hi` (fractions ×100). CI ribbon on kalman_ar1: row vs
  `damped_persistence_rho` at h1, rows vs `damped_persistence_reg` at h2–6, matching the
  ladder's `damped_ref` convention (asserted). Ribbon: [3.87,6.09] / [7.41,10.14] /
  [4.80,6.49] / [2.28,3.88] / [1.78,3.33] / [2.83,4.57].

**Numbers printed on the figure**: bracket annotation "+5.0–8.8%" at h1–2; zero-line
label "damped persistence (stronger variant)". No other numerals beyond axis ticks.

**Ledger checklist (plotted vs REWRITE_LEDGER.md §1)** — all asserted in script:
- kalman_ar1: plotted +4.98/+8.79/+5.62/+3.07/+2.55/+3.63 — ledger "+4.98 / +8.79 / +5.62 / +3.07 / +2.55 / +3.63" ✓
- ridge on filtered states: plotted +4.82/+7.87/+6.04/+5.36/+5.14/+5.16 — ledger "+4.82 / +7.87 / +6.04 / +5.36 / +5.14 / +5.16" ✓
- per-basin ridge: plotted +3.05/+5.94/+4.26/+2.05/+0.38/−0.29 — ledger identical ✓
- pooled ridge: plotted +1.17/+4.78/+5.08/+3.64/+3.17/+2.49 — ledger identical ✓
- persistence: plotted −20.75/−22.68/−21.39/−19.49/−16.04/−15.80 — ledger identical ✓
- bracket +5.0–8.8% — ledger "corrected +4.98/+8.79 … Annotation bracket is +5.0–8.8%" ✓

**Choice made**: the optional kalman-vs-per-basin-ridge contrast curve was NOT drawn
(the per-basin ridge ladder curve already shows the gap; adding the contrast doubled the
ink at h1–2). The plan's rule "open marker at h4 (ns, p=0.078)" therefore does not
apply; if the contrast is added later, contrasts CSV h4 has p=0.0783 (ledger:
"+1.04 (p=0.078, NOT SIGNIFICANT)").

---

## fig02_crossing (12 cm, single panel)

**Sources**
- `results/phase8b_li_comparison_headline.csv`, subset `all_matched` only. Columns
  `skill`, `ci_lo`, `ci_hi` are FRACTIONS — multiplied by 100 in the script. `dm_p`
  drives marker fill (filled p<0.05, open ns). Curves: `lstmres_corr_top1_ens` vs
  `li_lstm_full` (main, with CI ribbon), `kalman_ar1` vs `li_lstm_full` (thin dashed),
  `lstmres_corr_top1_ens` vs `li_lstm_nonseas` (thin dotted).
- `results/phase8b_li_comparison_perbasin.csv` — win counts for the caption (below).

**Numbers printed on the figure**: none besides ticks; "crossover" label on the dashed
vertical line between h2 and h3. Open markers: main h2 (p=0.50); kalman h2 (p=0.15);
nonseas h3 (p=0.55).

**Ledger checklist (REWRITE_LEDGER.md §6)** — asserted:
- main curve: plotted +20.01/−3.49/−12.94/−17.44/−23.64/−30.34 — ledger "+20.01 (p=3.4e-4) / −3.49 / −12.94 / −17.44 / −23.64 / −30.34" ✓
- matched sample: n_rows=13620, n_months=60 per lead — ledger "227 basins × 60 months = 13,620 rows per lead" ✓ (asserted)
- kalman h1: plotted +14.63 — ledger "h1 +14.63% (p=0.0039)" ✓
- nonseas h1–h2: plotted +30.94/+8.96 — ledger "+30.94% … +8.96%" ✓

**Per-basin win counts for the caption** (basins with dm_stat<0, of 227, read from the
CURRENT post-rerun perbasin file — the stale 133/81/69/65/57/57 was NOT used):
**137 / 82 / 71 / 65 / 59 / 59** at h1…h6.

---

## fig05_delivery (12 cm, single panel)

**Sources**
- `results/phase8b_h16_ensemble_headline.csv` (`skill_pct`, `ci_lo_pct`, `ci_hi_pct`,
  `dm_p`): main correction curve `lstmres_corr_top1_ens` vs `lstm_own_era5_ens` with CI
  ribbon; channel curve `lstm_corr_top1_era5_ens` vs `lstm_own_era5_ens` (ensemble row
  exists, so it is the main channel curve per spec) with CI error bars.
- `results/phase8b_h16_headline.csv`: per-seed light markers — correction
  `lstmres_corr_top1_s0/s1` vs `lstm_own_era5_s0/s1`; channel `lstm_corr_top1_era5_s0`
  vs `lstm_own_era5_s0` (no s1 channel row exists in that file).
- Placebo band: `results/phase8_lstm_combined_placebo_monthly.csv` (h1–3) +
  `results/phase8b_lstm_h46_placebo_monthly.csv` (h4–6), models
  `lstmres_corr_top1_s{0,1}_rand{0..19}` (`sum`,`count` per month), against reference
  MSEs computed from `results/phase8_lstm_combined_predictions.csv` +
  `results/phase8b_lstm_h46_predictions.csv` (models `lstm_own_era5_s{0,1}`).

**Placebo-band computation — FLAGGED FOR REVIEW (documented deviation from the literal
task wording).** The tasking said to pool to RMSE per horizon and take skill as an
RMSE ratio. Doing that puts the band on a scale ~½ of the plotted headline curve,
because the headline `skill_pct` in both h16 files is an **MSE-ratio** skill:
100·(1 − MSE_challenger/MSE_reference). Verified two ways in-script (asserted):
(i) recomputing the per-seed correction skill from the raw predictions with the
MSE-ratio formula reproduces the headline to <0.0015 pp at all 12 cells
(RMSE-ratio gives exactly half, e.g. s0 h1 0.320 vs headline 0.639);
(ii) the MSE-scale per-cell placebo means span −0.268…−0.050%, matching the ledger's
"Placebos cost only −0.27% to −0.05%" exactly (RMSE scale gives −0.13…−0.03).
So the band is computed as: per placebo draw and horizon, pool `sum`/`count` over all
months and folds of both files → MSE; increment = 100·(1 − MSE_placebo/MSE_own,seed);
shade min–max over the 40 draws (20 per seed, seed-matched references). Also asserted:
40 draws per horizon; 20/20 placebos beaten in all 12 lead×seed cells (ledger: "beats
20/20 … in all 12").
Band drawn (min…max, %): h1 −0.45…+0.21, h2 −0.33…+0.22, h3 −0.43…+0.16,
h4 −0.39…+0.10, h5 −0.54…+0.17, h6 −0.31…+0.13.

**Numbers printed on the figure**: none besides ticks (annotation "identical neighbor
information, two deliveries"; legend states "40 draws" and "2-seed ensemble").

**Ledger checklist (REWRITE_LEDGER.md §5)** — asserted:
- ensemble correction: plotted +0.91/+1.45/+1.33/+1.32/+1.42/+1.96 — ledger "+0.91 / +1.45 / +1.33 / +1.32 / +1.42 / +1.96" ✓
- its CI lower bounds: 0.64/1.07/1.02/0.98/1.11/1.67 — ledger "CI lower bounds +0.64/+1.07/+1.02/+0.98/+1.11/+1.67 (exclude zero everywhere)" ✓
- per-seed s0: +0.64/+1.33/+1.29/+1.26/+1.51/+1.90 — ledger "s0 +0.64/+1.33/+1.29/+1.26/+1.51/+1.90" ✓
- per-seed s1: +1.19/+1.56/+1.36/+1.37/+1.28/+1.93 — ledger "s1 +1.19/+1.56/+1.36/+1.37/+1.28/+1.93" ✓
- channel ensemble: +0.36/−0.42/−0.43/−0.61/−0.37/−0.38 — ledger "+0.36% (p=0.090) at h1; negative at h2–h6 (−0.42/−0.43/−0.61/−0.37/−0.38)" ✓

**LEDGER DISCREPANCY (not a figure value — needs a ledger fix).** Ledger §5 says
"All p ≤ 1.2e-10" for the ensemble correction. The CSV (sole source; cross-checked
against the `all` rows of `phase8_stratification.csv`, identical) has dm_p =
1.10e-10 / 2.30e-10 / 1.93e-09 / **3.14e-08 (h4)** / 6.17e-09 / 1.21e-13. Max is
3.1e-08, not 1.2e-10 — the ledger line looks like a transcription slip from the h1
value. Nothing plotted depends on it (dm_p is not printed on F5); the script asserts
all p < 5e-8 and this note flags the mismatch instead of silently weakening anything.

**Caption must keep the reused-draws caveat**: h4–6 placebo draws reuse the h1–3
per-fold graphs, so the six leads are not independent events.

---

## fig08_stratification (17 cm, two panels, shared y)

**Sources** — `results/phase8_stratification.csv` only (columns `challenger`,
`reference`, `stratum`, `horizon`, `n_basins`, `skill_pct`, `ci_lo_pct`, `ci_hi_pct`,
`dm_p`):
- Panel (a): `ridge_corr_top1_era5` vs `ridge_own_era5`, strata `cont_tercile_low/mid/high`.
- Panel (b): `lstmres_corr_top1_ens` vs `lstm_own_era5_ens`, same three terciles plus
  `resolved_x_cont_lowmid`.
Shared y-axis and zero line on both; tercile colors/styles identical across panels.
Marker fill = DM p<0.05 (all panel-(b) cells filled; panel (a) filled only at cont-high
h1–h2 — the visual form of the leakage-signature claim).

**CI presentation**: 95% CIs are drawn as error bars (with slight x-offsets) on every
curve in BOTH panels, including the required cont-low and resolved×low/mid strata in
panel (b). Ribbons were not used in (b): the four curves sit within ~0.7 pp of one
another, so overlapping ribbons were illegible — this uses the plan's explicit
declutter fallback ("markers+error bars are fine if ribbons overlap illegibly").

**Numbers printed on the figure**: legend n's — n=78 for each contamination tercile,
n=142 for resolved × cont. low/mid (asserted; ledger/addendum: "Terciles n=78 each;
resolved×low/mid n=142" ✓); "filled: DM p<0.05" key in panel (a).

**Ledger checklist (REWRITE_LEDGER.md §3)** — asserted:
- linear cont-high: plotted +1.11/+0.59/+0.37/+0.08/−0.11/+0.00 — ledger "cont-high h1 +1.11% p=2.3e-4, h2 +0.59% p=0.037, dead h3–h6" ✓ (CSV p: 2.28e-4, 0.0373 — asserted <3e-4 and <0.05)
- linear mid/low terciles negative at every lead — ledger "mid/low terciles negative at every lead" ✓ (asserted all 12 values < 0)
- stacked cont-low: plotted +0.69/+1.31/+1.29/+1.28/+1.31/+1.84 — ledger "cont-low tercile (n=78) +0.69/+1.31/+1.29/+1.28/+1.31/+1.84% (all p≤0.007)" ✓
- stacked resolved×low/mid: plotted +0.89/+1.57/+1.46/+1.38/+1.49/+2.00 — ledger "+0.89/+1.57/+1.46/+1.38/+1.49/+2.00% (all p<1e-4)" ✓
- stacked positive in all plotted stratum×lead cells — ledger "positive in all 60 stratum×lead cells" (the 24 drawn here asserted > 0) ✓
Values also drawn but not in the ledger (from the same CSV): stacked cont-mid
+1.09/+1.58/+1.44/+1.33/+1.61/+2.01, stacked cont-high +1.00/+1.48/+1.29/+1.35/+1.37/+2.03.

**Caption claim (per addendum)**: "the linear effect carries a leakage signature
(high-contamination, short lead); the stacked correction does not."
