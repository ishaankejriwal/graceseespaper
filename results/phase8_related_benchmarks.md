# Phase 8: Related TWSA forecasting papers with published quantitative benchmarks

Survey date: 2026-08-13. Scope: papers that genuinely FORECAST GRACE/GRACE-FO TWSA/TWSC
(as opposed to reconstructing, downscaling, gap-filling, or assimilating), with published
RMSE / NSE / CC values. Our setting for comparison: basin-scale, deseasonalized monthly
TWSA, 1-6 month leads, 234 global basins, pooled RMSE in cm EWH; own Kalman AR(1)
baseline ~5 cm pooled basin RMSE at h1 (227 basins with valid data).

---

## 1. Forecasting papers inside Li & Kusche's own reference list

From `Li_2026_references.txt`, the references that are themselves TWSA
forecasting/prediction papers (everything else is reconstruction, DA, drought
monitoring, or context):

| Reference | What it did |
|---|---|
| Ahmed, Sultan, Elbayoumi & Tissot (2019), *Remote Sensing* 11(15):1769, doi:10.3390/rs11151769 | NARX neural nets forecasting GRACE TWSA over 10 African watersheds using rainfall, temperature, ET, NDVI. Basin scale. |
| Li, Kusche, Rietbroek et al. (2020), *WRR* 56(5):e2019WR026551, doi:10.1029/2019wr026551 | Data-driven (MLR/ANN/ARX) reconstruction 1992-2002 AND prediction of the GRACE/GRACE-FO gap (Jul 2017 - May 2018), 1-degree grid, 26 regions. Hybrid recon/predict. |
| Li, Kusche, Sneeuw et al. (2024), *GRL* 51(17):e2024GL109101, doi:10.1029/2024gl109101 | ML forecast of non-seasonal global land water storage up to 12 months ahead from antecedent hydro-meteorological conditions. Direct precursor of GRACE-FCast. |
| Li, Springer, Kusche, Gutknecht & Ewerdwalbesloh (2025), *WRR* 61(2):e2024WR037926, doi:10.1029/2024wr037926 | ML TWSA forecast up to 1 yr over Europe, then assimilated into CLM. Basin CC ~0.91 (Iberia), 0.92 (Danube), 0.94 (Volga). |
| Li, F. (2025), PANGAEA dataset, doi:10.1594/PANGAEA.973113 | The GRACE-FCast (CSR/JPL/GSFC-FCast) seasonal-to-annual gridded forecast dataset 2010-2024. |
| Reager, Thomas & Famiglietti (2014), *Nature Geoscience* 7:588-592, doi:10.1038/ngeo2203 | Not a TWSA forecast: uses observed GRACE TWSA as a predictor of flood potential months ahead. Cite as "GRACE persistence has lead-time value", not as a benchmark. |
| Getirana et al. (2020), *J. Hydrometeorology* 21(1):59-71, doi:10.1175/jhm-d-19-0096.1 | GRACE-DA-initialized seasonal groundwater forecasts over the US (NHyFAS lineage). Skill in percentile/anomaly-correlation space, no TWSA RMSE in cm. |
| Dobslaw & Dill (2018), *Adv. Space Res.* 61(4):1047-1054; Dill, Dobslaw & Thomas (2019), *J. Geodesy* 93:287-295 | 90-day LSDM hydrosphere angular-momentum forecasts for Earth-orientation prediction. Physically a TWS forecast but evaluated in EOP space, not TWSA cm. |
| Arsenault et al. (2020), *BAMS* 101(7):E1007-E1025, doi:10.1175/bams-d-18-0264.1 | NHyFAS system paper: operational NASA hydrological forecasts (soil moisture, groundwater percentiles) initialized with GRACE DA; no TWSA-in-cm forecast benchmarks. |
| Johnson et al. (2019), *GMD* 12:1087-1117, doi:10.5194/gmd-12-1087-2019 | SEAS5 system description. SEAS5 soil-water (4 layers, leads to 215 days) is the benchmark Li & Kusche compare against; the SEAS5 paper itself publishes no TWSA-vs-GRACE metrics. |

Confirmed NOT forecasts (do not use as forecast benchmarks, cite as reconstruction context):
- **Humphrey & Gudmundsson (2019)** GRACE-REC, *ESSD* 11:1153-1170, doi:10.5194/essd-11-1153-2019 — statistical **reconstruction** driven by *observed* past precipitation/temperature (1901-present, near-real-time updates). It has no forecast mode and no forecast-lead benchmarks; its skill numbers (vs sea-level budget, streamflow, GHMs) are reconstruction skill.
- **Yin et al. (2023)** GTWS-MLrec, *ESSD* 15:5597-5615, doi:10.5194/essd-15-5597-2023 — ML **reconstruction** 1940-present, 0.25 deg; validated against water budgets over 341 basins and 10,168 gauges. Used by Li for gap filling; not a forecast.
- Sun, Scanlon et al. (2021) *WRR* (AutoML recon); Sun, Long et al. (2020) *WRR* (60-basin recon); Gyawali et al. (2022) *Remote Sensing* (gap filling); Mo et al. (2022) *J. Hydrology* 604:127244, doi:10.1016/j.jhydrol.2021.127244 (BCNN gap filling of the 2017-2018 gap — interpolation with data on both sides, not forecasting).

Note on venue: Li & Kusche's GRACE-FCast paper is published as **"Observation-Driven
Forecast of Global Terrestrial Water Storage and Evaluation for 2010-2024", WRR (2026),
doi:10.1029/2025WR041710** (dataset at PANGAEA 973113). If our draft cites it as ESSD,
correct to WRR.

---

## 2. Genuine forecasting papers found by search (2019-2026), with metrics

### 2.1 Mo et al. (2025) — closest published numbers, and the headline risk
**Mo, S., Schumacher, M., van Dijk, A.I.J.M., Shi, X., Wu, J., Forootan, E. (2025).
"Near-Real-Time Monitoring of Global Terrestrial Water Storage Anomalies and Hydrological
Droughts." *Geophysical Research Letters* 52, e2024GL112677. doi:10.1029/2024GL112677**

- Task: fill the ~3-month GRACE/FO latency, i.e. true out-of-sample 1-3 month-ahead
  forecasts of global gridded TWSA with a Bayesian CNN (with uncertainty).
- Scale: global grid (mascon-resolution fields). Evaluation Jun 2019 - Feb 2024.
- Published metrics (per-grid-cell **medians**, 1/2/3-month leads):
  **RMSE = 1.79 / 2.07 / 2.26 cm; NSE = 0.89-0.81; CC = 0.95-0.92.**
- Target is **full TWSA (seasonal cycle included)**, after internal series
  stationarization; drought detection up to 3 months earlier than GRACE release.
- Comparability with our pooled deseasonalized basin RMSE: **not directly comparable**,
  for three stacked reasons: (a) median-over-grid-cells vs pooled (RMS-over-everything)
  — the median is pulled down by vast low-variance arid/temperate areas while pooled
  RMSE is dominated by high-variance tropical basins; (b) full-signal metrics — CC/NSE
  are inflated by the predictable seasonal cycle relative to deseasonalized skill;
  (c) 1-3 month latency filling vs 1-6 month forecasting. Still, this is the paper a
  reviewer is most likely to put next to ours.

### 2.2 Li, B. et al. (2026) — the deseasonalized 1-6 month S2S comparison
**Li, B., Hazra, A., McNally, A., Slinski, K., Shukla, S., Anderson, W.
(2026). "Skills in sub-seasonal to seasonal terrestrial water storage forecasting:
insights from the FEWS NET land data assimilation system." *HESS* 30, 1097.
doi:10.5194/hess-30-1097-2026** (NASA GSFC group; preprint egusphere-2025-4198)
<!-- author list corrected 2026-08-13 per CrossRef during paper audit; previous list (Getirana/Rodell/Loomis/...) was wrong -->

- FLDAS-Forecast: Noah-MP + Catchment LSM forced with downscaled NMME precipitation,
  1-6 month leads, 0.25 deg over Africa/Middle East, 18-yr hindcasts 2003-2020,
  evaluated against GRACE/FO **non-seasonal (deseasonalized) TWS anomalies** — the same
  target definition as ours.
- Published metrics: domain-averaged deseasonalized reanalysis skill vs GRACE/FO:
  CC 0.72 (CLSM) / 0.57 (Noah-MP), RMSE 1.04 / 1.16 cm; forecast skill mostly as ROC
  scores for tercile categories (>0.6 over >50% of domain at 1-6 month leads for CLSM);
  CC decline over leads 2-6 months: -27% (CLSM), -48% (Noah-MP). GRACE/FO lag-6
  autocorrelation >0.37 over >75% of the domain (persistence argument).
- **No per-basin or pooled basin RMSE in cm is published** (their ~1 cm RMSE is for a
  domain-averaged time series, where spatial averaging over the whole of Africa removes
  most variance — do not compare it to our per-basin pooled 5 cm).
- This is the most important paper for our framing: same target (deseasonalized), same
  leads (1-6 months), physically-based competitor, and it argues initial-condition
  persistence dominates S2S TWS skill — which is exactly what an AR(1)/Kalman baseline
  formalizes.

### 2.3 Li & Kusche (2026) — the target paper itself
**Li, F., Kusche, J. (2026). "Observation-Driven Forecast of Global Terrestrial Water
Storage and Evaluation for 2010-2024." *WRR* 62, e2025WR041710. doi:10.1029/2025WR041710**
- 1-degree global, leads 1-12 months, hindcast 2010-2024, benchmark = ECMWF SEAS5;
  basin-averaged TWSC evaluated over 26 river basins (non-seasonal and full).
- Grid-scale numbers circulating from this line of work (2024 GRL / 2026 WRR):
  spatial CC of forecast maps falling from ~0.72 (lead 1) to ~0.29 (lead 12), grid RMSE
  ~5.0 to 7.4 cm across leads. These are grid-scale, spatial-correlation metrics — not
  pooled basin time-series RMSE — and their 26-basin evaluation is CC/RMSE per basin,
  not pooled over 200+ basins.

### 2.4 Ahmed et al. (2019) — basin-scale African ANN forecasts
**Ahmed, M., Sultan, M., Elbayoumi, T., Tissot, P. (2019). *Remote Sensing* 11(15):1769.
doi:10.3390/rs11151769**
- NARX ANNs, 10 major African watersheds, monthly TWSA.
- Published metrics: NSE > 0.75 ("very good") for 60% of watersheds, NSE > 0.65 for 10%,
  NSE > 0.50 for 30%.
- Caveats: full TWSA including seasonal cycle (NSE inflated by seasonality); inputs
  include contemporaneous observed rainfall/ET/NDVI, so it is a hindcast conditioned on
  observed climate rather than a pure lead-time forecast; 10 basins only. Citable as
  prior basin-scale work; NSE values not comparable to deseasonalized NSE.

### 2.5 Li et al. (2025) — Europe, ML + DA
(full citation in table above) Basin-scale CC 0.91 / 0.92 / 0.94 (Iberian Peninsula,
Danube, Volga) for TWSA forecasts up to 12 months, subsequently assimilated into CLM.
Three basins, Europe only; correlations, no published basin RMSE in cm.

### 2.6 Regional deep-learning prediction papers (citable, not comparable)
- **Lu, et al. (2024). "The changes prediction on terrestrial water storage in typical
  regions of China based on neural networks and satellite gravity data." *Scientific
  Reports* 14, doi:10.1038/s41598-024-67611-8.** BP / LSTM / BiLSTM-attention over six
  Chinese regions; test period Sep 2019 - May 2023 (45 months); best regions (Upper
  Yangtze, Southwest) RMSE 2.07-2.64 cm, R2 >= 0.8, NSE > 0.6. Full TWSA, regional,
  recursive multi-step with observed drivers — not comparable to pooled global
  deseasonalized RMSE.
- **Ahi, G., Cekim, H.O. (2021). "Long-term temporal prediction of terrestrial water
  storage changes over global basins using GRACE and limited GRACE-FO data." *Acta
  Geodaetica et Geophysica* 56:321-344. doi:10.1007/s40328-021-00338-4.** Purely
  statistical (exponential smoothing family) per-basin time-series prediction over
  global basins. Closest in spirit to a statistical basin baseline; full TWSA with
  seasonal cycle, so its per-basin RMSEs are seasonal-cycle-dominated.
- Saudi Arabia multi-step ML (Water 2024, 16(2):246, doi:10.3390/w16020246), Sudan
  probabilistic groundwater forecasting (Remote Sensing 2025, 17(18):3172,
  doi:10.3390/rs17183172): single-region, mostly groundwater; cite only if doing an
  exhaustive review.

### 2.7 Zhu, Yuan & Wood (2019) — basin-scale predictability benchmark (decadal)
**Zhu, E., Yuan, X., Wood, A.W. (2019). "Benchmark decadal forecast skill for
terrestrial water storage estimated by an elasticity framework." *Nature
Communications* 10:1237. doi:10.1038/s41467-019-09245-3**
- 32 global major river basins, 4-year-averaged TWS, leads 1-4 to 7-10 years.
- Benchmark NSE 0.51 (1-4 yr) declining to 0.11 (7-10 yr); initial conditions dominate
  skill at short leads.
- Decadal averages, so numerically incomparable to monthly 1-6 month skill, but a strong
  citation for (i) basin-scale TWS predictability framing and (ii) the
  initial-condition-persistence argument behind an AR(1) baseline.

---

## 3. Specific checks requested

- **(a) Humphrey & Gudmundsson 2019 GRACE-REC**: reconstruction, not forecast. Driven by
  observed (historical + near-real-time) precipitation/temperature. No forecast-mode
  benchmarks exist. Its ~monthly-updated near-real-time extension makes it a plausible
  *nowcast* competitor but its published skill is reconstruction skill.
- **(b) Operational NASA/JPL/GSFC forecast products**: no operational product forecasts
  TWSA itself in cm besides Li's CSR/JPL/GSFC-FCast. NASA's operational offerings are
  (i) the weekly GRACE-DA drought indicators (nowcast percentiles, drought.gov /
  nasagrace.unl.edu; Li, B. et al. 2019 WRR) and (ii) NHyFAS (Arsenault et al. 2020
  BAMS; Getirana et al. 2020 JHM), which forecasts soil moisture / groundwater
  *percentiles* 1-5 months out, evaluated with anomaly correlation and categorical
  skill — no TWSA RMSE in cm EWH. The new FLDAS-Forecast S2S evaluation is Li, B.
  et al. 2026 HESS (section 2.2).
- **(c) Yin et al. 2023 (GTWS-MLrec)**: reconstruction (1940-present), used by Li &
  Kusche for gap filling. Not a forecast.
- **(d) SEAS5 TWS benchmarks**: SEAS5 (Johnson et al. 2019 GMD) publishes no
  TWSA-vs-GRACE skill itself; it carries soil water in 4 layers to lead 215 days. The
  only published SEAS5-TWS-vs-GRACE benchmark we found is inside Li & Kusche 2026
  itself (GRACE-FCast beats SEAS5 on non-seasonal and full TWSC over 26 basins). The
  NMME-forced FLDAS work (2.2) is the independent S2S physical benchmark.

---

## 4. Honest answer: are there published numbers a ~5 cm pooled h1 basin RMSE would not beat?

**Direct, like-for-like comparisons do not exist.** No paper we found publishes pooled
RMSE over 200+ global basins for deseasonalized monthly TWSA at 1-6 month leads. The
closest-looking published numbers, and whether they "beat" us on their face:

1. **Mo et al. 2025: median grid RMSE 1.79 cm at 1 month (full TWSA).** On raw numbers
   this looks far better than ~5 cm and we should NOT claim to beat it. But it is a
   *median over grid cells* of *full-signal* errors over a *2019-2024* window — three
   choices that each shrink the number relative to our pooled deseasonalized basin
   metric (pooled RMSE is variance-weighted toward big wet basins; medians are not;
   sub-basin grid cells are noisier individually but most of the globe is low-variance).
   The defensible statement: "metrics are defined on different targets and aggregations
   and are not directly comparable; on matched definitions an evaluation would be needed."
   If we can compute *median per-basin RMSE* and/or full-TWSA metrics as a secondary
   table, we can bridge to their numbers instead of hand-waving.
2. **Li, B. et al. 2026 (FLDAS): domain-averaged deseasonalized RMSE ~1.0-1.2 cm.**
   Looks smaller than 5 cm but is an Africa-wide *spatially averaged time series* —
   averaging ~20 million km2 removes most variance. Not a fair comparison in either
   direction; their basin-level results are correlations/ROC only.
3. **Lu et al. 2024: 2.1-2.6 cm regional RMSE (China, full TWSA).** Region-specific,
   observed-driver hindcasts; not a global pooled metric.
4. **Ahmed et al. 2019: NSE > 0.75 in 60% of basins (full TWSA).** Deseasonalized NSE
   is structurally much lower than full-signal NSE (the seasonal cycle is the easy
   part); our deseasonalized NSE numbers will look worse than these and should never be
   put in the same table without saying so.
5. **Zhu et al. 2019: NSE 0.51 at 1-4 yr (4-yr means, 32 basins).** Different timescale
   entirely.

Bottom line for the paper: cite Mo et al. 2025, Li, B. et al. 2026 (HESS), Li et al.
2024 GRL, Li et al. 2025 WRR, Ahmed et al. 2019, Zhu et al. 2019, Ahi & Cekim 2021 as
the forecasting literature; state explicitly that published metrics differ in target
(full vs deseasonalized), aggregation (median-grid vs pooled-basin vs domain-average),
domain, and period, so we compare against baselines recomputed under our own protocol
(persistence, climatology, Kalman AR(1)) rather than against transplanted numbers. The
one number we should proactively defuse is Mo et al.'s 1.79 cm.
