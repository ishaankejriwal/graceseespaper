# Li & Kusche (2026) - LLM-Optimized Research Core

**Paper:** Fupeng Li & Jürgen Kusche, *Observation-Driven Forecast of Global Terrestrial Water Storage and Evaluation for 2010-2024*, Water Resources Research 62, e2025WR041710. DOI: 10.1029/2025WR041710.

## How to use this file
This is a semantic compression for model context. It removes repeated motivation, publisher/layout boilerplate, and duplicated prose while retaining the paper's scientific setup, datasets, dates, resolutions, equations, forecast logic, evaluation design, numerical results, interpretations, limitations, caveats, applications, data links, and source-specific inconsistencies. Author-year citations are retained where especially useful; the companion full-cleaned file contains the complete wording and bibliography. Exact graphical shapes/data remain in the companion figures PDF.

## 1. Problem, motivation, and objective
- **TWS/TWSC:** Total water storage (TWS) integrates groundwater, surface water, soil moisture, snow, and glaciers. TWS is an Essential Climate Variable. TWSC is total water storage change.
- Traditional hydrological/land-surface models estimate total storage by summing modeled compartments. The paper argues these models can miss large-scale human effects such as irrigation, dams, and reservoir operations and can have process deficiencies (e.g., soil percolation and evapotranspiration), producing biases in storage amplitude and long-term trends.
- GRACE has measured monthly TWSC since 2002 and captures both natural and anthropogenic effects. Applications cited include water-budget closure, hydrological-model constraints/data assimilation, extreme-event identification, and sea-level budgets.
- Original GRACE ended July 2017; GRACE-FO began June 2018, leaving an 11-month mission gap. Standard GRACE-FO products also have roughly **2-3 months latency**, motivating near-real-time/forecast GRACE-like products.
- Prior ML work mainly reconstructed historical/pre-GRACE or gap-period TWSC. GRACE-derived TWSC is known to respond to climate variables such as precipitation and sea-surface temperature with lags. Reager et al. (2014) used such lag behavior for flood potential; Li et al. (2024) developed a more complex observation-driven TWSC forecast; Li et al. (2025) simplified it but tested only Europe.
- **Primary objective:** operationally usable, observation-driven ML forecasting of global GRACE-like TWSC, with a public hindcast/forecast product. Maximum lead = **12 months**.
- Public hindcast dataset: https://doi.pangaea.de/10.1594/PANGAEA.973113. Semi-operational **Global Land Water Storage Forecast Release 1 (GLWFC1.0)** is updated from 2024 onward: https://www.igg.uni-bonn.de/apmg/de/data-and-models/grace-fo-forecasting.
- Intended uses: drought/flood early warning, sea-level prediction, hydrological/land-surface forecast constraints and validation, GRACE latency bridging, Earth-orientation forecasting through hydrological angular momentum, and loading corrections for GNSS/altimetry.
- **Source-date inconsistency to preserve:** the Introduction says forecasts are initialized monthly from **December 2010 to April 2024**; Section 2.4.3 and Table 2 instead start the rolling hindcast at **December 2009**, forecasting 2010 first. Results/Table 2 use the Dec 2009 start.

## 2. Data

### 2.1 GRACE/GRACE-FO targets
Three monthly mascon products are used as separate targets:
- **JPL RL06.1 v03:** Apr 2002-Apr 2024; effective/claimed mascon resolution about 3 degrees; distributed on 0.5-degree grid.
- **CSR RL06.2:** Apr 2002-Apr 2024; about 3-degree mascons; distributed on 0.25-degree grid.
- **GSFC RL06 v2.0:** Apr 2002-Apr 2024; about 3-degree mascons; distributed on 0.5-degree grid.
- Units: equivalent water thickness, **cm**. Correlated striping errors are already intrinsically removed, so no extra destriping/smoothing is applied.
- Solutions differ slightly because of processing algorithms and glacial-isostatic-adjustment (GIA) corrections.
- All three are arithmetically grid-averaged/resampled to a common **1-degree grid** for forecasting. The paper still describes the underlying information/equal-area resolution as roughly **3 degrees**, even though outputs are provided on a 1-degree grid.

**GRACE gaps/missing months:**
1. The July 2017-May 2018 GRACE/GRACE-FO mission gap is filled with the global TWSC reconstruction of **Yin et al. (2023)**.
2. Other missing GRACE/FO months are interpolated by: estimate seasonal cycle from complete original data -> remove seasonal cycle -> linearly fit/interpolate the non-seasonal series (which also smooths abrupt gap changes) -> add the extracted seasonal cycle back. Goal: continuity while preserving realistic seasonality.

### 2.2 Candidate hydrometeorological predictors
Eight predictor **types** are used. Five ERA5 land variables, two MOHC ocean variables, and one climate-index category:
1. Precipitation (**P**) - ERA5
2. Land-surface temperature (**T**) - ERA5
3. Evapotranspiration (**E**) - ERA5
4. Runoff (**R**) - ERA5
5. Soil moisture (**SOM**) - ERA5
6. Sea-surface temperature (**SST**) - Met Office Hadley Centre (MOHC)
7. Sea-surface salinity (**SSS**) - MOHC
8. Climate indices (**CI**) - NOAA

**16 climate indices:** Pacific-North American Pattern (PNA), Western Pacific Pattern (WP), East Atlantic Pattern (EA), North Atlantic Oscillation (NAO), Southern Oscillation Index (SOI), Tropical Northern Atlantic (TNA), Tropical Southern Atlantic (TSA), Western Hemisphere Warm Pool (WHWP), Oceanic Nino Index (ONI), Multivariate ENSO Index (MEI), Nino 1+2, Nino 3, Nino 4, Arctic Oscillation (AO), Quasi-Biennial Oscillation (QBO), Atlantic Meridional Mode (AMM).

Predictor period: **Apr 2001-Apr 2024**. ERA5 native grid = **0.25 degrees**, aggregated to **1 degree** to match the forecast grid. MOHC SST/SSS = **1 degree**. Climate indices are non-spatial scalar series.

Rationale: after removing seasonal effects and linear terms, prior work found GRACE TWSC lagged these variables by roughly **1-12 months**, permitting forecasts using observational/reanalysis inputs rather than future climate-model forecasts.

### 2.3 Products used for comparison/benchmarking
- **GLDAS-NOAH:** Jan 2010-Apr 2024, 1 degree. Combines satellite/ground observations with land-surface modeling; used for global-mean TWSC comparison.
- **GLWS2.0:** Jan 2010-Dec 2019, 0.5 degree. GRACE/FO-assimilated WaterGAP Global Hydrological Model (WGHM); contains groundwater, soil moisture, snow, and other surface water.
- **ECMWF SEAS5:** seasonal forecast initialized Dec 2009-Mar 2024; native 0.25 degree; benchmarked up to **6 months** in this paper although SEAS5 produces lead times to **215 days**. HTESSEL land component represents vegetation, soil, snow, open water, with soil water in four layers: **0-7, 7-28, 28-100, 100-289 cm**. SEAS5 is described as a seamless coupled system whose atmospheric/land components are nearly identical to the operational ECMWF IFS and are coupled to ocean, sea-ice, and wave models.

## 3. Forecast method

### 3.1 Four-component decomposition
Both GRACE/FO TWSC and hydrometeorological predictors are decomposed using the Li et al. (2020) time-series approach into:
1. seasonal cycle
2. linear term/trend
3. interannual component
4. sub-seasonal component

The paper calls **interannual + sub-seasonal = non-seasonal**, but crucially the ML pipeline forecasts **interannual and sub-seasonal separately**, not as one combined target. The same forward-moving predictor-selection procedure is applied separately to the corresponding interannual and sub-seasonal forcing components.

**What is actually ML-forecasted:** interannual and sub-seasonal GRACE/FO components.

**What is extrapolated:** seasonal cycle and linear trend. The authors assume these change little from one year to the next and extrapolate them approximately from previous-year GRACE/FO behavior, then add them to the ML non-seasonal forecasts to reconstruct full TWSC.

**Reason/caveat:** linear trends can reflect glacier/ice melt and human actions such as dam/reservoir management and water withdrawal, which are difficult to predict from hydrometeorological inputs alone. Previous-year extrapolation can be biased where trends/seasonality are non-stationary or human policies change (example given: inconsistent farming policies).

### 3.2 Forward-moving (FM) predictors
For either the interannual or sub-seasonal component, every hydrometeorological predictor type is shifted forward by **m = 1,...,12 months**:

`X'_(m,n)(t) = X_n(t - m),  m = 1,...,12; n = 1,...,8`

where `X_n(t)` is the original non-seasonal hydrometeorological field/series for predictor type `n`, and `X'_(m,n)` is that predictor associated with a forecast `m` months ahead.

This creates **12 lead shifts x 8 predictor types = 96 candidate groups**. Each group is a 2-D dataset: global spatial candidate series versus shifted monthly time labels.

### 3.3 Correlation-based optimal predictor selection and spatial shifting
For each target GRACE/FO land grid cell, predictor type, and FM length, the method searches the candidate field and chooses the time series with the strongest correlation to that target cell's non-seasonal GRACE/FO series:

`x'_(m,n) = max[corr(y, X'_(m,n))]`

- `y`: non-seasonal GRACE/FO series at the target grid.
- `X'_(m,n)`: global/allowed candidate matrix for predictor type `n` and lead `m`.
- `x'_(m,n)`: selected 1-D predictor series.

Across all leads/types this produces **96 selected/sensitive predictors** per target grid (conceptually one best spatial series for each lead x variable type). For one specific lead `m`, the forecast uses **8 selected predictors**, one per predictor type. Example: the eight predictors selected from the `m=6` groups drive the **6-month lead** forecast.

**Spatial shifting:** predictor and target do **not** have to occupy the same location. The paper states that predictor grid entries within a certain neighborhood may forecast a target elsewhere: place B can predict TWSC at place A. The physical argument is that hydrological and anthropogenic transport cannot be represented as an isolated 1-D vertical column. Spatial shifting substantially raises predictor-target correlation, especially for larger FM lengths. The paper does not specify the neighborhood radius/geometry in the main text.

### 3.4 LSTM
- Model: **Long Short-Term Memory (LSTM)**, an RNN variant with memory cells intended to capture immediate and long-term sequential dependencies.
- Inputs: the lead-specific optimal FM predictors above.
- Target: non-seasonal GRACE/FO component (interannual and sub-seasonal handled separately).
- Forecast horizon: up to **12 months**.
- Predictor selection is repeated per target grid and lead. The paper therefore describes training the LSTM **12 times at each grid point**, one model/run for each lead time. The process is also applied separately to the two non-seasonal components.
- The main paper does **not** report a detailed LSTM architecture/hyperparameter table (e.g., hidden size, layers, optimizer) beyond using LSTM; do not invent these.

### 3.5 Rolling hindcast/computational flow
- Hindcast evaluation period: **Jan 2010-Apr 2024**.
- Training window in each iteration: **Jan 2003 through time `t`**.
- Initial `t`: **Dec 2009** in Section 2.4.3, so first model trains Jan 2003-Dec 2009 and forecasts 2010 up to 12 months ahead.
- Then `t <- t + 1 month`; optimal predictors are reidentified and training/forecasting repeats with the expanded training record.
- Maximum lead in every iteration = 12 months.
- Full TWSC = forecasted interannual + forecasted sub-seasonal + extrapolated seasonal + extrapolated linear.

### 3.6 Forecast uncertainty
For each 1-degree grid and each lead, non-seasonal forecast uncertainty is estimated from **LSTM training error**, not from future observations:

`epsilon = sqrt( sum_i (y_i - yhat_i)^2 / p )`

where `y_i` = non-seasonal GRACE/FO training observations, `yhat_i` = LSTM estimates during training, and `p` = training length. It is essentially training-period RMSE and is used as a reliability/confidence indicator for future forecasting.

## 4. Produced GRACE-FCast datasets
Three separate forecast products are created using each mascon target:
- **JPL-FCast** -> JPL target
- **CSR-FCast** -> CSR target
- **GSFC-FCast** -> GSFC target

Shared forcing: P, T, E, R, SOM, SST, SSS, CI. Initialized Dec 2009-Apr 2024 per Table 2; 12-month maximum lead; monthly; underlying/equal-area resolution described as about 3 degrees but **provided on 1-degree grid**. Coverage includes all global land, including Greenland and Antarctica.

Each dataset contains **cm equivalent-water-height TWSC** as:
- seasonal
- linear trend
- interannual
- sub-seasonal
- full TWSC (sum/reconstruction)

Interannual/sub-seasonal are LSTM forecasts; seasonal/linear are extrapolated. Long-term trend maps from GRACE-FCast and the three GRACE mascons are highly consistent; both show pronounced water-storage decline in **northwestern India**, known for groundwater depletion.

### 4.1 Uncertainties
Mean non-seasonal uncertainty averaged over all 12 leads/global land:
- JPL-FCast: **2.8 cm**
- CSR-FCast: **3.0 cm**
- GSFC-FCast: **2.8 cm**

Spatial-mean uncertainty rises from lead 1 to around lead 6, then falls at longer leads. The authors speculate this counterintuitive behavior is caused by spatial shifting and lead-specific source/location selection, which can find stronger long-lag predictors; SST in many regions is cited as having stronger >6-month correlations with TWSC than at shorter lags.

### 4.2 Global-mean series
Area weighting accounts for varying grid-cell area with latitude. Global land averages are supplied in **mm equivalent-water height**, optionally including/excluding Greenland and Antarctica and optionally including/excluding seasonal+linear terms. Combining **12 leads x 3 FCast products x 2 seasonal/linear choices x 2 Greenland/Antarctica choices = 144 global-mean time series**.

## 5. Evaluation against independent forecast-period GRACE/FO
For evaluation, the authors average the three FCast products and compare to the mean of JPL/CSR/GSFC mascons. GRACE/FO is used in each model's historical training portion but is **excluded from the corresponding forecast period**, so forecast-period observations act as independent validation for that hindcast.

Metrics at grid/basin/global scales:
- **CC (correlation coefficient):** timing/variability agreement.
- **RMSE:** absolute error magnitude.
- **MRE (mean relative error):** absolute error relative to TWSC amplitude; added to interpret low-variability regions where CC can look poor despite small absolute errors.

### 5.1 Full TWSC - grid scale
Evaluated Jan 2010-Apr 2024 at leads **1, 3, 6, 12 months**.

**Land fraction with CC > 0.8:**
- 1 mo: **72%**
- 3 mo: **69%**
- 6 mo: **65%**
- 12 mo: **64%**

**Land fraction with RMSE < 4 cm:**
- 1 mo: **65%**
- 3 mo: **63%**
- 6 mo: **59%**
- 12 mo: **58%**

Central Australia shows notably low CC without correspondingly large RMSE. The paper attributes this to small TWSC variability + GRACE measurement noise -> low signal-to-noise ratio, plus regional hydrological/model-performance challenges. MRE is high in dry regions such as the **eastern Sahara and Australian desert**, supporting the low-SNR interpretation.

### 5.2 Full TWSC - basin scale
Comparison: 26 major river basins; 1-12 month leads; Jan 2010-Mar 2024.
- Generally close agreement across most basins/leads.
- Especially good where seasonality/trends are stable: **Yukon, Ganges, Danube**.
- Worse where hydrological patterns/trends/seasonality vary: **Amur, Murray, St. Lawrence**.
- Mechanism: full forecast relies on extrapolating past seasonal and linear components; stable signals extrapolate well, non-stationary ones create bias.
- Key limitation: weak handling of **non-stationary seasonal and long-term dynamics**.

### 5.3 Full TWSC - global scale
Global mean FCast agrees strongly with GRACE/FO whether Greenland/Antarctica are included or excluded. Including glacier regions gives a steeper long-term decline, consistent with GRACE/FO, suggesting glacier-melt trends are stable enough to be captured by trend extrapolation. Correlation decreases from lead 1 through about **lead 7**, then stops decreasing. Authors attribute the plateau/slight recovery to correlation-based predictor selection plus spatial shifting, whose selected-predictor correlations can remain stable/increase at long leads.

## 6. Non-seasonal evaluation (interannual + sub-seasonal)

### 6.1 Grid scale
At leads 1, 3, 6, 12 months:

**Land fraction with CC > 0.8:** **20%, 17%, 13%, 21%**.

**Land fraction with RMSE < 4 cm:** **79%, 78%, 76%, 77%**.

Compared with full TWSC, non-seasonal forecasts have **much lower CC but lower RMSE**. Thus adding extrapolated seasonal + linear terms boosts correlation because those persistent components are easy to align, but also increases absolute error. Full FCast has much larger MRE than non-seasonal FCast over arid areas such as central Australia; authors warn that the seasonal/linear **remove-restore** procedure may perform poorly in arid regions and recommend caution there.

### 6.2 Basin scale and extremes
Across the 26 basins, the non-seasonal FCast often captures timing of abnormal/extreme variations. Examples:
- **Mackenzie:** peak around **2021** appears in both GRACE/FO and FCast.
- **Amazon:** trough around **2016** appears in both.

This suggests potential for drought/extreme-event forecasting. However, FCast systematically **underestimates extreme magnitudes**. Explanations proposed here and in the Discussion: GRACE/FO observation error, insufficient predictors, LSTM/model bias/error, rare extremes underrepresented in training, and missing drivers.

### 6.3 Global scale
Global non-seasonal variation is consistent with GRACE/FO but extreme signals are muted. Correlation decreases from lead 1 through about **lead 8**, then no longer decreases, paralleling the full-signal behavior.

## 7. Comparison with ECMWF and hydrological/land-surface simulations
No other GRACE-like total-water-storage forecast product is identified by the authors; ECMWF SEAS5 soil-water forecasts are therefore used as the main forecast benchmark.

### 7.1 Basin comparison: GRACE-FCast vs SEAS5
- Mean of three FCast products vs ECMWF, up to **6-month lead**.
- 26 basins; both **non-seasonal and full TWSC**.
- Reference = mean GRACE/FO mascons.
- Detailed basin time-series comparisons are in Supporting Information Figures S1-S6.
- Figure 10: correlation boxplots; Figure 11: RMSE boxplots; period stated in captions as **Dec 2010-Sep 2023**.
- Result: **GRACE-FCast fits GRACE/FO better than ECMWF for both non-seasonal and full TWSC**, with higher correlations and lower RMSE across the 26 basins/leads.

### 7.2 Global comparison: FCast, ECMWF, GLDAS-NOAH, WGHM/GLWS2.0
- Paper also evaluates global means against GRACE/FO for leads up to 6 months and compares FCast/ECMWF with GLDAS-NOAH and WGHM-GLWS2.0 simulations.
- Figure 12: global-mean **non-seasonal** TWSC, **excluding Greenland and Antarctica**, Jan 2010-Sep 2023; linear trends removed from every series for fair comparison.
- Figure 13 caption: global-mean **full** TWSC **including Greenland and Antarctica**, Jan 2010-Sep 2023; each series' Jan 2010-Dec 2019 mean removed.
- **Source inconsistency:** surrounding Section 3.3 text says the global evaluations in Figures 12 and 13 exclude Greenland and Antarctica, but Figure 13's caption explicitly says it includes them. Preserve this ambiguity rather than silently resolving it.
- Figure 14: CC and RMSE for non-seasonal/full **global-mean TWSC excluding Greenland and Antarctica**; FCast and ECMWF leads 1-6 plus GLDAS-NOAH and WGHM simulations; comparison to global-mean GRACE/FO; caption period **Jun 2010-Dec 2019**.

**Performance ordering/results:**
- GRACE-FCast consistently has **higher correlation and lower RMSE** than ECMWF at every examined lead and also outperforms GLDAS-NOAH and WGHM-GLWS2.0.
- Authors explicitly note this advantage is partly expected because GRACE/FO was used to train the ML relationship; the model has learned GRACE patterns that recur during forecast validation.
- For **non-seasonal global mean TWSC**, ECMWF beats GLDAS-NOAH but is worse than WGHM-GLWS2.0 through about **5-month lead**.
- For the **full global signal**, ECMWF is worse than both GLDAS-NOAH and WGHM-GLWS2.0 at all leads, implying SEAS5 is relatively better for non-seasonal variability than for GRACE-like seasonal/trend components.

## 8. Discussion, limitations, and interpretation
- Overall 1-degree-grid performance is described as robust: CC >0.8 and RMSE <4 cm over most global land for full TWSC, but dry/desert regions (central Australia, Sahara) are harder because the signal is small relative to noise. MRE exposes this relative-error problem and forecasts should be interpreted cautiously in dry regions.
- FCast outperforms SEAS5, GLDAS-NOAH, and WGHM-GLWS2.0 in the paper's comparisons. SEAS5 is reasonably capable on short-lead non-seasonal storage but struggles with seasonal cycles and long-term trends as a GRACE surrogate.
- Non-seasonal FCast can indicate timing of hydrological extremes but **underestimates their magnitude** and produces smoother TWSC than GRACE.
- Reasons the paper gives for smoothing/attenuated peaks:
  1. common ML losses such as MSE minimize average error and push predictions toward a conditional mean, underrepresenting rare high-magnitude events;
  2. extremes are under-sampled, giving weak gradient/training information;
  3. regularization such as dropout/weight decay and architectural smoothing features such as convolution/pooling can suppress abrupt variation (general ML explanation in Discussion, even though the main forecast model here is LSTM);
  4. sub-monthly hydrological processes and human activities may be absent from the predictor set;
  5. GRACE observational error, insufficient predictors, and systematic model bias/error also contribute.
- The authors present the method as suitable for an operational TWSC forecasting system and position GRACE-FCast/GLWFC1.0 as resources for drought/flood studies, hydrological forecast-model constraints, geodesy (Earth-orientation and loading), and bridging GRACE-FO product latency.

## 9. Figure index (exact visual data are in companion figures PDF)
1. **Fig. 1:** complete rolling workflow: initialize Jan 2003:t training -> remove seasonal/linear terms -> create FM predictors 1-12 months -> select optimal predictors -> train lead-specific LSTM -> forecast non-seasonal TWSC to t+12.
2. **Fig. 2:** global 2010-2024 TWSC trend maps for JPL/CSR/GSFC mascons vs their FCast products; FCast trends shown as average across 12 leads.
3. **Fig. 3:** non-seasonal uncertainty maps for the three FCast products plus global-mean uncertainty vs lead; average over 12 leads.
4. **Fig. 4:** full-TWSC global maps of CC, RMSE, MRE for leads 1, 3, 6, 12.
5. **Fig. 5:** full TWSC time series in 26 basins, GRACE vs leads 1-12, Jan 2010-Mar 2024.
6. **Fig. 6:** global-mean full TWSC, with and without Greenland/Antarctica, and CC by lead, Jan 2010-Mar 2024.
7. **Fig. 7:** non-seasonal global maps of CC, RMSE, MRE for leads 1, 3, 6, 12.
8. **Fig. 8:** non-seasonal time series in 26 basins, GRACE vs leads 1-12, Jan 2010-Mar 2024.
9. **Fig. 9:** global-mean non-seasonal TWSC, with and without Greenland/Antarctica, and CC by lead, Jan 2010-Mar 2024.
10. **Fig. 10:** basin CC boxplots comparing FCast vs ECMWF, non-seasonal/full, leads 1-6, 26 basins, Dec 2010-Sep 2023.
11. **Fig. 11:** analogous basin RMSE boxplots.
12. **Fig. 12:** global non-seasonal GRACE vs FCast/ECMWF/GLDAS/WGHM at leads 1-6; excludes Greenland/Antarctica; trends removed.
13. **Fig. 13:** global full GRACE vs FCast/ECMWF/GLDAS/WGHM at leads 1-6; caption says includes Greenland/Antarctica; Jan 2010-Dec 2019 means removed.
14. **Fig. 14:** global CC/RMSE summary for non-seasonal/full forecasts/simulations, excluding Greenland/Antarctica.

## 10. Data/access/end matter
- Hindcast data: https://doi.pangaea.de/10.1594/PANGAEA.973113
- Semi-operational GLWFC1.0: https://www.igg.uni-bonn.de/apmg/de/data-and-models/grace-fo-forecasting
- Conflict of interest: authors declare none relevant to the study.
- Funding: German Aerospace Agency DLR WIKI project (50EE2208); German Research Foundation DFG grants KU1207/26-1, KU1207/26-2, KU1207/39-1 (project 524616797), SFB1502/1-2022 (project 450058266); NSFC grants 42104085, 42574004, 42274115; open-access funding via Projekt DEAL.
