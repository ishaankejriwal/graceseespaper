# Reference and scientific framing review, 18 September 2026

This review covers the attached Li et al. (2024) paper, the locally archived Li and Kusche (2026) text, and selected primary sources. It is not a systematic literature review or an independent reproduction of the experiments.

## Corrections supported by primary sources

- Distinguish grid sampling, native estimation geometry, and effective spatial resolution. CSR RL06 mascons use an equal-area geodesic grid approximately 1 degree at the equator, represented on a 0.25-degree grid. JPL uses approximately 3-degree mascons represented on a 0.5-degree grid. The original statement that both have native 3-degree mascons is incorrect. The CSR provider explicitly warns that the estimation cell size is not the effective resolution and urges caution for basins smaller than approximately 200,000 square kilometers. [CSR documentation](https://www2.csr.utexas.edu/grace/RL06_mascons.html); [JPL product documentation](https://podaac.jpl.nasa.gov/dataset/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4).
- CSR describes regularized estimation and explicitly states that no additional empirical smoothing or destriping is applied. Therefore, saying that both products already have correlated-error filtering applied is imprecise. Describe the regularized estimation, identify JPL's CRI filter separately, and state whether this study applied any further spatial filtering. The filter discussed in this manuscript is temporal.
- The JPL data product deserves its own citation. Added `wiese2023jpl` using the provider's exact recommended citation and DOI. Added `save2020rl06`, which CSR asks users to cite alongside Save et al. (2016), and `csr2026documentation` for current native-grid documentation. Preserve the actual downloaded CSR version and metadata discrepancy in the manuscript; the provider's current archive must not silently replace the analyzed version.
- Kankanige et al. (2026) use a univariate forward recursive Kalman filter for persistence-based gravity-coefficient prediction. Section 2 explicitly rules out backward smoothing. Replace the description of this study as a state-space smoother with a description of its gravity-coefficient filtering/prediction task. [Primary article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025EA004554).
- Zhu et al. (2019) explicitly investigate predictability from initial hydrological conditions versus climate forcing. Thus, the claim that Li et al. (2026, FLDAS) are the one study examining memory directly is too broad. Limit that sentence to the subseasonal/seasonal FLDAS result. [Primary article](https://www.nature.com/articles/s41467-019-09245-3).
- The supplied Li et al. (2024) PDF, Section 4.2, explicitly describes validation against observations in the absence of a comparable baseline. The original manuscript's source comment attributing a seasonal-long-term-mean benchmark to this article was not corroborated in the attached main paper; a supplement would be needed to establish that detail. A focused claim about the absence of an observation-aware persistence reference is preferable to a field-wide assertion that few studies use any benchmark.
- Nie et al. (2025) is still represented as a preprint by the inspected arXiv record. Added its arXiv DOI. Its HydroGlobe targets include simulated and data-assimilated storage, so the analogy to direct mascon observations needs the existing target caveat. [Author preprint](https://arxiv.org/abs/2510.10799).
- Added missing article identifiers to Li et al. (2024), Li and Kusche (2026), and Kankanige et al. (2026). Li (2024) authors, title, year, volume, issue, and DOI agree with the attached paper.

## Interpretation checks for the manuscript

Persistence can carry the latest valid observation through a missing month. A Kalman filter handles missing observations through a defined state transition and uncertainty update, but that does not mean that persistence has no previous observation during the mission gap. The complete-record model comparison also does not establish forecast performance inside the gap.

A crossing of forecast-error curves is consistent with changing contributions from initial conditions and forcing. It does not isolate those contributions when forecast systems also differ in predictor set, training data, preprocessing, and architecture. Keep this distinction in the abstract as well as the discussion.

The fitted observation-noise term can absorb fast hydrological variability and model misspecification. The noise-free ablation demonstrates the predictive contribution of that term, not an independently measured satellite-noise fraction. Similarly, JPL/CSR differences cannot by themselves establish that a particular processing choice caused the smaller filter gain.

Failure to reject equal predictive accuracy is not proof of equivalence. Prefer “no statistically detectable improvement” to “ties” where the inference rests on a nonsignificant test. A small significant gain should remain quantitatively described.

## Using the Li et al. (2024) sample

The most useful features are the order of presentation (target, data, method, validation, physical interpretation), explicit lead times and comparison periods, and figure captions that define the quantity and panels. The paper's discussion qualifies its drought interpretation and acknowledges GRACE's coarse spatial resolution. These features can guide this draft without copying sentences or importing its numerical results.

The sample also contains promotional wording and lengthy staged transitions. Those need not be copied to match its scientific register. Favor concrete subjects and measured claims: “the filter reduced mean squared error” rather than “the filter wins”; “uses predictors” rather than “reads forcing”; and “consistent with” when the mechanism was not isolated experimentally. Retain meaningful technical terminology and the author's supported numerical findings.

Several references printed in the sample contain obvious bibliographic inconsistencies, so it should not serve as an unquestioned metadata source. The current draft's Arsenault et al. (2020) entry, for example, correctly has BAMS volume 101; the sample prints 107. No change to that entry was needed.
