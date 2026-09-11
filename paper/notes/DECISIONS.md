# DECISIONS.md — judgment calls, provenance, and open items

Manuscript drafted 2026-08-13. Files: `paper/main.tex`, `paper/references.bib`,
`paper/notes/FIGURE_PLAN.md`, plus Copernicus class files (`copernicus.cls`, `.bst`,
`.cfg`) copied from the official package (downloaded from
publications.copernicus.org on 2026-08-13; documentclass `[hess, manuscript]`).

---

## 1. Numbers: recomputations, corrections to briefed values

1. **Matched-row baseline ladder recomputed** (new script
   `scripts/build_paper_ladder.py`, outputs `results/paper_baseline_ladder.csv`
   and `results/paper_baseline_contrasts.csv`). Reason:
   `phase2_baseline_predictions.csv` row counts shrink with lead (19,656 at h1
   → 18,486 at h6) while `phase3b_predictions.csv` holds 19,656 at every lead,
   so summary-file RMSEs are not cross-comparable at h≥2. All Table 1 numbers
   and the Kalman contrasts come from the matched intersection.
2. **"Beats damped persistence by 5–6% at h1–2" (task brief / memory) →
   computed +4.8% (h1) and +8.1% (h2)** on matched rows, against the stronger
   damped variant per lead (rho-damping at h1, regression-damping at h2+).
   The manuscript quotes the computed values.
3. **"Beats per-basin ridge at all leads (DM p≤.04)" → computed max p = 0.0412
   (h4)**; manuscript states "p ≤ 0.042". Per-lead: 4.8e-4 / 4.5e-7 / .019 /
   .041 / .0062 / 1.8e-5.
4. **"2-parameter Kalman filter" (task brief) → "three-parameter"** in the
   manuscript. `src/gracefc/kalman.py` fits (ρ, q, r) by MLE; the *point
   forecast* depends only on ρ and the ratio q/r, which is stated in Methods.
   (Correction 2026-08-13: this entry originally claimed the project context
   file already said "three-parameter" — it said "2-parameter" until the
   post-audit rewrite of `docs/STUDY_CONTEXT.md` fixed it.)
5. **LSTM-ERA5 gain quoted as the 2-seed ensemble (+1.95/+1.99/+1.43%)**, not
   the seed-0 values (+2.05/+2.24/+1.60%) that the task brief quoted. This is
   the audited phrasing constraint in `results/phase7_analysis.md` ("report as
   ensemble or seed range, never the s0 numbers alone"). Seed-0 values appear
   only as the disclosed seed-spread example.
6. **Crossing h3 = −12.4%**, not the −12.5% in memory/brief:
   `phase8b_li_comparison_headline.csv` has −0.12446 → −12.4% at 1 decimal.
7. **Kalman-vs-damped CI at h1** (+3.7..+5.8) etc. available in
   `paper_baseline_contrasts.csv`; only the point estimates are in Table 1 to
   keep it readable, CIs go into Fig. F1.
8. **Lead-1 neighbor vs the bare filter disclosed**: +0.31%, p = .15
   (`phase5_headline_table.csv`, kalman_corr_top1 vs kalman_ar1). The
   registered +0.49% headline is the capacity-matched ridge-twin contrast;
   both are now in the text. This follows the plan-file audit constraint
   ("h1 is the only clean headline") and pre-empts the obvious referee catch.
9. **Arctic domination of the pooled-cm Li comparison stated qualitatively.**
   The specific shares in project memory (4 glaciated basins = 68% of Li h3
   squared error; Svalbard 36%) exist only in the memory file, not in any
   results CSV, so they are not quoted. The pooled-vs-median contrast from
   `phase8_li_cm_comparison.csv` (pooled 5.0 vs 7.6 cm but median-basin 2.4 vs
   2.7 at h1, reversing at h2–3) carries the same point with citable numbers.
10. **Hybrid splice numbers quoted row-pooled** (+10.4 / +21.6 / +22.3–23.7%)
    as in the RUN_LOG; the RUN_LOG notes equal-weight-month pooling shifts
    levels ~1 pt with rankings unchanged — not repeated in the text since the
    splice is framed as descriptive only, with the post-hoc caveat mandatory.
11. **Stack ensemble contrasts at h4–6 left blank in Table 4** (per-seed only):
    the ensemble-vs-ensemble contrast was not emitted for h4–6 in
    `phase8b_h16_headline.csv`, and I did not synthesize one. Both seeds are
    individually significant (max p = 3.6e-5).
12. **45/45 positive fold-cells claimed for h1–3 only** (that is what
    phase8_analysis verified); h4–6 positivity is claimed at the
    per-seed-significance level, not per fold-cell.

## 2. Data-description choices (stub vs memory conflicts)

13. **CSR release string**: old stub said "RL06.1", project memory says
    "RL0603". Wrote **RL06.3** with an explicit `% TODO` to confirm against
    file metadata before submission. (High-confidence reading of "RL0603",
    but I could not open the raw file to check.)
14. **"257 monthly solutions"** (old stub) not repeated — unverifiable from
    the results files; the record span (April 2002–May 2026) is stated
    without a count.
15. **"Basins smaller than 90,000 km² excluded"** (old stub) not repeated —
    the current sample definition in memory is 284 − 46 ice-sheet − 4 water
    bodies = 234, which is what the manuscript states.
16. **Glaciated basin count = 8** (per phase5/phase6 analyses), not the old
    stub's 9.
17. **ERA5 variable list**: 11 variables per RUN_LOG (tp, e, ro, sro, ssro,
    swvl1–4, t2m, sd). The old stub's list included snowfall (12) — not
    repeated.

## 3. Citations

18. **All bib entries fetched programmatically** via DOI content negotiation
    or the arXiv bibtex endpoint on 2026-08-13 (see header of
    `references.bib`). No entry written from memory.
19. **RESOLVED 2026-08-17**: the 2025 *Journal of Hydrology* paper (PII
    S002216942500890X) is Zhang, Shen, Ji, Wang & Chen (2025), "An integrated
    postprocessing approach for extracting time variable signals from GRACE
    monthly gravity field models", J. Hydrol. 661, 133552, DOI
    10.1016/j.jhydrol.2025.133552 — identified via the DOI redirect to the PII
    and cross-checked against Semantic Scholar (author order confirmed; note
    first author is Zhang, not Chen). It iteratively separates a stochastic
    non-seasonal signal from noise in the existing record (22 basins) — signal
    extraction, not forecasting. Cited in Sect. 5.1 as the closest upstream
    machinery; the paper's benchmark novelty claim is scoped accordingly. The
    full text remains paywalled and uninspected; the characterization rests on
    the abstract-level record.
20. **Niraula & Goessling (2021), not "Niraula & Notz"**: the project memory
    referred to this JGR Oceans paper as Niraula & Notz; CrossRef metadata
    (DOI 10.1029/2021JC017784) gives authors Niraula, Bimochan and Goessling,
    Helge F. The manuscript and bib use the fetched authors. The novelty
    audit's to-do — read the full text before leaning on the framing — still
    stands; the title ("Spatial Damped Anomaly Persistence of the Sea Ice
    Edge as a Benchmark...") supports the benchmark-hygiene framing used.
21. **Steidl & Zhu (2025) workshop entry constructed manually** (no DOI):
    verified against the NeurIPS ML4PS 2025 accepted-papers listing
    (ml4physicalsciences.github.io/2025/, fetched: "Hierarchical Graph
    Networks for Forecasting Terrestrial Water Storage — Steidl, Xiao Xiang
    Zhu") and the companion EGU26 abstract (DOI 10.5194/egusphere-egu26-18659,
    fetched via content negotiation, authors Steidl, Viola and Zhu, Xiao
    Xiang). Kept as `@inproceedings` with the workshop URL.
22. **Kankanige et al. (2026) framing**: cited as coefficient-level
    persistence support. Novelty audit note — full text was paywalled to
    agents; characterization rests on the abstract. Flagged here per the
    audit's instruction; verify before final submission.
23. **GLWFC1.0 (Bonn semi-operational forecast, Jan 2024+)**: mentioned in
    the novelty audit as "mention, too short to hindcast". Omitted from the
    manuscript — no citable archival record was found, and the FLDAS
    paragraph already covers the "other public forecast products"
    obligation. Authors may add a URL mention in revision if desired.
24. **CSR mascon dataset DOI** (10.15781/cgq9-nh24) does not resolve at
    doi.org; Data availability cites the CSR portal URL instead, and the
    methods cite Save et al. (2016) for the mascon approach.
25. **Mo et al. (2025)**: the Discussion states only their *code* (not
    predicted fields) is archived. Source: novelty-audit verification pass
    (zenodo checked). If challenged, re-verify the zenodo record.

## 4. Framing decisions

26. **Contribution order follows the task brief** (benchmark → crossing →
    neighbor), not the novelty audit's "crossing-first" recommendation. The
    title leads with the noise-filtering claim as directed. The
    benchmark-critique section is written constructively ("what does it take
    to beat this benchmark") per the brief, with the Nie et al. (2025),
    Niraula & Goessling (2021), Kankanige (2026), Ahi & Cekim (2021)
    positioning stated in both Intro and Discussion so the novelty perimeter
    is defended twice.
27. **The Kalman benchmark is explicitly delimited from data assimilation**
    (Springer et al. 2026 review cited at both mentions), per the audit.
28. **Neighbor novelty wording** is exactly the audited retreat position:
    "no prior work *isolates* the incremental cross-basin information on
    *observed* GRACE nor *validates* it against nulls" — not "first to use
    spatial information". Steidl & Zhu cited proactively with their
    reconstructed-target/no-null limitations stated factually.
29. **Mechanism language**: "regional (300–1000 km) shared hydroclimatic
    signal / spatial denoising", with an explicit sentence disclaiming
    hydrological-transfer claims.
30. **Delivery finding** connected to gradient starvation (Pezeshki et al.
    2021) as a *consistent mechanism*, not proven cause ("The pattern
    matches gradient starvation...").
31. **British vs American English**: American throughout (matches the
    analysis files); HESS accepts either if consistent. The short summary
    uses "neighboring".

## 5. Skill guidance (ai-research-skills / 20-ml-paper-writing) — adopted vs adapted

Clone succeeded; `ml-paper-writing/SKILL.md` and its references were read.

- **Adopted**: never write BibTeX from memory (all entries fetched;
  unverifiable one marked and excluded); one-sentence contribution
  discipline; claim-first results paragraphs ("open with the claim, then the
  evidence"); quantitative abstract following the 5-sentence arc
  (what/why/how/evidence/number); self-contained figure and table captions;
  explicit limitations section that pre-empts referee findings; consistent
  terminology (lead/skill/filtered state); no hype vocabulary
  (novel/remarkable absent); statistical-significance reporting with test,
  correction, and CI methodology stated.
- **Adapted for journal**: no page limit → full Data/Methods/Results/
  Discussion/Conclusions structure instead of 8-page compression; HESS is not
  double-blind → author placeholders rather than anonymization; conference
  checklists and templates not applicable → Copernicus class + journal
  sections (code/data availability, author contribution, competing
  interests); "Figure 1 first" workflow not applicable (figures deferred to
  FIGURE_PLAN.md per task instructions, but each figure's claim-mapping table
  is included in the plan).
- **Rejected**: Exa MCP / Semantic Scholar tooling suggestions (plain DOI
  content negotiation sufficed); the skill's acknowledgment-citation
  suggestion for itself (not appropriate for a journal manuscript).

## 6. Build / verification status

32. **The manuscript has not been compiled** — no LaTeX toolchain on this
    machine (`pdflatex`/`latexmk`/`tectonic` absent). Performed static checks
    instead (script in session scratchpad `check_tex.py`): all \cite keys
    resolve against references.bib (0 missing, 0 unused), environments
    balanced, all \ref labels defined, brace-balanced. First compile should
    be done on Overleaf or a TeX Live install; expected friction points:
    none known, the class file is the official one and no exotic packages
    are used (`\tophline`/`\middlehline`/`\bottomhline` table rules,
    `\unit{}` unused).
33. **Short summary is 497 characters** (limit 500), verified
    programmatically; stored as a comment block above the abstract (it is
    entered in the submission form, not typeset).
34. **Figure files do not exist yet**; every `\includegraphics` is commented
    out with a pointer to FIGURE_PLAN.md, so the paper compiles without
    them (captions render as placeholder figures).
35. **Number sourcing convention**: every quantitative claim in main.tex has
    a `% source: <file>, <detail>` comment on the same or an adjacent
    preceding line; tables carry per-row or per-block source comments;
    figure captions with numbers carry a source comment inside the figure
    environment.

## 7. Open TODOs (marked in main.tex)

- Confirm CSR release string (RL06.3 vs distributed metadata). [Sect. 2.1]
- Verify + add the J. Hydrology 2025 postprocessing citation. [Sect. 5.1]
- Author list, affiliations, contributions, acknowledgements/funding.
- Zenodo archive DOI for code/data availability.
- Generate figures per FIGURE_PLAN.md and un-comment the includegraphics
  lines; then first full compile.
- Pre-submission: full-text checks of Niraula & Goessling (2021) and
  Kankanige et al. (2026) (novelty-audit instruction).
- Consider porting the ≥300 km + conditioning controls to the stacked arm
  before submission (limitations item 2 would then be deletable) — ranked
  next step 3 in results/phase8_analysis.md.

---

## 8. 2026-09-10 — the Kalman-benchmark reframe

A decision record for the change of spine. The manuscript has not yet been
rewritten to match; these are the decisions the rewrite has to implement.

36. **The neighbour claim is dropped as a paper claim.** On CSR a neighbour's
    propagated state, used as a correction, is worth +0.31% at lead 1 over the
    own-basin ridge and beats 50/50 seed-matched placebo graphs and 99/99 IAAFT
    surrogates (`results/phase3b_summary.csv`,
    `results/phase4_surrogate_summary.csv`). On JPL none of it replicates: every
    neighbour variant is worse than own-basin at every lead; the placebo count is
    0/50 for the correlation-selected neighbour at every lead and for every arm at
    leads 1 to 3, while the distance-selected arm beats at most 26/50, at lead 5
    (`kalman_geo_top1` 8/50, 26/50 and 9/50 at leads 4, 5 and 6, `kalman_geo_top2`
    1/50 at lead 5); surrogates are 0/99 at every lead
    (`results/jpl/phase3b_summary.csv`,
    `results/jpl/phase4_surrogate_summary.csv`). The per-lead table is in the
    `results/RUN_LOG.md` entry "JPL neighbor experiments: non-replication
    record". A controlled effect that reverses on a second mascon product of the
    same observations is not a finding we are willing to publish. Working interpretation, stated as such:
    JPL's 3-degree mascons with the CRI filter already perform the spatial
    denoising a CSR neighbour supplied. The code is retained as extended chain
    steps, not deleted, so the experiment stays reproducible.
37. **The new spine is three claims.** (1) The per-basin Kalman AR(1) plus
    observation-noise filter is proposed as the reference forecast for
    deseasonalized basin TWSA, because it handles observation noise and the
    2017-18 mission gap natively and persistence does neither: +4.98/+8.79% at
    leads 1-2 against the stronger damped variant
    (`results/paper_baseline_ladder.csv`). (2) The strongest own-basin model at
    leads 1 to 4 is the Kalman filter plus a ridge correction over a flat 12-month
    history of filtered state and ERA5, `ridge_own_era5_flat12`: +12.24% over
    damped persistence at lead 1 and +7.65% over the filter itself (same file plus
    `results/paper_baseline_contrasts.csv`). At leads 5 and 6 the own-state ridge
    correction `kalman_own_ridge` is marginally better, and the rewrite has to say
    so: `ridge_own_era5_flat12` scores -0.58% against it at lead 5 and -0.73% at
    lead 6 (`results/flat12_ridge_summary.csv`, `skill_vs_ridge_own`), which is
    RMSE 1.3804 against 1.3764 and 1.4421 against 1.4369
    (`results/paper_baseline_ladder.csv`). Every sequence model trained here loses
    to it once history length is equalized
    (`results/phase7_lstm_summary.csv`,
    `results/flat12_train85_sensitivity.csv`). (3) Both are compared with Li and
    Kusche's GRACE-FCast on CSR mascons, and the Kalman reference alone on JPL
    so far (the JPL run predates the flat-12 step), under one protocol and one
    strict subset rule.
38. **One strict subset rule, applied to both mascon products.**
    `joint_full_cells`: a basin must fully contain at least one native mascon of
    the product being scored and at least one valid 1-degree Li cell. 209 of 227
    basins on CSR, 67 on JPL. Previously the rule was gated on
    `source() == "jpl"`, so CSR was scored on partially covered basins and JPL
    was not; the two products are now comparable. Headline on
    `joint_full_cells`: `kalman_ar1` beats Li's full product by +16.4% at lead 1
    (p = 2.2e-3), ties at 2, loses from 3 on;
    `ridge_own_era5_flat12` beats Li's non-seasonal product by +32.8% and +11.2%
    at leads 1-2, ties at 3, loses from 4 on
    (`results/phase6_li_comparison_headline.csv`).
39. **The JPL `all_matched` inflation is documented, not fixed.** Li's
    `rmse_std` reads 5.47 at lead 1 on `all_matched` against 0.93 on
    `joint_full_cells`. Three leak checks came back negative (no sentinel fills,
    ocean and non-finite cells cannot enter a basin mean, the full and
    non-seasonal variants share one validity mask). The cause is the fold
    standardization by our own product's residual std on partially covered,
    low-variance basins, and the remedy is the sample restriction. Reason for no
    code change: there is no defect in the aggregation to repair. The manuscript
    must not quote the JPL `all_matched` Li rows as a statement about Li's
    accuracy. Full diagnosis in the 2026-09-10 RUN_LOG entry.
40. **The mission-split filter is reported as a tested-and-rejected variant,
    not adopted.** A separate GRACE-FO observation-noise variance loses at every
    lead, -1.84% at lead 1 (p = 1.7e-7) through -0.38% at lead 6, with 0 basins
    helped and 0 hurt under BH correction at lead 1 and 234 of 1170 basin-folds
    falling back to one variance because fold 1 has fewer than 12 GRACE-FO
    training months (`results/kalman_mission_summary.csv`). The one-variance
    filter is kept. Reason to report it at all: the obvious referee question is
    whether the headline depends on treating two missions as one instrument.
41. **Nothing in the August audit narrative survives into the current docs.**
    `README.md`, `docs/CODE_MAP.md`, `docs/STUDY_CONTEXT.md` and
    `results/README.md` were rewritten on 2026-09-10 to state what is true now,
    with a source file cited beside each number. The history of how the numbers
    moved stays in `results/RUN_LOG.md`, which remains append-only.
42. **Open at the time of this record.** `paper/main.tex` still carries the
    three-finding structure and pre-reframe figures; `docs/ARCHIVE_MANIFEST.md`
    is flagged pre-reframe and will be regenerated after the rewrite; the
    `figures` chain step still reads extended outputs, so a default-only machine
    stops there; and `run_phase8b_merge.py` and `run_phase6_hybrid.py` still
    gate `joint_full_cells` on `source() == "jpl"`.
43. **Two corrections from the external audit (2026-09-11).** (a) The two-product
    claim applies to the Kalman reference only until the JPL rerun: the versioned JPL
    tables contain no `ridge_own_era5_flat12` rows. (b) "Strongest own-basin model"
    is a statement about pooled standardized MSE with equal basin weighting. In
    pooled raw centimetres `ridge_own_flat12` is lowest at leads 1 to 3 (5.118 cm at
    lead 1 against 5.128 for `kalman_ar1` and 5.231 for `ridge_own_era5_flat12`) and
    the ERA5 ridge trails the plain filter at every lead
    (`results/conventional_metrics_summary.csv`, `pooled_rmse_cm`). The manuscript
    must name the weighting whenever it ranks models. JPL-side audit findings (chain
    default JPL file differs from the provenance file; provenance sentence on
    bootstrap block length) are deferred to the JPL rerun.
