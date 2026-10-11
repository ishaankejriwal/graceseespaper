# What remains before the HESS submission

> **Update, 11 October 2026 (branch `hess-submission`).** The seven "missing" CSR files
> were present in the first author's checkout (they are git-ignored, so a fresh clone
> lacks them). With them, all seven figures were rebuilt with one Arial family for text and
> mathematics and spaces before percentage units, and every source assertion passed, so no
> plotted value changed. `make_manifest.py --check` verified the large files. Full
> institutional addresses and a separate Financial support section were added. The deposit is
> staged by `scripts/build_deposit.py` (README: `docs/DEPOSIT_README.md`), and the checker now
> fails until the availability statement cites the Zenodo DOI. **Still open:** (1) upload the
> deposit and cite its DOI; (2) the JPL row-level prediction files are not in this checkout,
> so the second author should add them to the deposit; (3) the AI-policy statement; (4) the
> author confirmations in Sect. 6. Sections 2 and 4 below are kept as the record of what was
> done; their boxes are ticked.

**Status checked: 10 October 2026.** This handoff describes the current
manuscript and files in this checkout. It covers submission preparation;
it is not a new review of the scientific conclusions.

The manuscript compiles, its PDF is current relative to the TeX source,
and the stored [technical report](submission/technical_report.md) records
**113 passing checks and no failed automated checks**. Six figure review
items remain. Passing those automated checks does not resolve missing
research data, incomplete addresses, or author declarations.

## 1. Overview of the remaining work

| Item | Current state | What is needed to finish | Input or decision needed from |
|---|---|---|---|
| Institutional addresses | Both school names are present, but affiliations only give the school and country | Confirm and add full institutional mailing addresses | Both authors |
| Supporting data and code archive | The availability statement still says the archive is being prepared | Assemble the underlying files, deposit the final version, obtain its persistent identifier, and cite it | Authors and the people holding the complete experiment outputs |
| Figures 1, 3, 5 and 6 | Existing PDFs remain; the corrected plotting script has not rebuilt these four | Restore seven missing CSR input files, rebuild, and inspect the outputs | Whoever has the original CSR run files |
| AI-policy review | Deferred at the user's request; no statement was inserted | Assess previous use against the journal policy and address any applicable disclosure/compliance issue | Authors; leave deferred until they are ready |
| Final author checks and registration | A local draft packet exists | Confirm metadata and declarations, refresh the packet after changes, and complete journal registration/upload | Authors |

The practical dependencies are: recovering the CSR files enables the
remaining figure rebuilds and helps complete the research archive;
final addresses and the archive citation enable the final manuscript
build; registration follows author review of that final version.

## 2. Complete the institutional affiliations

### Current entries

[main.tex](main.tex) currently contains:

```latex
\affil[1]{Johns Creek High School, United States}
\affil[2]{Illinois Mathematics and Science Academy, United States}
```

Both authors therefore already have named school affiliations. The
remaining task is **institutional address information**, not selecting a
school or adding an affiliation from scratch. 

HESS asks for full institutional addresses on the title page. Obtain the
author-approved school mailing address, including city, state, postal
code, and country, with the street address where appropriate. Check the
official institution names. [HESS manuscript composition guidance](https://www.hydrology-and-earth-system-sciences.net/submission.html#composition)

### Actions and completion criteria

- [x] Confirm the institutional address for Johns Creek High School.
- [x] Confirm the institutional address for Illinois Mathematics and
  Science Academy.
- [x] Update the two `\affil` entries in `paper/main.tex`.
- [x] Rebuild and inspect the title page for correct addresses and layout.

**Finished when:** both title-page affiliations contain the confirmed
institutional address, and both authors approve the entries.

## 3. Complete and cite the research archive

### Current gap

The code/data availability section in [main.tex](main.tex) links to the
GitHub repository and third-party data providers, but says that the
benchmark forecasts, per-prediction files, result tables and provenance
manifest are still being prepared for archival deposit. There is no final
archive DOI citation for this study.

The existing `paper/submission/source.zip` is a LaTeX compilation archive.
It does **not** contain the complete analysis code and supporting research
data needed to reproduce the findings. Likewise, `figures.zip` contains
finished illustrations, not the underlying plotting data.

HESS requests accessible research data in a reliable repository with a
persistent identifier, preferably a DOI, and a corresponding manuscript
and reference-list citation. Reviewers need access to the underlying
material. A GitHub link alone leaves this project's archive task
unfinished. [HESS data policy](https://www.hydrology-and-earth-system-sciences.net/policies/data_policy.html)

### Proposed contents for this project's deposit

Use [the archive manifest](../docs/ARCHIVE_MANIFEST.md) to determine which
files support each manuscript table, figure and numerical claim. The
deposit should contain, or provide documented persistent access to:

| Material | What to include or document |
|---|---|
| Analysis code | The exact repository revision used for the manuscript, including `src/`, the relevant scripts and reproduction instructions |
| Environment | `requirements-lock.txt`, Python/environment information and instructions for installing the required software |
| Processed basin inputs | Basin metadata, forecast-coverage information, basin storage series and any additional processed inputs needed by the reported analysis |
| CSR results | The reported result tables and the row-level forecast/prediction files behind them, including the seven missing files listed below where applicable |
| JPL results | The JPL results and underlying files used for the comparisons reported in this manuscript |
| Plotting provenance | `docs/ARCHIVE_MANIFEST.md`, `figures/BUILD_NOTES.md`, the plotting script and its input files |
| Integrity and provenance | Checksums, experiment configuration, product versions, fold definitions, dates and run information needed to identify the exact outputs |
| Third-party inputs | Dataset citations, versions, download/access instructions and processing steps; redistribute original data only where their terms permit |
| Archive README | File descriptions, units, model identifiers, regeneration steps, authors and appropriate code/data licensing information |

This is a project-specific deposit plan. The exact package can be
organized as one archive or linked code/data deposits, provided that the
manuscript identifies the relevant versions and readers can find the
supporting material. The journal data policy also describes justified
access restrictions; do not claim a completed public deposit unless its
access actually works. [HESS data availability guidance](https://www.hydrology-and-earth-system-sciences.net/policies/data_policy.html)

### Actions and completion criteria

- [ ] Restore the missing experiment outputs and reconcile them with the
  existing result tables and provenance.
- [ ] Assemble the deposit and check that every manuscript item in the
  archive manifest has its supporting files or documented access route.
- [ ] Choose the repository and finalize archive metadata and licences.
  Zenodo is already contemplated in `docs/ARCHIVE_MANIFEST.md`.
- [ ] Deposit the final files and obtain the actual persistent identifier.
- [ ] Check the landing page and file access from a session without the
  authors' repository login.
- [ ] Add the archive citation to [references.bib](references.bib), including
  creators, title, repository, year, identifier and relevant version.
- [ ] Cite that entry in the code/data availability section of `main.tex`.
- [ ] Replace the future-tense archive-preparation wording with accurate
  access information for the completed deposit.

**Finished when:** the supporting materials are accessible to reviewers,
the deposited version is identifiable, and the manuscript and bibliography
point to it correctly. Do not insert a guessed DOI.

## 4. Restore the missing inputs and rebuild four figures

### What still needs correction

| Figure | Existing output | Remaining change |
|---|---|---|
| 1 — basin sample | `figures/fig01_basins.pdf` | Rebuild with one font family for ordinary and mathematical lettering |
| 3 — example series | `figures/fig03_example_series.pdf` | Rebuild with one font family |
| 5 — where the forcing window helps | `figures/fig05_era5_where.pdf` | Rebuild with one font family and spaces before percentage units in annotations |
| 6 — sequence models | `figures/fig06_sequence_models.pdf` | Rebuild with one font family and spaces before percentage units in annotations |

These four PDFs currently mix Arial text with DejaVu lettering. Their
fonts are embedded, and their PNG exports already meet 300 dpi. The
remaining family/spacing work is recorded separately from the automated
resolution and embedding checks. HESS asks authors to use one font family
per figure and spaces between numbers and units.
[HESS figure guidance](https://www.hydrology-and-earth-system-sciences.net/submission.html#figures)

Inspection of the actual PDF text confirms that Figures 5 and 6 are almost
entirely Arial. In Figure 5, only five mathematical characters use
DejaVu Sans Oblique: two occurrences of `q`, and one each of `rho` (the
Greek symbol), `p`, and `n`. In Figure 6, only five characters use that
family: two occurrences of `h` and three of `p`. Arial regular and bold
belong to the same family and are not an additional family mismatch.
This is a small mathematical-lettering consistency issue, easily missed
by eye; it is not evidence that the whole figure needs redesigning.

The seven missing files below are needed by the existing plotting script
to regenerate the figures. They are not inherently necessary for changing
fonts in an existing vector file. Editing the actual vector text is an
alternative that has not yet been tested; it would need verification of
symbol positions and plot geometry. Recovering underlying research data
also serves the separate data-availability task in Section 3.

The fixes are already in [scripts/make_figures.py](../scripts/make_figures.py).
The blocker is the missing source data, not an unfinished plotting-code
change. The PNGs for these four figures were rendered from their existing
vector PDFs; that export did not change the original typography.

### Exact missing files

The following seven paths were checked and are **absent in this checkout
as of 10 October 2026**:

| Missing path, relative to repository root | Figure(s) needing it | Purpose |
|---|---|---|
| `data/processed/basin_meta.csv` | 1, 3, 5 | Basin roster, names, locations and metadata |
| `data/processed/li2026_basin_coverage.csv` | 1 | Classification of CSR basins by coverage in the GRACE-FCast comparison |
| `data/processed/basin_month_twsa_global.csv` | 5 | Monthly CSR basin storage series used to compute training-window variability |
| `results/phase2_baseline_predictions.csv` | 3 | Observations and damped-persistence forecasts for the example series |
| `results/kalman_predictions.csv` | 3 | Kalman forecasts used in the example series |
| `results/flat12_ridge_predictions.csv` | 5 | Basin-level forecast errors and significance calculations |
| `results/phase7_lstm_predictions.csv` | 6 | Predictions used for seed aggregation, model comparisons and bootstrap intervals |

The basin mask `HydroShed+Mascon_Basins_L3.nc` is present. The supporting
comparison, conventional-metric, LSTM-summary, training-sensitivity and
baseline-contrast CSVs used by these figures are also present. Full input
details are in [figures/BUILD_NOTES.md](../figures/BUILD_NOTES.md) and the
[archive manifest](../docs/ARCHIVE_MANIFEST.md).

### Recovery and rebuilding procedure

1. Recover these files from the original **CSR experiment run** that
   produced the manuscript's current result tables. Prefer the original
   outputs over rerunning the experiments merely to repair typography.
2. Put each file at its exact relative path above. Some large derived
   outputs and processed data are excluded from Git, so another ordinary
   repository clone may also lack them.
3. Confirm provenance and compare restored files with the existing
   checksum manifest where entries exist:
   `results/SHA256_MANIFEST_LIVE.csv`.
4. Preserve the distinction between CSR and JPL. Files under
   `results/jpl/` are not replacements for the missing CSR outputs.
5. Run the figure build using the existing environment. Its numerical
   assertions should pass against the reported results.
6. Inspect the rebuilt figures, including all labels, legends, hatching,
   percentage spacing and colour-vision inspection copies.

If the original files cannot be recovered, inspect the reproduction
instructions in [README.md](../README.md) and the inputs declared by the
relevant experiment scripts before scheduling a rerun. A rerun may need
additional source datasets and substantial computation; that work has
not been performed as part of the formatting pass.

If an assertion fails, investigate the difference between runs. Do not
change an expected number merely to make the figure build succeed. A real
result change may require corresponding manuscript/table updates.

- [x] Restore all seven missing files with matching provenance.
- [x] Rebuild Figures 1, 3, 5 and 6 successfully.
- [x] Confirm the font-family and percentage-spacing review items disappear.
- [ ] Repeat visual and colour-vision review on the new outputs.

**Finished when:** the actual submitted PDFs contain the corrected
typography, the figures reproduce the current reported values, and the
technical report no longer lists these six figure review items.

## 5. Resolve the deferred AI-policy review

**Status: deferred at the user's request.** No AI disclosure was added to
the manuscript, and this handoff does not classify the authors' earlier
use of AI.

The current HESS policy distinguishes assistive language checks, which
do not require declaration, from generative use. Where permitted
generative use is involved, the policy asks for the specific tool and its
purpose and extent in Methods or Acknowledgements. It also states:

> Generative AI must not be used for text or interpretations

Therefore, if earlier use included generating manuscript text or
interpretations, the authors need to address the policy compliance itself;
adding a disclosure should not be assumed to resolve that issue.
[HESS AI policy](https://www.hydrology-and-earth-system-sciences.net/policies/ai_policy.html)

When the authors resume this task:

- [ ] Review what tools were used, for what tasks, and which outputs entered
  the research or manuscript.
- [ ] Assess that use against the current policy.
- [ ] Address any applicable compliance issue and permitted-use disclosure.
- [ ] Make an accurate certification during manuscript registration.

**Finished when:** both authors have resolved the policy question and can
accurately complete the journal's registration certification. Until then,
keep this item explicitly deferred rather than marking it completed.

## 6. Final author review and registration

These are final author decisions and checks, separate from the missing
files and formatting fixes:

- [ ] Confirm author order and the equal-contribution statement.
- [ ] Confirm both correspondence emails on the title page.
- [ ] Approve the author-contribution description.
- [ ] Confirm the conflict-of-interest declaration.
- [ ] Confirm the existing statement that the research received no
  external funding, or supply the correct funding information.
- [ ] Read and approve the final manuscript, captions, tables and summary.
- [ ] Select the manuscript type; a research article is the current
  working assumption, not a completed registration decision.
- [ ] Select the subject areas and index terms for registration.
- [ ] Review and accept the journal's licence, terms and applicable
  article processing charges.
- [ ] Register and upload the final version through the journal system.

The registration/upload stage follows the
[HESS submission instructions](https://www.hydrology-and-earth-system-sciences.net/submission.html#getready).
No manuscript has been registered, uploaded or published by this
formatting workflow.

## 7. Rebuild and refresh the submission packet

After restoring the figure inputs and updating manuscript information,
run from the repository root:

```powershell
.venv\Scripts\python.exe scripts\make_figures.py
Push-Location paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
.venv\Scripts\python.exe scripts\check_hess_submission.py --bundle
```

The first command currently cannot complete because of the missing files
listed above. The commands rebuild figures, compile the manuscript, check
the technical requirements, and refresh the local packet. They do not
deposit research data or submit anything to the journal.

Inspect the regenerated
[technical report](submission/technical_report.md) and
[manuscript PDF](submission/manuscript.pdf). The checker distinguishes
failed automated checks from items needing review; inspect both parts.

| Local packet file | Purpose and remaining action |
|---|---|
| `paper/submission/manuscript.pdf` | Complete review manuscript; regenerate after final changes |
| `paper/submission/abstract.txt` | Plain-text abstract; refresh and paste into the upload form |
| `paper/submission/short_summary.txt` | Non-technical summary; refresh, confirm the character limit, and paste into the form |
| `paper/submission/figures.zip` | Flat archive of the seven vector figures; refresh after rebuilding |
| `paper/submission/source.zip` | Self-contained LaTeX source archive; refresh after all source changes |
| `paper/submission/technical_report.md` | Automated results and remaining figure review items |
| `paper/submission/accessibility/` | Inspection copies for colour-vision review; not submission illustrations |
| `paper/submission/SHA256.csv` | Checksums for the refreshed local packet |

Initial review uses the complete manuscript PDF plus the abstract and
short summary. Separate editable sources, the production figure archive
and a key figure are principally post-acceptance tasks. The current
supporting material is already included as appendices, so no separate
supplement is currently planned.
[HESS review and production file instructions](https://www.hydrology-and-earth-system-sciences.net/submission.html)

## 8. Work already completed

These items do not need to be restarted unless subsequent changes affect
them:

- Official Copernicus package 7.16 is installed; its downloaded template
  is preserved in `paper/template/`.
- The manuscript uses the HESS one-column class with line and page numbers.
- The manuscript has consistent portrait pages and embedded fonts.
- End-matter order, template commands, map credits, colour-map citation
  and missing reference article identifiers were addressed.
- Figures 2, 4 and 7 were rebuilt with one Arial font family and their
  existing numerical assertions passed.
- All seven PNG exports are 300 dpi; figure PDF raster layers meet the
  resolution check at their actual placements.
- Each figure PDF is below 2 MB and each PNG is below 5 MB; the stored
  report records 2.36 MB for the manuscript plus separate figure PDFs.
- The title page, large appendix table, rebuilt figures and colour-vision
  inspection sheets were visually reviewed.
- The local draft submission packet already exists and was checked for
  archive integrity when prepared.

For the broader preparation record, see
[HESS_SUBMISSION_CHECKLIST.md](HESS_SUBMISSION_CHECKLIST.md).

## 9. Submission readiness checklist

- [ ] Both institutional addresses are complete and author-approved.
- [x] The seven missing CSR files are recovered and their provenance is verified.
- [x] Figures 1, 3, 5 and 6 are rebuilt; all six figure review items are resolved.
- [ ] Every figure passes the final visual and colour-vision review.
- [ ] The research archive is accessible and its version/identifier is verified.
- [ ] The manuscript availability section and bibliography cite that archive.
- [ ] The deferred AI-policy question is resolved by the authors.
- [ ] Author metadata, contributions, interests and funding are confirmed.
- [x] The final manuscript and local submission packet are regenerated.
- [ ] The technical report has no failed checks or unresolved figure review items.
- [ ] Both authors approve the actual PDF and text being submitted.
- [ ] Journal registration and file upload are completed.
