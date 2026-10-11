# HESS submission preparation

Checked against the live journal instructions on **9 October 2026**:
[submission and manuscript preparation](https://www.hydrology-and-earth-system-sciences.net/submission.html),
[data policy](https://www.hydrology-and-earth-system-sciences.net/policies/data_policy.html),
and [AI policy](https://www.hydrology-and-earth-system-sciences.net/policies/ai_policy.html).

The formatting work is complete where the required inputs are available.
This is a **draft preparation packet**, not a certification that the paper
is ready to submit. No manuscript was registered, uploaded or published.

## Applied changes

- Updated to the official Copernicus package **7.16**; preserved the
  downloaded package and `template.tex` in `template/`. The actual bundled
  class identifies itself as 10.1.35. See `template/PROVENANCE.md`.
- Kept `\documentclass[hess, manuscript]{copernicus}`: one column,
  portrait pages, consecutive page numbers and line numbers.
- Removed custom equal-contribution/class patches and the local table
  line-spacing redefinition. Equal contribution uses the standard template.
- Added the school affiliations for Ian Wang and Ishaan Kejriwal.
- Moved code/data availability after the appendices and before author
  contributions. Used the journal's standard conflict-of-interest wording.
- Defined the mission and processing-centre abbreviations in the abstract
  and again in the main text. Quantitative findings were preserved.
- Added HydroSHEDS citations and Natural Earth credits to all three map
  captions. Natural Earth is public domain; its credit is optional under
  the [provider's terms](https://www.naturalearthdata.com/about/terms-of-use/).
  HydroSHEDS attribution follows its
  [HydroBASINS guidance](https://www.hydrosheds.org/products/hydrobasins).
- Added the Crameri, Shephard and Heron (2020) reference required when using
  the scientific colour maps recommended by HESS.
- Added missing article identifiers to nine bibliography entries, checked
  against DOI/publisher records. Kept the official Copernicus BibTeX style.
- Rebuilt Figures **2, 4 and 7** from available source inputs and passed
  their existing numerical assertions. They use Arial for both ordinary
  text and mathematics. Exported every PNG at **300 dpi**. Figures **1, 3,
  5 and 6** were rendered from their existing vector PDFs, rather than
  enlarged from the 150 dpi previews.
- Updated the plotting script so future full builds use one font family,
  300 dpi PNGs, spaces before percentage units, and PDF/PNG size checks.
- Prepared a plain-text abstract and an abbreviation-free short summary
  of at most 500 characters, including spaces, from the manuscript.
- Inspected the title page, the large appendix table, all rebuilt figures,
  and colour-vision simulations for all seven figures. The inspection
  copies are in `submission/accessibility/` and are excluded from the
  figure/source ZIPs.

## Verified technical requirements

The generated [technical report](submission/technical_report.md) records
the checks for each file:

- Seven separate, correctly numbered, single-page composite figure PDFs;
  all are **17 cm wide**, above HESS's **8 cm minimum**. The 17 cm width is
  this project's production target, not an explicit maximum on the HESS
  submission page.
- Embedded fonts in the manuscript and figure PDFs; no optional hidden
  PDF layers in figure files.
- Raster map layers at 300 dpi at their actual placement in the figure
  PDFs, and 300 dpi PNG exports of consistent physical width. Vector lines
  and text do not have a raster dpi.
- Every figure PDF below **2 MB**, and every PNG below **5 MB**. The page
  also states a general 5 MB figure limit; the packet follows the stricter
  PDF-specific 2 MB limit.
- Manuscript below the **50 MB review PDF limit**; manuscript and separate
  figure PDFs together below the **30 MB overall submission limit**.
- Consistent portrait page dimensions; defined and unique references and
  labels; required end-matter sections; no undefined citations after build.

Tables remain editable LaTeX, with captions and horizontal rules, and
figures/captions remain near their first mention in the main text.
Ordinary supporting tables remain in Appendices A–C rather than being
moved to a separate supplement. URLs are written out with `\url`.

## Remaining before submission

1. **Institutional affiliation details.** Both school names are supplied.
   Confirm the schools' institutional address information for the final
   affiliations; this refers to school addresses rather than home addresses.
2. **Archive the code and supporting data with a persistent identifier.**
   The current availability statement explicitly says the archive is being
   prepared. Deposit the benchmark/prediction files, result tables and
   figure/table provenance in a suitable FAIR repository, obtain its real
   DOI, cite the deposited version in `references.bib`, and replace the
   future-tense statement. GitHub alone does not satisfy HESS's requested
   persistent data citation. Reviewers need access to the underlying data.
3. **Rebuild Figures 1, 3, 5 and 6 once missing inputs are restored.** Their
   existing PDF fonts are embedded, but they still mix Arial text with
   DejaVu math. HESS asks for one font family per figure. Figures 5 and 6
   also retain percentage annotations without spaces. The plotting code is
   corrected; the output PDFs have not been silently edited or regenerated
   from substitute data. Missing inputs include `data/processed/basin_meta.csv`,
   `data/processed/li2026_basin_coverage.csv`, the CSR basin storage table,
   and CSR `flat12_ridge_predictions.csv` / `phase7_lstm_predictions.csv`.
   Full input lists are in `figures/BUILD_NOTES.md`.
4. **Repeat visual review after the remaining figures are rebuilt.** The
   current seven figures were inspected with protanopia, deuteranopia and
   tritanopia simulations using the Machado et al. (2009) severity-100
   matrices, as published in
   [colorspacious's upstream table](https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py).
   Map signs and categories remain distinguishable; markers, line styles,
   open/filled symbols and significance hatching provide additional cues.
   These inspection copies approximate colour-vision deficiencies; they
   are not submission figures or a guarantee for every reader. Authors
   should repeat the visual check after changing any figure.
5. **AI-policy review is deferred at the user's request.** No AI statement
   was inserted. The current policy exempts assistive language checking but
   distinguishes it from generative use. Authors must assess earlier use
   against the live policy and complete any applicable disclosure and
   registration certification. Formatting checks alone do not resolve this.
6. **Author and registration checks.** Confirm both correspondence emails,
   author order/equal contributions, competing interests, and the existing
   no-external-funding statement. Select the manuscript type (this draft
   appears to be a research article), subject areas/index terms, and agree
   to the journal's licence, terms and article processing charges during
   registration. These author decisions have not been made on their behalf.

## Local files and rebuilding

From the repository root:

```powershell
python scripts/make_figures.py  # full build after restoring source inputs
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ..
python scripts/check_hess_submission.py --bundle
```

The checker fails on unmet technical requirements; it separately reports
items needing review. It also checks that the PDF is newer than its TeX,
bibliography, template and figure inputs.

`paper/submission/` contains:

| File | Use |
|---|---|
| `manuscript.pdf` | Draft complete review manuscript; update after resolving the items above |
| `abstract.txt` | Paste into the abstract field |
| `short_summary.txt` | Paste into the short-summary field; character count in technical report |
| `figures.zip` | Flat archive of seven vector figures, with no subfolders |
| `source.zip` | Self-contained LaTeX source, bibliography, template files and figures |
| `technical_report.md` | File dimensions, resolution, embedded fonts, sizes and outstanding technical review items |
| `accessibility/` | Original and simulated colour-vision inspection sheets for every figure |
| `SHA256.csv` | Checksums of the local generated packet |

**Initial review:** upload the complete PDF and paste the abstract/summary
after resolving remaining items. Separate sources and a production figure
archive are principally needed after acceptance. No separate supplement
has been prepared because the current manuscript already includes its
supporting material as appendices.

**After acceptance:** follow the production instructions for editable text,
tables and figure captions, the flat figure archive, and a key figure.
Select the key figure then; it is not an initial-review requirement. The
current source ZIP is a reproducible manuscript source archive, not the
post-acceptance text/table/caption layout requested by production.
