# What remains before the HESS submission

**Status: 11 October 2026, branch `hess-submission`.** The manuscript, figures,
references and end matter meet the HESS rules checked by
`scripts/check_hess_submission.py` (131 of 132 checks pass). The one failing check
is the data/code DOI below. The earlier, longer version of this file (10 October)
is in the git history.

## Open items

| Item | What to do | Who |
|---|---|---|
| Zenodo deposit and DOI | See "Deposit" below. Then cite the DOI in `\codedataavailability` and `references.bib`; the checker fails until a `10.5281/zenodo.*` DOI is cited | Authors |
| JPL row-level predictions | Not in the first author's checkout. Add them to the deposit, or keep the sentence in `docs/DEPOSIT_README.md` saying they are excluded | Ian Wang |
| Licences | `docs/DEPOSIT_README.md` proposes MIT for code and CC BY 4.0 for data. Confirm, add a `LICENSE` file, and choose the same on Zenodo | Authors |
| Basin mask redistribution | `HydroShed+Mascon_Basins_L3.nc` (Teresa Howard and Benjamin Krichman) goes in the data archive. Confirm they agree | Authors |
| AI-use statement | Deferred by the authors. HESS asks for it in Methods or Acknowledgements and in the registration form | Authors |
| Author confirmations | Order and equal contribution, contribution text, emails, school addresses, the SEES affiliation and acknowledgement, "no external funding" | Both authors |
| Registration | Research article; subject areas and index terms; licence, terms and APC; upload | Authors |

## Deposit

1. Commit everything, then run `.venv/Scripts/python.exe scripts/make_manifest.py --check`
   and `.venv/Scripts/python.exe scripts/build_deposit.py`. This writes two zips to
   `deposit/` (code at the commit, and large CSR files, processed inputs and the
   basin mask).
2. On zenodo.org choose New upload, then Reserve DOI. Upload both zips (plus the JPL
   prediction files if available), paste `docs/DEPOSIT_README.md` as the
   description, set the licences, and publish.
3. Add a `@misc` entry with `howpublished = {Zenodo [code and data set]}` and the DOI,
   cite it in `\codedataavailability`, and replace the sentence that says the
   archive is being prepared.
4. Recompile (`pdflatex`, `bibtex`, `pdflatex` twice in `paper/`), then run
   `.venv/Scripts/python.exe scripts/check_hess_submission.py --bundle`. With all checks
   passing it writes the upload packet to `paper/submission/` (`manuscript.pdf`,
   `abstract.txt`, `short_summary.txt`, figure and source zips, colour-vision
   previews). The packet is generated, not tracked in git.

The code archive is `git archive HEAD`, so everything committed is published,
including `results/RUN_LOG.md` and the notes under `docs/` and `paper/notes/`.
Review that content against the AI-use statement before publishing the deposit.

## Done

- Copernicus LaTeX package 7.16 (`paper/template/`), HESS one-column manuscript
  with line and page numbers, end matter in the HESS order.
- Title page with full addresses, SEES affiliation, equal contribution.
- All seven figures rebuilt from the full CSR inputs: one Arial family, 300 dpi
  PNGs, spaced units, map credits, colour-vision checks.
- References verified against Crossref and DataCite, journal names abbreviated,
  data sets and preprints tagged, data sets cited by DOI.
- Number audit of about 800 numerals against the result files; referee-style,
  compliance, citation and reproducibility audits applied.
