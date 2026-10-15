# What remains before the HESS submission

**Status: 14 October 2026, branch `hess-submission`.** The manuscript, figures,
references and end matter meet the HESS rules checked by
`scripts/check_hess_submission.py`. The failing check is the data/code DOI below; the
"PDF newer than inputs" check clears on the final recompile. All deposit inputs,
including the 33 JPL row-level files from Ian's machine, are now in the first
author's checkout and pass both checksum checks. Earlier versions of this file are in
the git history.

## Open items

| Item | What to do | Who |
|---|---|---|
| Basin mask redistribution | `HydroShed+Mascon_Basins_L3.nc` (Teresa Howard and Benjamin Krichman) goes in the data archive under CC BY 4.0. Confirm they agree before publishing the deposit | Authors |
| Zenodo deposit and DOI | See "Deposit" below. Then cite the DOI in `\codedataavailability` and `references.bib`; the checker fails until a `10.5281/zenodo.*` DOI is cited | Authors |
| AI-use statement | Deferred by the authors. HESS asks for it in Methods or Acknowledgements and in the registration form. The working notes that mention AI tools (`results/RUN_LOG.md`, `paper/notes/`, `docs/EXTERNAL_AUDIT_PROMPT.md`) are left out of the Zenodo code archive but are on GitHub | Authors |
| Author confirmations | Order and equal contribution, contribution text, emails, school addresses, the SEES affiliation and acknowledgement, "no external funding" | Both authors |
| Registration | Research article; subject areas and index terms; licence, terms and APC; upload | Authors |

Licences are settled: MIT for code (`LICENSE`; the Copernicus LaTeX files under
`paper/` keep Copernicus's terms) and CC BY 4.0 for data. Choose the same two on Zenodo.

## Deposit

1. Commit everything. Check the large files with
   `.venv/Scripts/python.exe scripts/make_manifest.py --check` and again with the
   environment variable `GRACEFC_SOURCE=jpl` (25 CSR and 33 JPL files). Then run
   `.venv/Scripts/python.exe scripts/build_deposit.py`. This writes three zips to
   `deposit/`: the code at the commit (`git archive HEAD`, minus the export-ignore
   paths in `.gitattributes`), the large CSR and JPL result files, processed inputs and
   the basin mask, and the code at commit 71be5ec that produced the JPL tables.
2. On zenodo.org choose New upload, then Reserve DOI. Upload the three zips, paste
   `docs/DEPOSIT_README.md` as the description, set the licences, and publish.
3. Add a `@misc` entry with `howpublished = {Zenodo [code and data set]}` and the DOI,
   cite it in `\codedataavailability`, and replace the sentence that says the
   archive is being prepared.
4. Recompile (`pdflatex`, `bibtex`, `pdflatex` twice in `paper/`), then run
   `.venv/Scripts/python.exe scripts/check_hess_submission.py --bundle`. With all checks
   passing it writes the upload packet to `paper/submission/` (`manuscript.pdf`,
   `abstract.txt`, `short_summary.txt`, figure and source zips, colour-vision
   previews). The packet is generated, not tracked in git.

The code archive is `git archive HEAD`, so everything committed is published except
the paths marked export-ignore in `.gitattributes` (working notes, audits, the run log,
submission tooling and the Li and Kusche text copy).

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
- JPL deposit files (14 October): Ian's 33 row-level files checked against
  `results/jpl/SHA256_MANIFEST_LIVE.csv`; his 4 processed inputs reproduce the JPL
  baseline and GRACE-FCast comparison tables and are pinned in
  `results/jpl/RUN_PROVENANCE.md`.
- Code comments and docstrings rewritten for the archive (14 October), with every
  file checked to compute exactly what it did before.
