# What remains before the HESS submission

**Status: 15 October 2026, branch `hess-submission`.** The manuscript, figures,
references and end matter meet the HESS rules checked by
`scripts/check_hess_submission.py`. The failing check is the data/code DOI below; the
"PDF newer than inputs" check clears on the final recompile. All deposit inputs,
including the 33 JPL row-level files from Ian's machine, were reported present in the first
author's checkout and passing both checksum checks on 14 October. Basin-mask and naming
release restrictions remain unresolved as described below. Earlier versions of this file are in
the git history.

## Open items

| Item | What to do | Who |
|---|---|---|
| Public basin mask | Benjamin Krichman's reply, shared on 15 October, permits study use of `HydroShed+Mascon_Basins_L3.nc` but prohibits its publication or public archival. It is excluded from the deposit. Independently generate and validate a replacement from HydroSHEDS basins intersected with the CSR land/ocean mask, using verified Pfafstetter codes or independently chosen names, or resolve the resulting reproducibility limitation before submission | Authors |
| No map or basin names in public releases | Remove map/basin names from every release artifact and use consistent numerical identifiers instead. Complete the release checks below before publishing; excluding the mask alone is insufficient | Authors |
| Zenodo deposit and DOI | See "Deposit" below. Then cite the DOI in `\codedataavailability` and `references.bib`; the checker fails until a `10.5281/zenodo.*` DOI is cited | Authors |
| AI-use statement | Deferred by the authors. HESS asks for it in Methods or Acknowledgements and in the registration form. The working notes that mention AI tools (`results/RUN_LOG.md`, `paper/notes/`, `docs/EXTERNAL_AUDIT_PROMPT.md`) are left out of the Zenodo code archive but are on GitHub | Authors |
| Author confirmations | Order and equal contribution, contribution text, emails, school addresses, the SEES affiliation and acknowledgement, "no external funding" | Both authors |
| Registration | Research article; subject areas and index terms; licence, terms and APC; upload | Authors |

Licences are settled: MIT for code (`LICENSE`; the Copernicus LaTeX files under
`paper/` keep Copernicus's terms) and CC BY 4.0 for the authors' original data/results.
Third-party terms still apply; the supplied basin mask is excluded from these grants.
Choose matching licences on Zenodo with those exceptions documented.

## Map and basin name release checks

- [ ] Replace map/basin names with consistent numerical identifiers in processed
  metadata, prediction files, result tables and saved model state. Check column
  values, column/index labels, filenames and embedded metadata. Do not release
  a lookup table mapping the identifiers back to the removed names.
- [ ] Remove map/basin names from figure labels, annotations, legends, captions,
  manuscript text, appendices and PDF metadata; inspect the rendered PDFs and PNGs.
- [ ] Check released source code, tests, documentation, notebooks, logs and provenance
  records for hard-coded names, examples or name-to-ID mappings.
- [ ] Extract and inspect the actual final Zenodo archives and HESS upload packet,
  including the historical JPL code archive and any nested ZIPs or binary files.
  Confirm that no map/basin names or the restricted supplied mask are included.
- [ ] Review files exposed through the public GitHub repository, including older
  revisions and downloadable artifacts, for the same names. Record any existing
  exposure and resolve how it will be handled before release.
- [ ] Verify that identifier changes preserve basin matching and scientific results.
  Record the reviewed commit and archive checksums; repeat the name checks after
  any subsequent changes. These checks remain open until explicitly verified.

## Deposit

1. Commit everything. Check the large files with
   `.venv/Scripts/python.exe scripts/make_manifest.py --check` and again with the
   environment variable `GRACEFC_SOURCE=jpl` (25 CSR and 33 JPL files). Then run
   `.venv/Scripts/python.exe scripts/build_deposit.py`. This writes three zips to
   `deposit/`: the code at the commit (`git archive HEAD`, minus the export-ignore
   paths in `.gitattributes`), the large CSR and JPL result files and processed inputs
   (excluding the supplied basin mask), and the code at commit 71be5ec that produced the JPL tables.
2. Resolve the basin-mask and naming release items above. On zenodo.org choose New upload,
   then Reserve DOI. Upload the three zips, paste
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
