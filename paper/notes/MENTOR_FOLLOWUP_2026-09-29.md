# Mentor feedback follow-up, 29 September 2026

Reviewed the annotated `Kejriwal2026_ben_edits.pdf` against the newer
`paper/main.tex` and `paper/main.pdf`, dated 27 September. The comments in
the mentor's latest message still applied, although the newer introduction
had already added an explanation of the fitted noise term. Edits address
those comments and the passages needed to keep their terminology consistent.

The manuscript is `paper/main.tex`; its compiled PDF is `paper/main.pdf`.
Use these canonical files and Git history. The redundant delivery PDF
created during this revision was removed at the author's request.

| Mentor comment | Finding in the newer draft | Revision |
| --- | --- | --- |
| Define noise and avoid claiming physical signal separation. | The introduction contained a caveat, but the abstract, examples, methods and JPL interpretation still implied measurement-error removal. | Defined the additive observation-equation term as **model noise** and the forecast target as the deseasonalized **residual**. Explicitly ruled out separating physical variability from instrument or processing errors, or assessing satellite-data accuracy. Distinguished the additive term from process noise. |
| Line 35: replace "noisy instrument." | The phrase remained, along with a broad claim about errors in small and arid basins. | Changed it to "subject to observation noise" and removed the unsupported basin-specific error claim. |
| Line 408: "what is filtered out is presumed to be a mixture." | The limitations still asserted that the removed component was a known mixture. | Used "presumed to be a mixture" and included physical variability, instrument and processing errors, and errors in the trend and seasonal fit. Stated that their shares are not estimated. |
| Support GRACE-FCast claims. | The numerical results were supported, but some descriptions overstated what the comparison established. | Distinguished the forcing inputs to the non-seasonal networks from the use of past GRACE data to extrapolate trend and seasonality. Clarified individual-member verification, monthly versus fold-wise fitting, and the residual transformation. Removed the inference that training RMSE establishes a hindcast-accuracy ranking. Preserved the JPL release mismatch, basin-selection and operational-availability qualifications. |
| Line 351: Save does not support the drought-indicator claim. | The discussion still said drought indicators were "least certain" in these basins; the results also asserted that formal mascon errors were largest there. | Removed both claims. Described the forecast gains and residual scatter shown in this study's figures, without inferring observation error or drought-indicator uncertainty. |

The zero-variance fit is now called the **zero-r variant**, including in
Table 1 and the Figure 4 caption. Figure 4's legend and panel title use
the revised terminology. No forecast values or analysis inputs were changed.

GRACE-FCast descriptions were checked against [Li and Kusche (2026)](https://doi.org/10.1029/2025WR041710),
the locally archived full text, the [PANGAEA record](https://doi.pangaea.de/10.1594/PANGAEA.973113),
and `scripts/run_phase6_li_comparison.py`. Subtracting our climatology from
the full forecast and applying a training mean offset do not make the
authors' decomposition identical to ours. The revised methods state this
explicitly. The spatial-predictor paragraph also now describes GRACE-FCast's
forcing predictors accurately.

## Verification

- Recomputed all 30 CSR skill scores and Diebold-Mariano p-values reported
  in Table 5 from the saved, paired predictions on 209 basins and 60 months.
  Maximum differences from the saved headline were below 1e-15.
- Checked all 18 JPL skill scores against saved RMSE values and the
  reciprocal conversion from product-first skill. JPL raw predictions are
  unavailable locally, so its p-values were checked against the saved
  headline rather than independently recalculated.
- Checked all 48 printed skill/p-value pairs in Table 5 at their displayed
  precision. The numerical audit is saved in
  `tmp/pdfs/mentor_followup/fcast_audit.json`.
- Rebuilt Figure 4; its source-value assertions passed.
- Compiled the revised manuscript to 34 pages, with no undefined citations
  or references, overfull boxes, or unresolved cross-reference warnings.
  Rendered all pages for visual review and inspected the revised abstract,
  introduction, methods, examples and limitations at readable size. The
  existing class caption/subfloat warnings remain without visible defects.
- Applied the humanizer review and ran the requested Slopless checker on
  a Markdown export of manuscript prose. Raw findings are saved in
  `.slopless/findings/2026-09-29--mentor-followup.json`. Most findings concern
  paragraph length and repeated technical terms. Those do not justify a
  broad rewrite; the explicit scientific distinctions requested by the
  mentor were retained.

The analysis pipeline was not rerun or models retrained. This revision
addresses interpretation and wording; it does not establish equal input
availability or identical training conditions in the external comparison.
