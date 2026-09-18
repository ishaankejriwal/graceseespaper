# Mentor draft revision, 18 September 2026

The revised manuscript is `paper/main.tex`; the compiled review copy is
`paper/main.pdf`. A delivery copy is `output/pdf/GRACE_mentor_review.pdf`.
The detailed cross-session handoff is `CONTINUATION_PROMPT.md` in this folder.

## Changes

- Reworked the abstract, introduction, interpretation of results, discussion,
  and conclusions using the humanizer skill. The abstract now defines
  persistence before explaining its limitation, following the author's
  request for an opening understandable to a GRACE researcher unfamiliar with
  this particular study.
- Corrected CSR/JPL native geometry and processing descriptions, the account
  of Kankanige's forward filter, and the characterization of Li (2024)'s
  main-text evaluation. Added provider data citations and missing reference
  identifiers.
- Replaced area-weighting claims with the actual variance-scaling distinction.
  Corrected the multiplication of MSE ratios and removed additive claims about
  skill overstatement across different benchmarks or studies.
- Made retrospective selection of the stronger damped-persistence reference
  explicit. Clarified two-sided tests, the monthly temporal sample, and the
  scope of false-discovery-rate adjustment. Replaced statements of statistical
  equivalence with the supported nonsignificant-difference interpretation.
- Qualified causal interpretations of the external forecast crossover and
  the noise ablation. Corrected the operational interpretation of observation
  latency and the claim that ridge models had never been run on JPL.
- Added metric labels to Tables B1/B2 and placed the entire `<0.001` expression
  in math mode. Also standardized this notation in the other tables. Added a
  CSR label to the basin counts beneath the JPL block in Table 5.
- Fixed clipped figure labels, distinguished the direct r=0 comparison from
  the shaded gap in Figure 4, and made Figure 5's off-scale basins visible with
  triangles at their true x positions. Captions describe these conventions.
- Kept Appendix B's tables together before Appendix C. Used single line
  spacing in Table C1, retaining its font size, so its introduction and table
  fit on one page.
- Removed the fictitious Zenodo DOI from rendered text and stated that the
  archival deposit is being prepared. No external deposit or publication was
  made.

## Verification

The statistics audit recomputed all 66 CSR benchmark rows from saved
predictions and verified all 54 printed nondamped-model RMSE/p-value pairs
after the table redesign. The original Table B1 values were correct; the
changes improve labeling and interpretation. Maximum RMSE and p-value
differences against the saved CSV were 4.44e-16 and 1e-16 respectively.

All seven figure builds passed their source assertions. LaTeX and BibTeX
completed successfully; the final manuscript has 32 pages. The final log has
no undefined citations/references, overfull boxes, or unresolved label-change
warnings. The class retains caption/subfloat compatibility warnings, which
did not produce observed layout defects. Pages were rendered with Poppler and
visually reviewed, with changed pages checked after layout and abstract edits.

The analysis inputs, stored predictions, and numerical results were not
changed. The full research pipeline and neural training were not rerun.

## Items for mentor review and before submission

1. Review the central recommendation and novelty against the closest monthly
   gravity/storage forecasting work. The citation audit was targeted, not a
   systematic literature review.
2. Consider uncertainty sensitivity to longer HAC bandwidths and bootstrap
   blocks, particularly for borderline comparisons. Current settings address
   forecast overlap but do not establish the absence of longer dependence.
3. Decide whether to report the available JPL correction-model results in more
   detail. This draft limits the detailed architecture-ranking claim to CSR;
   the JPL tables already in the project must not be described as nonexistent.
4. Preserve the scope of the external comparison: different verification
   populations, a JPL release mismatch, different training/gap treatment, and
   retrospective input availability. A matched operational experiment or
   blending experiment would be additional research.
5. Complete the public archive and verify its contents and real DOI before
   submission. Confirm the code repository's public accessibility and the
   final author, affiliation, acknowledgment, and journal requirements.

This is a revised mentor-review draft. Editorial and numerical consistency
checks do not substitute for scientific peer review or the remaining archive
and submission work.

## Abstract and prose revision, 18 September 2026

The abstract was rewritten to about 250 words. It now opens with how
data-driven storage forecasts are currently evaluated, states the gap in one
sentence, defines persistence in one sentence rather than a paragraph, and
puts the Kalman filter result before the correction models and the external
comparison. Every number in it was re-verified against the result CSVs.

Four scoping items were restored or added after an audit of that draft:

- The abstract again says the reference is the stronger of two
  damped-persistence variants, which is what Sect. 3.3 and the tables use.
- The introduction now states the survey finding the abstract rests on: of
  the ten data-driven TWSA forecasting studies examined, none reports a
  damped-persistence comparison. It was previously only a source comment.
- Appendix B says what the JPL table covers and that the twelve-month
  correction models are reported for CSR only. Item 3 below is unchanged:
  those JPL results exist and must not be described as nonexistent.
- The nonlinear-model sentence in the abstract now names its comparator.

Ten prose edits were applied in the body: over-long sentences split in
Methods, Results and Discussion; "beats" replaced with the neutral wording
used elsewhere; two filler connectives removed; the KF-R1 deficits at 1 and
2 months reported alongside its gains; the descending skill range written
low to high; the Fig. 5 x-axis description made consistent with the caption;
the Appendix C sentence rewritten without the trailing participle and given
the anomaly-only correlation and efficiency it compares against; a section
map added at the end of the introduction.

Figures were rebuilt. The Fig. 5b Spearman annotation no longer sits on a
data point, and Fig. 6b now writes small p values as < 0.001, matching the
convention declared in Sect. 3.7 and used in every table. Dead code and a
stale comment were deleted from the Fig. 7 builder. All source assertions
pass. The document compiles to 33 pages with no undefined references or
citations and no overfull boxes.
