# iDARVP v1.1.0 — revision-aligned release

Release date: 2026-09-08

This release aligns the public iDARVP repository with the BIB-26-1036 revised manuscript and its locked analysis outputs.

## Changes since v1.0.0

- Standardized Stage 1 validation-threshold selection to 501 equally spaced thresholds from 0 to 1 (step 0.002).
- Updated the kmer13 validation-selected threshold to 0.452 and synchronized threshold-dependent Stage 1 outputs.
- Added revision robustness outputs: cluster-bootstrap confidence intervals, Top-K uncertainty, repeated cluster-assignment sensitivity, and the explicit Anti-HIV Negative-B exclusion sensitivity.
- Clarified the Stage 2 label scheme as six biological activity categories plus the MAP analytical multi-activity grouping.
- Added revision-locked Figures 1–3 and their source map/generation script.
- Added fresh-directory reproducibility validation notes and a repository-wide SHA256 manifest.
- Added a third-party data/licensing notice.

## Scope

This release does not introduce a new predictor family, fine-tuning experiment, or unrelated dataset. It is a reproducibility and revision-alignment release for the homology-aware benchmarking framework.

## Reproducibility

See `docs/CLEAN_REPRODUCIBILITY_VALIDATION_2026-08-07.md`, `docs/minimal_reproducibility_workflow.md`, and `SHA256SUMS.txt`.

## Citation and archival DOI

`CITATION.cff` contains software citation metadata. After the GitHub `v1.1.0` release is archived through the Zenodo GitHub integration, the resulting DOI should be added to the manuscript Availability statement, reviewer response, and repository default branch.
