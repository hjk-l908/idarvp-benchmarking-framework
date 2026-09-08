# iDARVP benchmarking framework — revision-aligned release

This repository is the manuscript-facing public release for the iDARVP study.

**Release target:** `v1.1.0` (revision-aligned release for manuscript BIB-26-1036).

## Project framing

iDARVP is presented as a homology-aware benchmarking / analysis framework with a two-stage case-study implementation, rather than as a pure original-method predictor paper.

This release is designed to support:
- release-facing data splits
- core manuscript workflow scripts
- main result summaries used in the manuscript
- traceability for the Stage 2 main comparison table

## Repository structure

- `data_release/`
  - release-facing split files and Stage 2 hom40 CLEAN table
- `scripts_release/`
  - core Stage 1 / Stage 2 workflow scripts
- `results_release/`
  - main manuscript-facing result files and Table 2 traceability files
- `docs/`
  - release-facing notes for label definitions, embeddings, reproducibility workflow, and Stage 2 Table 2 traceability

## What this subset is for

This subset is intended to let readers understand:
- how the release dataset subset is organized
- which scripts represent the main workflow
- which result files support the main manuscript claims
- how Stage 2 Table 2 macro-MCC values are traced

## Stage 2 Table 2 traceability

For the Stage 2 comparison table:
- mean AUPRC / mean AUROC are supported by the released summary files in `results_release/`
- macro-MCC traceability is documented in:
  - `results_release/stage2_table2_release_summary.csv`
  - `docs/stage2_table2_traceability.md`

The release also includes validation-selected threshold support files for selected models.

## Important scope note

This repository is a clean public release subset, not a mirror of the full internal research workspace.

It does not aim to expose every intermediate file, experimental byproduct, or internal planning document from the full project history.

## Embedding note

Embedding-based and fusion-based result files preserve model provenance in filenames, including:
- `esm2_t6_8M_UR50D`
- `esm2_t30_150M_UR50D`

Please see:
- `docs/embedding_notes.md`
- `docs/minimal_reproducibility_workflow.md`

## Label note

Please see `docs/stage2_labels.md` for the current Stage 2 label scheme used by the manuscript-facing release subset.

## Citation

Please cite the associated manuscript and, once the Zenodo archive is published, the DOI-backed software release. See `CITATION.cff` for repository citation metadata. The DOI will be added to the default branch after Zenodo completes archival of the `v1.1.0` GitHub release.

## License

Code in this release is provided under the MIT License unless otherwise noted. The MIT license does **not** override source-specific rights or reuse conditions associated with third-party peptide/database content or derived data. See `docs/data_and_third_party_notice.md`.

Users remain responsible for checking applicable source-database terms when reconstructing or redistributing upstream resources.


## Revision reproducibility note

For the BIB-26-1036 revision, Stage 1 validation-threshold calibration uses one common grid for the k-mer and ESM-2 baselines: 501 equally spaced thresholds from 0 to 1 inclusive (step 0.002), with the first threshold attaining the maximum validation MCC retained. For kmer13 + logistic regression this selects 0.452. MAP is treated as an analysis-specific multi-activity grouping rather than an independent biological mechanism. See `docs/revision_reproducibility_notes.md`.

## Revision-locked figures
The `figures_release/` directory contains the P3D revision-locked Figure 1-3 files. Figure 2 and Figure 3 can be regenerated from the release-facing result tables with `scripts_release/41_make_revision_figures.py`. See `docs/figure_source_map.md` for source mapping and interpretation boundaries.


## v1.1.0 revision release

This release synchronizes the public repository with the manuscript revision package: the unified Stage 1 threshold grid, corrected kmer13 threshold-dependent outputs, uncertainty/sensitivity results, revised Stage 2 label terminology, and revision-locked figures. It does not add a new model family or change the study's core benchmark scope. See `RELEASE_NOTES_v1.1.0.md`.
