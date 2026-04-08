# iDARVP public release staging subset

This repository is a manuscript-facing public release subset for the iDARVP study.

## Project framing

iDARVP is presented as a homology-aware benchmarking / analysis framework with a two-stage case-study implementation, rather than as a pure original-method predictor paper.

The current public staging subset is designed to support:
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

The current release subset also includes validation-selected threshold support files for selected models.

## Important scope note

This repository is a clean release subset, not a mirror of the full internal research workspace.

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

Please cite the associated manuscript when using this release subset.
See `CITATION.cff` for repository citation metadata.

## License

Code in this release subset is provided under the MIT License unless otherwise noted.

Users remain responsible for checking any third-party data-source restrictions when reconstructing upstream resources beyond the files directly included in this release subset.
