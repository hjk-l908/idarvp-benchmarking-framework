# Stage 2 label definitions

Stage 2 in iDARVP is formulated as a seven-label multi-label mechanism-prediction task.

The seven labels used in this release subset are:

- `VIP` — viral infection peptide
- `VEIP` — viral entry inhibitory peptide
- `VINIP` — viral integration inhibitory peptide
- `MAP` — multi-activity annotated peptide
- `PIP` — protease inhibitory peptide
- `RTIP` — reverse transcriptase inhibitory peptide
- `SFIP` — syncytium formation inhibitory peptide

## Important note on MAP

In this project, `MAP` is used as an analysis-specific derived label for peptides associated with more than one of the single-mechanism labels.

Accordingly, `MAP` is not treated as mutually exclusive with the other mechanism labels in the Stage 2 multi-label setting.

## Source-policy note

These labels were derived by rule-based mapping from source target / mechanism annotations in the integrated ARVP master-table workflow, consistent with the Stage 2 task definition used in the manuscript-facing project materials.

This public staging subset is intended to document the release-facing label scheme used by the current manuscript and result files.
