# Results release notes

This folder stores the main manuscript-facing result artifacts included in the current public release subset.

## Main file groups

### 1. Stage 1 baseline files
Files beginning with:

- `stage1_baseline_lr_kmer13_`

support the released Stage 1 k-mer logistic-regression baseline summary.

### 2. Stage 2 summary files
Files ending with:

- `_summary.txt`

support manuscript-facing mean summary reporting, including mean AUPRC and mean AUROC for the corresponding released model.

### 3. Stage 2 per-label metric files
Files ending with:

- `_perlabel_metrics.csv`

support per-label AUPRC / AUROC / F1-style reporting for the corresponding released model.

### 4. Stage 2 Table 2 MCC traceability files
Files ending with:

- `_test_metrics_at_validMCC_threshold.csv`

support the released Stage 2 Table 2 macro-MCC traceability pathway.

For these files:
- thresholds were selected on the validation split by maximizing MCC
- those thresholds were then applied unchanged to `test_hom40`
- `macro-MCC` is the arithmetic mean of the seven `test_MCC` values in the file

### 5. Threshold support files
Files ending with:

- `_best_thresholds_by_label_validMCC.csv`

provide label-level threshold-selection support for selected models.

## Table 2 release summary

The file:

- `stage2_table2_release_summary.csv`

provides a compact release-facing summary for the manuscript Table 2 macro-MCC layer.

Please also see:

- `../docs/stage2_table2_traceability.md`

## Naming note

Internal provenance-oriented filenames are preserved where useful.

In particular:
- `cwbalanced` in filenames corresponds to the manuscript-facing balanced-control description:
  - `class_weight=balanced`

## Scope note

This folder is a clean manuscript-facing result subset, not a dump of every intermediate result generated during the full research workflow.
