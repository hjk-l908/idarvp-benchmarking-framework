# Stage 2 Table 2 traceability

This note explains how the Stage 2 Table 2 `macro-MCC` values are traced in this release subset.

## Core rule

For each model, the released file named:

- `*_test_metrics_at_validMCC_threshold.csv`

contains one row per Stage 2 label on `test_hom40`, where the threshold for that label was selected on the validation split by maximizing MCC and then applied unchanged to `test_hom40`.

`macro-MCC` in Table 2 is the arithmetic mean of the seven `test_MCC` values in the corresponding file.

`sd_perlabel_MCC` in Table 2 is reported as the sample standard deviation across the same seven `test_MCC` values.

## Model-to-file mapping

- `kmer13 baseline`
  - `results_release/stage2_baseline_lr_kmer13_test_metrics_at_validMCC_threshold.csv`

- `ESM2-t6 embedding baseline`
  - `results_release/stage2_esm2_t6_8M_UR50D_mean_ovr_lr_test_metrics_at_validMCC_threshold.csv`

- `Fusion(kmer13 + ESM2-t6)`
  - `results_release/stage2_fusion_esm2_t6_8M_UR50D_test_metrics_at_validMCC_threshold.csv`

- `Fusion(kmer13 + ESM2-t6, class_weight=balanced)`
  - `results_release/stage2_fusion_esm2_t6_8M_UR50D_cwbalanced_test_metrics_at_validMCC_threshold.csv`

- `Fusion(kmer13 + ESM2-t30)`
  - `results_release/stage2_fusion_esm2_t30_150M_UR50D_test_metrics_at_validMCC_threshold.csv`

## Threshold files currently included

The following threshold-support files are also included in this public staging subset:

- `results_release/stage2_baseline_lr_kmer13_best_thresholds_by_label_validMCC.csv`
- `results_release/stage2_fusion_esm2_t6_8M_UR50D_cwbalanced_best_thresholds_by_label_validMCC.csv`

These files provide label-level validation-selected thresholds for the corresponding models.

## Naming note

The release subset preserves existing internal filenames where needed for provenance.
In particular, `cwbalanced` in filenames corresponds to the manuscript-facing description:

`class_weight=balanced`

## Scope note

This staging subset is designed to support manuscript-facing traceability for the main Stage 2 comparison table.
It is not intended to include every intermediate working artifact from the full research workspace.
