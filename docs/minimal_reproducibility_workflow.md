# Minimal reproducibility workflow

This note describes the minimum manuscript-facing workflow represented by the current public staging subset.

## 1. Data layer

The `data_release/` folder provides the main release-facing split files used to support the current manuscript subset, including:

- Stage 1 train / valid / test_easy / test_hard split files
- Stage 2 hom40 CLEAN table
- Stage 2 label-coverage / split summary files

## 2. Script layer

The `scripts_release/` folder provides the core workflow subset for:

- Stage 1 k-mer logistic-regression baseline
- Stage 2 hom40 k-mer logistic-regression baseline
- TSV-to-FASTA conversion
- ESM-2 mean embedding extraction
- embedding matrix construction
- Stage 1 embedding baseline
- Stage 2 embedding baseline
- Stage 2 fusion baseline
- Stage 2 threshold-based MCC collection / aggregation

## 3. Result layer

The `results_release/` folder provides the main manuscript-facing release artifacts for:

- Stage 1 baseline result summary files
- Stage 2 per-label and summary metric files
- Stage 2 balanced-fusion release files
- Stage 2 Table 2 MCC traceability files
- Stage 2 validation-selected threshold support files

## 4. Main manuscript-facing traceability

In the current release subset:

- Stage 1 main baseline summaries are supported by `stage1_baseline_lr_kmer13_*`
- Stage 2 mean AUPRC / mean AUROC summaries are supported by `stage2_hom40_*_summary.txt`
- Stage 2 per-label AUPRC / AUROC files are supported by `*_perlabel_metrics.csv`
- Stage 2 Table 2 macro-MCC traceability is documented in:
  - `results_release/stage2_table2_release_summary.csv`
  - `docs/stage2_table2_traceability.md`

## 5. Scope limitation

This public staging subset is not intended to mirror the full internal research workspace.

Its purpose is to provide a clean, manuscript-facing reproducibility subset that aligns the released split files, core scripts, and the main release result artifacts used in the paper.
