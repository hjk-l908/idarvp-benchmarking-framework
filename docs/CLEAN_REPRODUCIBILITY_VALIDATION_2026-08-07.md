# Revision release candidate reproducibility validation

Date: 2026-08-07
Scope: BIB-26-1036 revision release candidate P3C v0.1.

## Validation boundary

This was a fresh-directory validation from an isolated release-candidate copy using the packages already installed in the execution environment. It was not a dependency-installation test from a completely empty operating-system image. Therefore the result supports script/data reproducibility within the documented Python stack, but should not be described as a fully containerized clean-machine certification.

Validation environment:
- Python 3.13.5
- NumPy 2.3.5
- pandas 2.2.3
- scikit-learn 1.8.0
- PyTorch 2.10.0+cpu

## Stage 1 validation

Command pattern:

```bash
python scripts_release/00_stage1_baseline_lr.py \
  --train data_release/stage1_train_balanced_labelaware_with_seqid.tsv \
  --valid data_release/stage1_valid_balanced_labelaware_with_seqid.tsv \
  --test_easy data_release/stage1_test_easy_balanced_labelaware_with_seqid.tsv \
  --test_hard data_release/stage1_test_hard_negB_labelaware_with_seqid.tsv \
  --out_prefix <fresh-output-prefix>
```

PASS criteria and observed results:
- validation-only threshold grid: 0 to 1 inclusive, 501 points, step 0.002
- selected kmer13 threshold: 0.452
- validation MCC: 0.73226665
- test_easy AUROC / AUPRC / MCC: 0.96799268 / 0.97432625 / 0.76830648
- test_hard AUROC / AUPRC / MCC: 0.27907732 / 0.01057627 / -0.10001730
- test_hard TN / FP / FN / TP: 387 / 909 / 13 / 6

The generated Stage 1 metrics were promoted into `results_release/stage1_baseline_lr_kmer13_metrics.csv` for the revision-aligned release candidate.

## Stage 2 validation

Command pattern:

```bash
python scripts_release/00_stage2_hom40_baseline_lr.py \
  --tsv data_release/stage2_hom40_stage2_all_multilabel_hom40_CLEAN.tsv \
  --out_prefix <fresh-output-prefix>
```

PASS criteria and observed test_hom40 macro summaries:
- mean AUPRC: 0.8343
- mean AUROC: 0.9381
- mean F1@0.5: 0.7262
- freshly generated per-label metrics were byte-for-byte identical to the preserved release-facing `stage2_hom40_baseline_lr_kmer13_perlabel_metrics.csv`.

## Revision-alignment finding

The preserved 2026-08-06 public-repository snapshot still used the historical coarse Stage 1 threshold scan (`np.linspace(0.05, 0.95, 19)`) and selected threshold 0.45. The revision release candidate changes only the release-facing Stage 1 threshold search to the unified 501-point validation grid and synchronizes the corresponding metrics. The archived public snapshot remains unchanged for historical traceability.

## Key SHA256 values

- revised `scripts_release/00_stage1_baseline_lr.py`: `bd329c1539f843d6f805875c23c9f9ffe90d3872bde9d52fe5ff3c67f7f4084e`
- revised `results_release/stage1_baseline_lr_kmer13_metrics.csv`: `fcfb31578accd277b953376493ce17b8f37959ec6c3a84416f9ac2142ac1040e`
- validated Stage 2 per-label metrics: `b83de51a1dfd034f792ff70021cc40e00507b59979af6b949f73f5dae68df0aa`
- revised `docs/stage2_labels.md`: `db93939d7491b626c834b912e2f711b9748a8ca2d2cabcc31f13bff59bae3d15`
- revision reproducibility notes: `8955a335fe49132d005388a2eb2c40bb40286ea8034f38d78502c724676281ba`

## Status

PASS for fresh-directory regeneration of the transparent Stage 1 and Stage 2 k-mer baselines. Public GitHub synchronization and DOI archival remain release actions to perform after manuscript/figure content lock.
