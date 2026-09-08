# Revision reproducibility notes

This release-candidate note records manuscript-facing corrections made for the BIB-26-1036 revision.

## Stage 1 threshold policy

Both Stage 1 baselines use validation-only threshold selection on 501 equally spaced thresholds from 0 to 1 inclusive (step 0.002). The first threshold attaining the maximum validation MCC is retained and applied unchanged to test sets. For kmer13 + logistic regression, the selected threshold is 0.452.

## Stage 1 uncertainty

The revised manuscript reports stratified CD-HIT80 cluster-bootstrap intervals (2,000 replicates, seed 42), a 100-seed repeated cluster-assignment robustness analysis, and sensitivity after excluding 11 Negative-B records with explicit Anti-HIV provenance. Release-facing summary tables are included under `results_release/revision_robustness/`.

## Stage 2 MAP terminology

VIP, VEIP, VINIP, PIP, RTIP, and SFIP are biological activity categories. MAP is retained only as an analysis-specific grouping for peptides mapped to multiple activity labels and is not interpreted as an independent biological mechanism.

## External web-tool pilot

The historical external web-tool pilot is not part of the reproducibility claim for this release candidate because paired per-sequence predictions were not preserved. It is treated only as supplementary descriptive context in the revised manuscript.
