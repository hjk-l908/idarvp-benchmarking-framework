# Embedding notes

This public staging subset includes the core scripts needed to reconstruct the embedding-based paths used in the manuscript-facing analyses.

## Models used in the current release subset

The current Stage 2 release subset includes result files associated with:

- `esm2_t6_8M_UR50D`
- `esm2_t30_150M_UR50D`

These model names are preserved in result filenames for provenance and traceability.

## Core embedding-related scripts included

The current `scripts_release/` folder includes:

- `10_tsv_to_fasta.py`
- `11_esm2_extract_mean.py`
- `20_build_embedding_matrix.py`
- `21_embedding_stage1_binary_baseline.py`
- `22_embedding_stage2_multilabel_baseline.py`
- `34_stage2_fusion_kmer_emb_ovr_lr.py`

## Release-scope note

This public staging subset does not aim to bundle every large embedding artifact from the full research workspace.

Instead, it preserves:
- the data splits needed to identify the release dataset
- the scripts needed to reconstruct the embedding workflow
- the release-facing result summaries used in the manuscript

## Practical interpretation

- `embedding baseline` results correspond to Stage 2 models that use ESM-2-derived mean embeddings with one-vs-rest logistic regression.
- `fusion` results correspond to models that combine k-mer features with ESM-2-derived mean embeddings.

## Provenance note

Internal filenames such as `cwbalanced` are preserved where needed for reproducibility and alignment with existing server-side result artifacts.
