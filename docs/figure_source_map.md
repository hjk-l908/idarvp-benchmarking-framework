# Revision figure source map

## Figure 1
`figures_release/Figure_1_iDARVP_workflow_final_v1.*`

Conceptual workflow schematic aligned to the revised manuscript. The editable PPTX is included. The external AI4AVP/AVPpred pilot is deliberately shown as a supplementary exploratory branch, and Stage 2 explicitly separates six biological activity categories from MAP, an analytical multi-activity grouping.

## Figure 2
`figures_release/Figure_2_stage1_easy_hard_topK_final_v1.*`

Panel A uses `results_release/revision_robustness/stage1_cluster_bootstrap_CI_summary.csv` for kmer13 test_easy/test_hard AUPRC, AUROC, and MCC under the unified threshold 0.452.

Panel B uses:
- `stage1_kmer13_test_hard_topk_bootstrap_ci.csv`
- `stage1_esm2_t6_test_hard_topk_bootstrap_ci.csv`
- `stage1_topk_K50_100_200.csv`

Top-K intervals are sequence-level bootstrap intervals (2,000 replicates; seed 42) and are intentionally interpreted directionally because the intervals are wide and overlapping.

## Figure 3
`figures_release/Figure_3_stage2_hom40_perlabel_AUPRC_final_v1.*`

Values are read from the five `stage2_hom40_*_perlabel_metrics.csv` files in `results_release/` for `split=test_hom40`. Columns are displayed as six biological categories (VIP, VEIP, VINIP, PIP, RTIP, SFIP), followed by MAP as a visually separated analytical grouping.

## Boundary
No new model fitting or new scientific analysis was introduced during figure locking. Figure generation is a presentation/traceability step over already locked results.
