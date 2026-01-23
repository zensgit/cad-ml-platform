# DEV_TRAINING_DXF_2D_GRAPH_FUSION_IMPACT_20260123

## Summary
- Compared DXF batch analysis results with FusionAnalyzer override disabled vs enabled.
- With override enabled, 35/50 samples changed classification, largely switching to the Graph2D label `other`.

## Results
- Changes: 35/50 (70%)
- Override targets: `other` (33), `再沸器` (2)
- Baseline top labels: complex_assembly(14), moderate_component(13), 机械制图(8)
- Override top labels: other(33), 机械制图(6), complex_assembly(4)

## Notes
- The cleaned Graph2D model predicts `other` with high confidence, so fusion override can dominate.
- Recommended follow-up: restrict override labels or exclude `other` from L4 fusion inputs.

## Artifacts
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_no_override/*`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_override/*`
- `reports/experiments/20260123/dxf_batch_analysis_graph2d_fusion_compare/override_diff.csv`
