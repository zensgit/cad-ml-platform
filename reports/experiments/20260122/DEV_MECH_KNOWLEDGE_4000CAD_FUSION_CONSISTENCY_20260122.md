# DEV_MECH_KNOWLEDGE_4000CAD_FUSION_CONSISTENCY_20260122

## Summary
- Audited fusion consistency on batch analysis results with Graph2D enabled.
- All samples had Graph2D predictions, but only 57/110 responses used fusion as the confidence source.
- Mismatch summary highlights frequent disagreements between Graph2D labels and final part_type.

## Inputs
- Batch results: `reports/experiments/20260122/batch_analysis_graph2d/batch_results.csv`

## Outputs
- Mismatch list: `reports/experiments/20260122/batch_analysis_graph2d/fusion_mismatch.csv`
- Mismatch summary: `reports/experiments/20260122/batch_analysis_graph2d/fusion_mismatch_summary.csv`
- Coverage: `reports/experiments/20260122/batch_analysis_graph2d/fusion_coverage.json`

## Key Stats
- Total: 110
- Graph2D present: 110
- Confidence source: fusion=57, rules=53
- Mismatch count (part_type != graph2d_label): 102
- Top mismatches: complex_assembly→机械制图(19), moderate_component→零件图(11), moderate_component→机械制图(10), 盖→机械制图(9)

## Notes
- Consider lowering `FUSION_ANALYZER_OVERRIDE_MIN_CONF` or refining rule overrides to allow Graph2D labels to influence final output.
