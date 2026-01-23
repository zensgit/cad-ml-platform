# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_LOW_CONF_LABEL_GUARD_VALIDATION_20260122

## Checks
- Batch analysis completed with low-conf guard enabled for Graph2D overrides.
- Integration tests passed for DXF fusion flow.

## Tests
- `pytest tests/integration/test_analyze_dxf_fusion.py -v`

## Validation Outputs
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/batch_results.csv`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/label_distribution.csv`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/batch_low_confidence.csv`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/summary.json`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_mismatch.csv`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_mismatch_summary.csv`
- `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_coverage.json`

## Summary Snapshot
- Success: 110/110
- Fusion sources: 62
- Rules sources: 48
- Mismatch count: 91
