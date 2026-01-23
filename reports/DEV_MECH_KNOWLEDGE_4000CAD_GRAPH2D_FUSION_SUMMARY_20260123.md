# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_FUSION_SUMMARY_20260123

## Scope
This summary consolidates recent Graph2D + FusionAnalyzer updates, validation runs, and
threshold selection for Graph2D overrides.

## Key Updates
- Graph2D node feature schema upgraded to 9 dimensions (border/title-block hints).
- Graph2D model loading now respects checkpoint node_dim and updated default model path.
- FusionAnalyzer schema v1.2 normalization expanded to L2/L3 features with calibrated confidence.
- Override logic aligned to calibrated confidence; override thresholds clamped to [0, 1].
- Startup validation warns on misconfigured Graph2D/Fusion env flags.

## Validation Summary
- DXF fusion integration tests (3 cases) pass with Graph2D override coverage.
- Batch analysis sample (20): baseline rules-only vs Graph2D vs Graph2D override.
- Batch analysis comparison (50): override thresholds 0.5 vs 0.6.

## Batch Analysis Highlights
### 20-sample baseline (rules/L2)
- Success: 20/20
- Low-confidence (<=0.6): 13
- Label mix dominated by complex_assembly/moderate_component.

### 20-sample Graph2D (override disabled)
- Graph2D labels populated; confidence >=0.6 only 2/20
- Part-type distribution unchanged (override not enabled)

### 20-sample Graph2D override (min 0.5)
- Overrides applied: 3/20
- Key shifts: complex_assembly/机械制图 -> 装配图 on three samples

### 50-sample override comparison
- Min 0.5: overrides=7/50, low_conf=27, buckets >=0.8=22
- Min 0.6: overrides=3/50, low_conf=25, buckets >=0.8=24

## Decision
- Default Graph2D override threshold remains **0.6**.
- Threshold **0.5** reserved for higher-coverage experiments.

## Reports
- `reports/experiments/20260123/DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_20260123.md`
- `reports/experiments/20260123/DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_20260123.md`
- `reports/experiments/20260123/DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_OVERRIDE_20260123.md`
- `reports/experiments/20260123/DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_50_GRAPH2D_OVERRIDE_COMPARE_20260123.md`
- `reports/experiments/20260123/DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_DEFAULT_RECOMMENDATION_20260123.md`

## Tests
- `pytest tests/integration/test_analyze_dxf_fusion.py -v`
