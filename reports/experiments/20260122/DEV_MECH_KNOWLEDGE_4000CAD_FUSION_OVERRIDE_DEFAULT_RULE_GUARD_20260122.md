# DEV_MECH_KNOWLEDGE_4000CAD_FUSION_OVERRIDE_DEFAULT_RULE_GUARD_20260122

## Summary
- Added a guard to skip FusionAnalyzer override when the decision is the default rule fallback (RULE_DEFAULT).
- Re-ran batch analysis with Graph2D override min 0.5 and Fusion override min 0.5 to validate the guard.

## Code Changes
- `src/api/v1/analyze.py` (skip fusion override when fusion decision is rule-based default)

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Sample size: 110 (max-files=110, seed=22)
- Model: `models/graph2d_parts_upsampled_20260122.pth`
- Override thresholds: Graph2D >= 0.5, Fusion override >= 0.5

## Outputs
- Results: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/batch_results.csv`
- Label distribution: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/label_distribution.csv`
- Low-confidence samples: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/batch_low_confidence.csv`
- Summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/summary.json`
- Mismatch list: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/fusion_mismatch.csv`
- Mismatch summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/fusion_mismatch_summary.csv`
- Coverage: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard/fusion_coverage.json`

## Key Stats
- Success: 110/110
- Low-confidence (<=0.6): 53
- Confidence buckets: <0.4=0, 0.4-0.6=28, 0.6-0.8=28, >=0.8=54
- Label distribution: moderate_component(25), complex_assembly(23), 盖(11), 机械制图(8)
- Graph2D present: 110
- Confidence source: fusion=62, rules=48
- Mismatch count (part_type != graph2d_label): 91

## Commands
- `FUSION_GRAPH2D_OVERRIDE_ENABLED=true FUSION_GRAPH2D_OVERRIDE_LABELS=模板,零件图,装配图 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 GRAPH2D_MODEL_PATH=models/graph2d_parts_upsampled_20260122.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard --max-files 110 --seed 22`

## Notes
- The guard prevents default-rule overrides from collapsing outputs to Standard_Part while retaining fusion overrides for higher-confidence signals.
