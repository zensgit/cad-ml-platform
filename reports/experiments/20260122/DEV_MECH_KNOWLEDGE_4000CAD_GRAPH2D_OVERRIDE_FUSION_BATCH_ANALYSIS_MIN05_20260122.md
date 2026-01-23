# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_FUSION_BATCH_ANALYSIS_MIN05_20260122

## Summary
- Re-ran batch DXF analysis with Graph2D override threshold lowered to 0.5 while keeping FusionAnalyzer override at 0.6.
- Output distribution and mismatch counts remained unchanged versus the 0.6 override run.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Sample size: 110 (max-files=110, seed=22)
- Model: `models/graph2d_parts_upsampled_20260122.pth`
- Override labels: 模板, 零件图, 装配图
- Override thresholds: Graph2D >= 0.5, Fusion override >= 0.6

## Outputs
- Results: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/batch_results.csv`
- Label distribution: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/label_distribution.csv`
- Low-confidence samples: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/batch_low_confidence.csv`
- Summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/summary.json`
- Mismatch list: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/fusion_mismatch.csv`
- Mismatch summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/fusion_mismatch_summary.csv`
- Coverage: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05/fusion_coverage.json`

## Key Stats
- Success: 110/110
- Low-confidence (<=0.6): 70
- Confidence buckets: <0.4=0, 0.4-0.6=22, 0.6-0.8=51, >=0.8=37
- Top labels: moderate_component(25), Standard_Part(23), complex_assembly(22), 机械制图(6)
- Graph2D present: 110
- Confidence source: fusion=63, rules=47
- Mismatch count (part_type != graph2d_label): 101
- Top mismatches: Standard_Part→机械制图(21), complex_assembly→机械制图(16), moderate_component→零件图(11)

## Commands
- `FUSION_GRAPH2D_OVERRIDE_ENABLED=true FUSION_GRAPH2D_OVERRIDE_LABELS=模板,零件图,装配图 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6 GRAPH2D_MODEL_PATH=models/graph2d_parts_upsampled_20260122.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05 --max-files 110 --seed 22`

## Notes
- Lowering the Graph2D override min confidence to 0.5 did not change the overall mismatch count or label distribution in this sample.
