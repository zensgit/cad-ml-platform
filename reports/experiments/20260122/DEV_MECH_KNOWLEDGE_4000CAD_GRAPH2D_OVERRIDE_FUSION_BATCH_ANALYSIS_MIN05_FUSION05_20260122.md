# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_FUSION_BATCH_ANALYSIS_MIN05_FUSION05_20260122

## Summary
- Re-ran batch DXF analysis with Graph2D override min confidence set to 0.5 and FusionAnalyzer override min confidence set to 0.5.
- Fusion override applied to all samples, shifting output distribution heavily toward Standard_Part.

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Sample size: 110 (max-files=110, seed=22)
- Model: `models/graph2d_parts_upsampled_20260122.pth`
- Override labels: 模板, 零件图, 装配图
- Override thresholds: Graph2D >= 0.5, Fusion override >= 0.5

## Outputs
- Results: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/batch_results.csv`
- Label distribution: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/label_distribution.csv`
- Low-confidence samples: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/batch_low_confidence.csv`
- Summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/summary.json`
- Mismatch list: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/fusion_mismatch.csv`
- Mismatch summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/fusion_mismatch_summary.csv`
- Coverage: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05/fusion_coverage.json`

## Key Stats
- Success: 110/110
- Low-confidence (<=0.6): 104
- Confidence buckets: <0.4=0, 0.4-0.6=81, 0.6-0.8=26, >=0.8=3
- Label distribution: Standard_Part(99), 装配图(9), 模板(2)
- Graph2D present: 110
- Confidence source: fusion=110, rules=0
- Mismatch count (part_type != graph2d_label): 99
- Top mismatches: Standard_Part→机械制图(70), Standard_Part→零件图(21)

## Commands
- `FUSION_GRAPH2D_OVERRIDE_ENABLED=true FUSION_GRAPH2D_OVERRIDE_LABELS=模板,零件图,装配图 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 GRAPH2D_MODEL_PATH=models/graph2d_parts_upsampled_20260122.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05 --max-files 110 --seed 22`

## Notes
- Lowering the FusionAnalyzer override threshold to 0.5 causes all responses to adopt fusion outputs, collapsing most labels to Standard_Part.
- This setting improves mismatch count slightly but introduces a strong label skew and a large low-confidence population.
