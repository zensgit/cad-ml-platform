# DEV_MECH_KNOWLEDGE_4000CAD_GRAPH2D_OVERRIDE_LOW_CONF_LABEL_GUARD_20260122

## Summary
- Added a low-confidence guard for Graph2D overrides on "机械制图/零件图" labels.
- Re-ran batch DXF analysis with Graph2D override min 0.5 and Fusion override min 0.5 to validate impact.

## Code Changes
- `src/core/knowledge/fusion_analyzer.py` (per-label min confidence guard)
- `src/main.py` (env validation for low-conf guard)
- `.env.example` (documented low-conf guard envs)

## Inputs
- DXF directory: `/Users/huazhou/Downloads/训练图纸/训练图纸_dxf`
- Sample size: 110 (max-files=110, seed=22)
- Model: `models/graph2d_parts_upsampled_20260122.pth`
- Override labels: 模板, 零件图, 装配图
- Graph2D override min: 0.5
- Low-conf guard labels: 机械制图, 零件图
- Low-conf guard min: 0.6
- Fusion override min: 0.5

## Outputs
- Results: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/batch_results.csv`
- Label distribution: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/label_distribution.csv`
- Low-confidence samples: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/batch_low_confidence.csv`
- Summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/summary.json`
- Mismatch list: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_mismatch.csv`
- Mismatch summary: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_mismatch_summary.csv`
- Coverage: `reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf/fusion_coverage.json`

## Key Stats
- Success: 110/110
- Low-confidence (<=0.6): 53
- Confidence buckets: <0.4=0, 0.4-0.6=28, 0.6-0.8=28, >=0.8=54
- Graph2D present: 110
- Confidence source: fusion=62, rules=48
- Mismatch count (part_type != graph2d_label): 91
- Top labels: moderate_component(25), complex_assembly(23), 盖(11), 机械制图(8)

## Observations
- Graph2D predictions labeled "机械制图/零件图" never reached 0.5 confidence in this sample, so the new low-conf guard does not change outcomes yet. It is still useful as a safety rail for future runs once those labels improve.

## Commands
- `FUSION_GRAPH2D_OVERRIDE_ENABLED=true FUSION_GRAPH2D_OVERRIDE_LABELS=模板,零件图,装配图 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 FUSION_GRAPH2D_OVERRIDE_LOW_CONF_LABELS=机械制图,零件图 FUSION_GRAPH2D_OVERRIDE_LOW_CONF_MIN_CONF=0.6 FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 GRAPH2D_MODEL_PATH=models/graph2d_parts_upsampled_20260122.pth .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260122/batch_analysis_override_graph2d_fusion_min05_fusion05_guard_lowconf --max-files 110 --seed 22`
