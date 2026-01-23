# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_50_GRAPH2D_OVERRIDE_COMPARE_20260123

## Summary
- Ran 50-sample DXF batch analysis with Graph2D override at 0.5 and 0.6 thresholds.
- Lowering the threshold to 0.5 increased override hits (7 vs 3) with slightly higher low-confidence count.

## Commands
- min 0.5: `GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 DISABLE_MODEL_SOURCE_CHECK=True .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_50_graph2d_override_min05 --max-files 50 --seed 23 --min-confidence 0.6`
- min 0.6: `GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.6 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.6 DISABLE_MODEL_SOURCE_CHECK=True .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_50_graph2d_override_min06 --max-files 50 --seed 23 --min-confidence 0.6`

## Results
- Min 0.5: overrides=7/50, low_conf=27, buckets: 0.4-0.6=14, 0.6-0.8=14, >=0.8=22
- Min 0.6: overrides=3/50, low_conf=25, buckets: 0.4-0.6=12, 0.6-0.8=14, >=0.8=24

## Override Hits (min 0.5)
- J2925001-00再沸器v1.dxf: complex_assembly -> 装配图 (0.98)
- J2925001-00再沸器v2.dxf: complex_assembly -> 装配图 (0.98)
- J0724006-01下锥体组件v2.dxf: 机械制图 -> 装配图 (0.54)
- J0225009-04-03阀体v2.dxf: 复杂分类 -> 模板 (0.57)
- J0724006-05上封头组件v1.dxf: complex_assembly -> 装配图 (0.70)
- J0724006-01下锥体组件v3.dxf: 机械制图 -> 装配图 (0.55)
- J0225009-04-03阀体v1.dxf: 复杂分类 -> 模板 (0.57)

## Override Hits (min 0.6)
- J2925001-00再沸器v1.dxf: complex_assembly -> 装配图 (0.98)
- J2925001-00再沸器v2.dxf: complex_assembly -> 装配图 (0.98)
- J0724006-05上封头组件v1.dxf: complex_assembly -> 装配图 (0.70)

## Artifacts
- `reports/experiments/20260123/batch_analysis_50_graph2d_override_min05/*`
- `reports/experiments/20260123/batch_analysis_50_graph2d_override_min06/*`
