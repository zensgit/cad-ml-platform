# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_OVERRIDE_20260123

## Summary
- Re-ran 20-sample DXF batch analysis with Graph2D override enabled (min conf 0.5).
- Overrides were applied to 3/20 samples, shifting labels toward 装配图.

## Command
- `GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample_graph2d_override --max-files 20 --seed 23 --min-confidence 0.6`

## Results
- Sample size: 20
- Success: 20, Error: 0
- Confidence buckets: 0.4-0.6=8, 0.6-0.8=4, >=0.8=8
- Low-confidence (<=0.6): 12
- Overrides applied: 3 (FusionAnalyzer rule_version)
- Label mix: complex_assembly=7, moderate_component=4, 蜗杆=2, 装配图=3, 箱体=1, 底板=1, 盖=1, 机械制图=1

## Override Hits
- J2925001-00再沸器v1.dxf: complex_assembly -> 装配图 (graph2d_conf=0.98)
- J2925001-00再沸器v2.dxf: complex_assembly -> 装配图 (graph2d_conf=0.98)
- J0724006-01下锥体组件v2.dxf: 机械制图 -> 装配图 (graph2d_conf=0.54)

## Artifacts
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/summary.json`

## Notes
- Model hoster connectivity check appeared once; set `DISABLE_MODEL_SOURCE_CHECK=True` to skip.
