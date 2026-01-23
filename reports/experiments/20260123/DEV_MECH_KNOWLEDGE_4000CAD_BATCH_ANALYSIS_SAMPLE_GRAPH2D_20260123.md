# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_20260123

## Summary
- Re-ran a 20-sample DXF batch analysis using the Graph2D-enabled Python 3.11 venv.
- Graph2D predictions were emitted for all samples, but override was not enabled (confidence mostly below 0.6).

## Command
- `.venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample_graph2d --max-files 20 --seed 23 --min-confidence 0.6`

## Results
- Sample size: 20
- Success: 20, Error: 0
- Graph2D label counts: 机械制图=13, 装配图=4, 零件图=3
- Graph2D confidence >= 0.6: 2/20
- Part-type distribution unchanged vs. baseline run (override disabled by config).

## Artifacts
- `reports/experiments/20260123/batch_analysis_sample_graph2d/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/summary.json`

## Notes
- First import triggered a model hoster connectivity check; set `DISABLE_MODEL_SOURCE_CHECK=True` to skip.
