# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_VALIDATION_20260123

## Checks
- Verified Graph2D predictions were populated in batch outputs.
- Confirmed summary + label distribution artifacts were written.

## Command
- `.venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample_graph2d --max-files 20 --seed 23 --min-confidence 0.6`

## Artifacts Verified
- `reports/experiments/20260123/batch_analysis_sample_graph2d/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d/summary.json`
