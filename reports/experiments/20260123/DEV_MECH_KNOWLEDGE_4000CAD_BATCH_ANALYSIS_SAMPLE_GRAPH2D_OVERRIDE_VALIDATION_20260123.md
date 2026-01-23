# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_GRAPH2D_OVERRIDE_VALIDATION_20260123

## Checks
- Verified override run produced 20/20 successful results.
- Confirmed overrides applied to 3 samples and artifacts were written.

## Command
- `GRAPH2D_ENABLED=true GRAPH2D_FUSION_ENABLED=true FUSION_ANALYZER_ENABLED=true FUSION_ANALYZER_OVERRIDE=true FUSION_ANALYZER_OVERRIDE_MIN_CONF=0.5 FUSION_GRAPH2D_OVERRIDE_MIN_CONF=0.5 .venv-graph/bin/python scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample_graph2d_override --max-files 20 --seed 23 --min-confidence 0.6`

## Artifacts Verified
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample_graph2d_override/summary.json`
