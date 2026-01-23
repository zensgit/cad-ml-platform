# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_20260123

## Summary
- Ran a 20-sample DXF batch analysis on the local training set to spot-check FusionAnalyzer outputs.
- Graph2D inference was unavailable (Torch missing), so results reflect rules/L2 fusion only.

## Command
- `python3 scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample --max-files 20 --seed 23 --min-confidence 0.6`

## Results
- Sample size: 20
- Success: 20, Error: 0
- Confidence buckets: 0.4-0.6=9, 0.6-0.8=4, >=0.8=7
- Low-confidence (<=0.6): 13
- Label mix: complex_assembly=9, moderate_component=4, 蜗杆=2, 机械制图=2, 箱体=1, 底板=1, 盖=1

## Artifacts
- `reports/experiments/20260123/batch_analysis_sample/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample/summary.json`

## Notes
- Torch was not available in the local environment, so Graph2D predictions were skipped.
