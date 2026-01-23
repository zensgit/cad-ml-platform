# DEV_MECH_KNOWLEDGE_4000CAD_BATCH_ANALYSIS_SAMPLE_VALIDATION_20260123

## Checks
- Verified batch analyze completed with 20/20 success.
- Confirmed summary and distribution artifacts were written.
- Noted Graph2D inference disabled due to missing Torch.

## Command
- `python3 scripts/batch_analyze_dxf_local.py --dxf-dir /Users/huazhou/Downloads/训练图纸/训练图纸_dxf --output-dir reports/experiments/20260123/batch_analysis_sample --max-files 20 --seed 23 --min-confidence 0.6`

## Artifacts Verified
- `reports/experiments/20260123/batch_analysis_sample/batch_results.csv`
- `reports/experiments/20260123/batch_analysis_sample/batch_low_confidence.csv`
- `reports/experiments/20260123/batch_analysis_sample/label_distribution.csv`
- `reports/experiments/20260123/batch_analysis_sample/summary.json`
