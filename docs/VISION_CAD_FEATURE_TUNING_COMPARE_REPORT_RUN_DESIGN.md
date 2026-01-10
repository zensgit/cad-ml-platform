# Vision CAD Feature Tuning Compare Report Run Design

## Overview
Generate a markdown comparison report for the tuned (non-grid) compare run to
summarize summary deltas and top sample changes.

## Command
```
python3 scripts/vision_cad_feature_compare_report.py \
  --input-json reports/vision_cad_feature_tuning_compare_20260106.json \
  --output-md reports/vision_cad_feature_tuning_compare_report_20260106.md \
  --top-samples 5
```

## Output
- `reports/vision_cad_feature_tuning_compare_report_20260106.md`
