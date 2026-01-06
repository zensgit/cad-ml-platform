# Vision CAD Feature Grid Compare Report Run Design

## Overview
Generate a markdown comparison report from the multi-combo grid compare JSON to
summarize per-combo deltas and top sample changes.

## Command
```
python3 scripts/vision_cad_feature_compare_report.py \
  --input-json reports/vision_cad_feature_grid_compare_20260106.json \
  --output-md reports/vision_cad_feature_grid_compare_report_20260106.md \
  --top-samples 5
```

## Output
- `reports/vision_cad_feature_grid_compare_report_20260106.md`
