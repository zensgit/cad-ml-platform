# Vision CAD Feature Benchmark Compare CSV Design

## Overview
Add a comparison summary CSV output for benchmark runs that include a baseline
comparison. The CSV captures per-combo deltas for totals and averages so tuning
impact can be reviewed in spreadsheet tooling.

## CLI
- New flag: `--output-compare-csv <path>`
- Requires `--compare-json`.

## Output Schema
Columns:
- `combo_index`
- `status` (`ok` or `missing_baseline`)
- `total_lines_delta`
- `total_circles_delta`
- `total_arcs_delta`
- `avg_ink_ratio_delta`
- `avg_components_delta`

## Error Handling
- If `--output-compare-csv` is set without `--compare-json`, the script exits
  with an error.

## Command
```
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir data/dedup_report_train_local_version_profile_spatial_full_package/assets/images \
  --max-samples 20 \
  --threshold min_area=24 \
  --threshold line_aspect=6 \
  --threshold line_elongation=8 \
  --threshold circle_fill_min=0.4 \
  --threshold arc_fill_min=0.08 \
  --output-json reports/vision_cad_feature_tuning_compare_20260106.json \
  --output-csv reports/vision_cad_feature_tuning_compare_20260106.csv \
  --compare-json reports/vision_cad_feature_baseline_spatial_20260106.json \
  --output-compare-csv reports/vision_cad_feature_tuning_compare_summary_20260106.csv
```

## Outputs
- `reports/vision_cad_feature_tuning_compare_20260106.json`
- `reports/vision_cad_feature_tuning_compare_20260106.csv`
- `reports/vision_cad_feature_tuning_compare_summary_20260106.csv`
