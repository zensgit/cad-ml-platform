# Vision CAD Feature Compare Export Design

## Overview
Provide a lightweight export utility that turns the benchmark comparison JSON
into top-sample delta summaries for analysis tooling.

## CLI
- `--input-json <path>` (required)
- `--output-json <path>` (optional)
- `--output-csv <path>` (optional)
- `--top-samples <n>` (default: 10)
- `--combo-index <n>` (optional 1-based filter)

## JSON Output
```
{
  "top_samples": 10,
  "combo_exports": [
    {
      "combo_index": 1,
      "status": "ok",
      "thresholds": {"min_area": 24.0},
      "summary_delta": {"total_lines": -82},
      "top_samples": [
        {
          "name": "sample.png",
          "lines_delta": -5,
          "circles_delta": -3,
          "arcs_delta": 0,
          "ink_ratio_delta": 0.0,
          "components_delta": -8
        }
      ]
    }
  ]
}
```

## CSV Output
Columns:
- `combo_index`
- `status`
- `sample`
- `lines_delta`
- `circles_delta`
- `arcs_delta`
- `ink_ratio_delta`
- `components_delta`
- `thresholds`

## Error Handling
- Raises if the comparison block is missing or malformed.
- Validates `--combo-index` is >= 1 and within range.

## Command
```
python3 scripts/vision_cad_feature_compare_export.py \
  --input-json reports/vision_cad_feature_tuning_compare_20260106.json \
  --output-json reports/vision_cad_feature_tuning_compare_top_20260106.json \
  --output-csv reports/vision_cad_feature_tuning_compare_top_20260106.csv \
  --top-samples 10
```

## Outputs
- `reports/vision_cad_feature_tuning_compare_top_20260106.json`
- `reports/vision_cad_feature_tuning_compare_top_20260106.csv`
