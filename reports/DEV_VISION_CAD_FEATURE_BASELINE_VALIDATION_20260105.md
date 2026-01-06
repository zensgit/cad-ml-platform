# DEV_VISION_CAD_FEATURE_BASELINE_VALIDATION_20260105

## Scope
Run the CAD feature benchmark on real raster samples to establish a default
baseline.

## Command
- `python3 scripts/vision_cad_feature_benchmark.py --input-dir data/train_artifacts_subset5 --output-json reports/vision_cad_feature_baseline_20260105.json --output-csv reports/vision_cad_feature_baseline_20260105.csv`

## Results
- `total_samples=5` (PNG files)
- `total_combos=1`
- Summary: `total_lines=48`, `total_circles=9`, `total_arcs=0`, `avg_ink_ratio=0.0174`, `avg_components=11.4`

## Outputs
- `reports/vision_cad_feature_baseline_20260105.json`
- `reports/vision_cad_feature_baseline_20260105.csv`

## Notes
- Optional vision provider clients were unavailable; heuristic extraction ran without them.
