# DEV_VISION_CAD_FEATURE_BENCHMARK_COMPARE_VALIDATION_20260106

## Scope
Validate benchmark comparison output against an existing baseline JSON.

## Command
- `python3 scripts/vision_cad_feature_benchmark.py --input-dir data/train_artifacts_subset5 --output-json reports/vision_cad_feature_baseline_compare_20260106.json --compare-json reports/vision_cad_feature_baseline_20260105.json`

## Results
- `comparison.summary_delta` zeros across totals.
- Per-sample deltas are all zero for the baseline run.

## Outputs
- `reports/vision_cad_feature_baseline_compare_20260106.json`
