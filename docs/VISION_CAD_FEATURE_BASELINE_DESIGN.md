# Vision CAD Feature Baseline Benchmark Design

## Overview
Capture a real-sample baseline for the heuristic CAD feature extractor using
current default thresholds. The baseline provides a reference for future tuning
and regression checks.

## Dataset
- Input directory: `data/train_artifacts_subset5`
- Raster samples only (PNG files)

## Outputs
- JSON summary: `reports/vision_cad_feature_baseline_20260105.json`
- CSV detail: `reports/vision_cad_feature_baseline_20260105.csv`

## Command
```
python3 scripts/vision_cad_feature_benchmark.py \
  --input-dir data/train_artifacts_subset5 \
  --output-json reports/vision_cad_feature_baseline_20260105.json \
  --output-csv reports/vision_cad_feature_baseline_20260105.csv
```

## Notes
- Uses default `cad_feature_thresholds` (no overrides).
- Results serve as the baseline for future tuning comparisons.
