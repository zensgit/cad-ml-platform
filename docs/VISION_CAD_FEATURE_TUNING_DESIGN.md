# Vision CAD Feature Tuning Design

## Overview
Provide a lightweight benchmark script to evaluate heuristic CAD feature
extraction against a small image set. This enables manual tuning of the
threshold overrides without adding heavy dependencies.

## Script
- `scripts/vision_cad_feature_benchmark.py`

## Inputs
- `--input-dir`: directory of raster images (png/jpg/bmp/tif).
- `--max-samples`: limit the number of images.
- `--threshold-file`: JSON/YAML file with thresholds, grid, or variants (YAML requires PyYAML).
- `--threshold key=value`: override heuristic defaults.
- `--grid key=v1,v2,...`: sweep threshold values across combinations.
- `--output-json`: optional JSON export with thresholds and results.
- `--output-csv`: optional CSV export for grid results.
- `--compare-json`: baseline JSON to compare against.
- `--output-compare-csv`: optional summary CSV for comparison deltas.
- `--no-clients`: skip external client initialization during benchmarking.
- If `--input-dir` is omitted, the script generates synthetic samples.

## Outputs
- Console summary per image: counts for lines, circles, arcs, and ink ratio.
- Optional JSON payload with thresholds and results (`--output-json`).
- Optional `comparison` block when `--compare-json` is supplied.
- Optional CSV export with per-sample rows for each grid combination.
- Optional summary CSV for comparison deltas (`--output-compare-csv`).

## Compare Export
Use `scripts/vision_cad_feature_compare_export.py` to extract top sample deltas
from a compare JSON into JSON/CSV for analysis.

## Example
```
python3 scripts/vision_cad_feature_benchmark.py \
  --input-dir examples/cad_samples \
  --threshold line_aspect=5.0 \
  --threshold arc_fill_min=0.08
```

### Grid Sweep Example
```
python3 scripts/vision_cad_feature_benchmark.py \
  --grid line_aspect=4,5 \
  --grid arc_fill_min=0.05,0.08 \
  --output-json /tmp/cad_grid.json \
  --output-csv /tmp/cad_grid.csv
```

## Notes
- Thresholds map to the `cad_feature_thresholds` dict used in the analyzer.
- Synthetic samples provide a fast smoke check when real CAD renders are not
  available locally.
- API usage example: see README `Vision 分析响应（可选 CAD 特征统计）` section.
