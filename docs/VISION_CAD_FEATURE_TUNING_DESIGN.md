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
Use `--combo-index` to filter a single grid combo when needed.
If no output files are provided, the script prints JSON to stdout.

## End-to-End Workflow
```
# Baseline (grid sweep)
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir /path/to/cad_images \
  --max-samples 20 \
  --threshold-file examples/cad_feature_thresholds.json \
  --output-json /tmp/cad_grid_baseline.json

# Compare (override a threshold, export summary CSV)
python3 scripts/vision_cad_feature_benchmark.py \
  --no-clients \
  --input-dir /path/to/cad_images \
  --max-samples 20 \
  --threshold-file examples/cad_feature_thresholds.json \
  --threshold min_area=24 \
  --output-json /tmp/cad_grid_compare.json \
  --compare-json /tmp/cad_grid_baseline.json \
  --output-compare-csv /tmp/cad_grid_compare_summary.csv

# Compare report
python3 scripts/vision_cad_feature_compare_report.py \
  --input-json /tmp/cad_grid_compare.json \
  --output-md /tmp/cad_grid_compare_report.md \
  --top-samples 10

# Export top sample deltas
python3 scripts/vision_cad_feature_compare_export.py \
  --input-json /tmp/cad_grid_compare.json \
  --output-json /tmp/cad_grid_compare_top.json \
  --output-csv /tmp/cad_grid_compare_top.csv \
  --top-samples 10
```

## Artifacts Summary
| Artifact | Description |
| --- | --- |
| `/tmp/cad_grid_baseline.json` | Baseline grid results (threshold sweep). |
| `/tmp/cad_grid_compare.json` | Compare run with overrides and comparison block. |
| `/tmp/cad_grid_compare_summary.csv` | Summary deltas per combo (CSV). |
| `/tmp/cad_grid_compare_report.md` | Markdown compare report for deltas. |
| `/tmp/cad_grid_compare_top.json` | Top sample deltas (JSON). |
| `/tmp/cad_grid_compare_top.csv` | Top sample deltas (CSV). |

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
