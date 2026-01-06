# Vision CAD Feature Compare Report Design

## Overview
Provide a lightweight markdown report generator that converts benchmark
comparison JSON into a human-readable summary table.

## Script
- `scripts/vision_cad_feature_compare_report.py`

## CLI
- `--input-json`: benchmark JSON with a `comparison` block.
- `--output-md`: optional markdown output file.
- `--top-samples`: number of sample deltas to include (default 10).

## Output
- Markdown report with summary deltas and top sample deltas per combo.
