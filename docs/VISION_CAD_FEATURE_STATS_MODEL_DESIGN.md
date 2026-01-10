# Vision CAD Feature Stats Model Design

## Overview
Add a typed Pydantic model for `cad_feature_stats` so the vision response
contract and examples stay explicit while still allowing forward-compatible
extensions.

## Model
`CadFeatureStats` captures the current heuristic summary output and is used in
`VisionAnalyzeResponse`:
- `line_count`, `circle_count`, `arc_count`: non-negative counts.
- `line_angle_bins`: histogram of line angles in degrees.
- `line_angle_avg`: average line angle or null when none.
- `arc_sweep_avg`: average arc sweep or null when none.
- `arc_sweep_bins`: histogram of arc sweep angles in degrees.

## Compatibility
- The model allows extra keys to preserve forward compatibility if new stats are
  added later.
- API responses remain JSON objects; only the schema typing is strengthened.

## Tests
- `pytest tests/test_contract_schema.py -v`
