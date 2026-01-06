# Vision CAD Feature Threshold File Design

## Overview
Add a `--threshold-file` option to load benchmark thresholds from JSON or YAML
so teams can version tuning presets.

## Supported Formats
JSON/YAML payloads accept any of:
- `thresholds`: mapping of numeric overrides.
- `grid`: mapping of threshold keys to numeric lists for sweep.
- `variants`: list of full threshold mappings (explicit combos).

If the payload is a list, it is treated as `variants`.

## CLI Behavior
- CLI `--threshold` overrides values from the file.
- CLI `--grid` values merge with file `grid` (CLI wins on key conflicts).
- If `variants` are provided, grid sweeps are ignored.
- YAML files require PyYAML.

## Example
```json
{
  "thresholds": {"min_area": 12, "line_aspect": 4},
  "grid": {"arc_fill_min": [0.05, 0.08]}
}
```
