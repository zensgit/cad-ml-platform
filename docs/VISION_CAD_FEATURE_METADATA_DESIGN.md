# Vision CAD Feature Metadata Design

## Overview
Expose lightweight CAD feature statistics in the local Vision analyzer metadata to
summarize detected geometry at a glance.

## Outputs
The metadata includes a `cad_feature_stats` object with:
- `line_count`
- `circle_count`
- `arc_count`
- `line_angle_bins` (0-30, 30-60, 60-90, 90-120, 120-150, 150-180)
- `line_angle_avg`
- `arc_sweep_avg`

## Notes
- Values are derived from the heuristic CAD feature extractor.
- Averages are `null` when no applicable components are present.
