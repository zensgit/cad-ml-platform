# Vision CAD Feature API Response Design

## Overview
Expose heuristic CAD feature statistics in the vision analyze API response when
requested by the client.

## Request
- `include_cad_stats`: boolean flag to enable CAD feature stats.
- `cad_feature_thresholds`: optional overrides for the heuristic extractor.

## Response
- `cad_feature_stats`: summary counts and angle metrics for detected geometry.
  - `line_count`, `circle_count`, `arc_count`
  - `line_angle_bins`
  - `line_angle_avg`
  - `arc_sweep_avg`

## Notes
- When `include_cad_stats` is false, the field is omitted (null).
- Missing dependencies or parsing failures degrade gracefully without failing
  the main vision response.
