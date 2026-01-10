# Vision CAD Feature Threshold Effect Design

## Overview
Add an API test that demonstrates CAD feature threshold overrides affect the
returned `cad_feature_stats` payload.

## Approach
- Generate a simple raster line drawing.
- Call `/api/v1/vision/analyze` with `include_cad_stats=true` and default thresholds.
- Call again with a strict `min_area` override to suppress detections.
- Assert the line count differs between the two responses.

## Notes
- Uses a synthetic image to keep the test deterministic.
- Confirms overrides flow through the API and change heuristic outcomes.
