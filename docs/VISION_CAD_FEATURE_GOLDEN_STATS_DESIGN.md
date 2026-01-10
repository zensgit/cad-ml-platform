# Vision CAD Feature Golden Stats Design

## Overview
Add golden raster fixtures and regression tests that lock the expected
`cad_feature_stats` output for line, circle, and arc samples. This guards the
heuristic extractor against silent regressions.

## Fixtures
Stored in `tests/vision/fixtures/cad_features`:
- `cad_line.png`: horizontal line sample
- `cad_circle.png`: filled circle sample
- `cad_arc.png`: 240-degree arc sample

## Tests
- `tests/unit/test_vision_cad_feature_golden_stats.py`
- Assertions cover counts and angle/sweep bins for each fixture.

## Notes
- Fixtures use grayscale PNGs for deterministic preprocessing.
- Expected bins align with the default thresholds in the heuristic extractor.
