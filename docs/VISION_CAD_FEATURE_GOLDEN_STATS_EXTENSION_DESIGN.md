# Vision CAD Feature Golden Stats Extension Design

## Overview
Extend the CAD feature golden fixtures to cover additional angle and sweep bins
so regressions in binning logic are caught.

## Fixtures
Added to `tests/vision/fixtures/cad_features`:
- `cad_line_diagonal.png`: negative-slope line (expected 120-150° bin).
- `cad_arc_mid.png`: ~170° arc (expected 90-180° sweep bin).

## Tests
- `tests/unit/test_vision_cad_feature_golden_stats.py` now asserts the new bins.

## Notes
- Fixtures use grayscale PNGs for deterministic preprocessing.
- Expected stats align with default heuristic thresholds.
