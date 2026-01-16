# BREP_V4_INTEGRATION_EXTENDED_SHAPES_DESIGN

## Goal
Extend the B-Rep v4 integration test coverage with additional primitive shapes and tighter assertions on
surface types and bounding boxes.

## Changes
- Add sphere and torus STEP fixtures to the integration test.
- Assert bounding box dimensions for box, cylinder, sphere, and torus.
- Assert surface type histograms match expected primitive face types.

## Rationale
The original test validated entropy and surface counts for box/cylinder. Adding sphere/torus plus
bbox/type checks ensures the B-Rep extractor is consistent across common primitives.

## Testing
- `pytest tests/integration/test_brep_features_v4.py -v`
  - Local macOS run skips without pythonocc-core; linux/amd64 micromamba is the target validation.
