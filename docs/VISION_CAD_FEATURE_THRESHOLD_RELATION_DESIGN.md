# Vision CAD Feature Threshold Relation Design

## Overview
Add a guard on CAD feature threshold overrides to ensure the arc fill range is
valid when both bounds are supplied.

## Behavior
- When both `arc_fill_min` and `arc_fill_max` are provided, enforce
  `arc_fill_min < arc_fill_max`.
- If either bound is omitted, no relation check is applied.

## Tests
- `tests/vision/test_vision_endpoint.py`
