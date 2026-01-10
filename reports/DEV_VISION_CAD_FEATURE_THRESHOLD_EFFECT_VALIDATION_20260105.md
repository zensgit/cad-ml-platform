# DEV_VISION_CAD_FEATURE_THRESHOLD_EFFECT_VALIDATION_20260105

## Scope
Validate that CAD feature threshold overrides affect response stats.

## Command
- `pytest tests/vision/test_vision_endpoint.py -v`

## Results
- `12 passed`

## Notes
- Added coverage asserting strict thresholds zero out detected line stats.
