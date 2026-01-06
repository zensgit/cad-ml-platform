# DEV_VISION_CAD_FEATURE_THRESHOLD_RANGE_VALIDATION_20260105

## Scope
Validate CAD feature threshold value range enforcement.

## Command
- `pytest tests/vision/test_vision_endpoint.py -v`

## Results
- `11 passed`

## Notes
- Added coverage for non-positive threshold values returning HTTP 422.
