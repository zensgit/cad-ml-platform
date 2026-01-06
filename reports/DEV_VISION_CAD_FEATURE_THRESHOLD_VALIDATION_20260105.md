# DEV_VISION_CAD_FEATURE_THRESHOLD_VALIDATION_20260105

## Scope
Validate request-side CAD feature threshold validation.

## Command
- `pytest tests/vision/test_vision_endpoint.py -v`

## Results
- `10 passed`

## Notes
- Added coverage for invalid threshold keys returning HTTP 422.
