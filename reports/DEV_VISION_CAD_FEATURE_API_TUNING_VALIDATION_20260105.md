# DEV_VISION_CAD_FEATURE_API_TUNING_VALIDATION_20260105

## Scope
Validate API request support for CAD feature heuristic threshold overrides.

## Command
- `pytest tests/unit/test_vision_api_coverage.py -v`

## Results
- `19 passed`

## Notes
- Added coverage to ensure the request carries `cad_feature_thresholds` through the endpoint handler.
