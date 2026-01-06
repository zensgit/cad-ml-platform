# DEV_VISION_CAD_FEATURE_SWEEP_BINS_API_VALIDATION_20260105

## Scope
Validate arc sweep bin exposure in the vision analyze API response.

## Command
- `pytest tests/vision/test_vision_endpoint.py -v`

## Results
- `13 passed`

## Notes
- Added endpoint coverage to confirm `arc_sweep_bins` is populated for arc inputs.
