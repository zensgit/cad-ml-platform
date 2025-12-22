#!/usr/bin/env markdown
# API Integration Test

## Summary
- Fixed missing `get_cost_estimator` import path used by the analyze API.

## Changes
- `src/core/cost/estimator.py`: added `get_cost_estimator()` singleton helper.
- `tests/test_api_integration.py`: uses valid STEP header content for signature check.

## Tests
- `python3 -m pytest tests/test_api_integration.py -q`

## Verification
- `cost_estimation` appears in analyze response when L4 options are enabled.
