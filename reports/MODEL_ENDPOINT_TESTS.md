#!/usr/bin/env markdown
# Model Endpoint Tests

## Summary
- Verified model endpoint responses and reload flows after Pydantic config changes.

## Tests
- `python3 -m pytest tests/unit/test_model_endpoint_coverage.py -q`
- `python3 -m pytest tests/unit/test_model_reload_endpoint.py -q`
- `python3 -m pytest tests/unit/test_ml_classifier_fallback.py -q`
