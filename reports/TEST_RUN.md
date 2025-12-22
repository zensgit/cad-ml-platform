#!/usr/bin/env markdown
# Test Run Report

## Summary
- Adjusted ML classifier to refresh model settings from environment.
- Ran a focused regression suite across L3/L4, API integration, and model endpoints.

## Changes
- `src/ml/classifier.py`: refresh model path/version and reset loaded model on env change.

## Tests
- `python3 -m pytest tests/test_api_integration.py -q`
- `python3 -m pytest tests/test_l3_fusion_flow.py -q`
- `python3 -m pytest tests/test_l4_cost.py -q`
- `python3 -m pytest tests/test_l4_dfm.py -q`
- `python3 -m pytest tests/test_contract_schema.py -q`
- `python3 -m pytest tests/unit/test_active_learning_loop.py -q`
- `python3 -m pytest tests/unit/test_model_endpoint_coverage.py -q`
- `python3 -m pytest tests/unit/test_model_reload_endpoint.py -q`
- `python3 -m pytest tests/unit/test_ml_classifier_fallback.py -q`

## Results
- 45 passed
