# DEV_LINT_E722_CLASSIFIER_API_20260204

## Summary
Replaced bare `except` clauses in DXF feature extraction/render loops with
`except Exception` to satisfy flake8 E722 in CI.

## Changes
- Updated `src/inference/classifier_api.py` to use explicit `Exception` handling in
  DXF entity parsing loops.

## Validation
- CI lint job should pass once the change is picked up.
