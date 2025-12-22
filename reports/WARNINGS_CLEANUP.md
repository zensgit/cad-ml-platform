#!/usr/bin/env markdown
# Warnings Cleanup

## Summary
- Resolved Pydantic protected namespace warning for `model_version`.
- Aligned NumPy to match SciPy compatibility range.

## Changes
- Added `model_config` to deprecated `ModelReloadResponse` in `src/api/v1/analyze.py`.
- Added `model_config` to `ModelReloadResponse` in `src/api/v1/model.py`.
- Installed `numpy==1.24.3` (matches `requirements.txt`).

## Tests
- `python3 -m pytest tests/unit/test_active_learning_loop.py -q`
- `python3 -m pytest tests/test_l4_cost.py -q`

## Notes
- `pip` reported dependency conflicts with `faiss-cpu` and `nlopt` due to NumPy downgrade.
  If those packages are required, consider aligning their versions or relaxing pins.
