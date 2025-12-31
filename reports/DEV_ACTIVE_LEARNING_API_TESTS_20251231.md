# Active learning API tests (2025-12-31)

## Changes
- Fixed active learning feedback error path to use `DATA_NOT_FOUND` error code.
- Added API coverage for pending, feedback, stats, and export routes.

## Command
- `.venv/bin/python -m pytest tests/test_active_learning_api.py -v`

## Results
- **Pass**: 6
- **Fail**: 0
- **Duration**: 36.72s
