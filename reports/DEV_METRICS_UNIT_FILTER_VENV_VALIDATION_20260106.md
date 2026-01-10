# DEV_METRICS_UNIT_FILTER_VENV_VALIDATION_20260106

## Scope
Run unit tests filtered by `-k metrics` with metrics enabled via `.venv`.

## Command
- `.venv/bin/python -m pytest tests/unit -k metrics -v`

## Results
- 223 passed, 3500 deselected.
