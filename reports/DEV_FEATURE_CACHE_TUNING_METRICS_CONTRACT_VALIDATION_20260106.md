# DEV_FEATURE_CACHE_TUNING_METRICS_CONTRACT_VALIDATION_20260106

## Scope
Validate cache tuning metric label schemas (requests and recommendation gauges).

## Command
- `pytest tests/test_metrics_contract.py -k metric_label_schemas -v`
- `.venv/bin/python -m pytest tests/test_metrics_contract.py -k metric_label_schemas -v`

## Results
- 1 skipped, 21 deselected (`metrics client disabled in this environment`).
- 1 passed, 21 deselected (metrics enabled via `.venv`).
