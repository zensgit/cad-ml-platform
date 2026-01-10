# DEV_METRICS_UNIT_SECURITY_ROLLBACK_VECTOR_VENV_VALIDATION_20260106

## Scope
Run metrics-focused unit tests with metrics enabled for model security/rollback and vector migrate metrics.

## Command
- `.venv/bin/python -m pytest tests/unit/test_model_security_validation.py tests/unit/test_model_rollback_health.py tests/unit/test_model_rollback_level3.py tests/unit/test_vector_migrate_metrics.py tests/unit/test_vector_migrate_dimension_histogram.py -v`

## Results
- 61 passed.
