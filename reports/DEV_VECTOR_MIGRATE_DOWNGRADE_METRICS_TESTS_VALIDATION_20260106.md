# DEV_VECTOR_MIGRATE_DOWNGRADE_METRICS_TESTS_VALIDATION_20260106

## Scope
Validate vector migration downgrade metrics counter increments when collectors are available.

## Command
- `pytest tests/unit/test_vector_migrate_metrics.py -v`

## Results
- 1 passed, 2 skipped (`prometheus client disabled in this environment`).
