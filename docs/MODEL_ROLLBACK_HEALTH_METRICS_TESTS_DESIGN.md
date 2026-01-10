# Model Rollback Health Metrics Tests Design

## Overview
Verify that `/health/model` increments model health metrics for ok and rollback
states when metrics are available.

## Tests
- `tests/unit/test_model_rollback_health.py`

## Validation
- `pytest tests/unit/test_model_rollback_health.py -v`
