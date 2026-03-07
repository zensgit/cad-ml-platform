# Drift And Active Learning Coarse Observability Validation 2026-03-07

## Goal

Extend operational APIs with stable coarse/fine observability without breaking
existing response contracts:

- `GET /api/v1/active-learning/pending`
- `GET /api/v1/active-learning/stats`
- `GET /api/v1/analyze/drift`

## Changes

### Active Learning

Files:

- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `tests/test_active_learning_api.py`
- `tests/unit/test_active_learning_export_context.py`

Behavior:

- Pending samples now expose:
  - `predicted_fine_type`
  - `predicted_coarse_type`
  - `predicted_is_coarse_label`
- Samples are normalized on load and on creation so old stored samples remain
  compatible.
- Missing `sample_type` and `feedback_priority` are now derived eagerly, not
  only during export.
- Stats endpoint now exposes:
  - `sample_type_stats`
  - `feedback_priority_stats`
  - `predicted_coarse_stats`
  - `predicted_fine_stats`
  - `labeled_true_coarse_stats`
  - `labeled_true_fine_stats`
  - `correction_count`
- Export payload now preserves:
  - `predicted_fine_type`
  - `predicted_coarse_type`
  - `predicted_is_coarse_label`

### Drift

Files:

- `src/api/v1/drift.py`
- `tests/unit/test_drift_endpoint_coverage.py`

Behavior:

- Drift response now exposes coarse prediction observability:
  - `prediction_current_coarse`
  - `prediction_baseline_coarse`
  - `prediction_coarse_drift_score`
- Coarse metrics are derived from the same fine prediction baseline, so no new
  baseline storage format is required.

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  src/api/v1/drift.py \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_drift_endpoint_coverage.py

flake8 \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  src/api/v1/drift.py \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_drift_endpoint_coverage.py \
  --max-line-length=100

pytest -q \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_drift_endpoint_coverage.py \
  tests/contract/test_openapi_schema_snapshot.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `39 passed`
- OpenAPI snapshot: unchanged

## Notes

- Existing fields remain in place; new observability fields are additive.
- No schema baseline refresh was required because path/operation/component
  counts did not change.
