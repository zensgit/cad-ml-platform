# Qdrant Migration Plan Coverage Validation

## Goal
- Extend `GET /api/v1/vectors/migrate/plan` with coverage-oriented summary fields.
- Show how many pending vectors are covered by the current plan window and how many remain outside the planned batches.

## Changes
- Added:
  - `planned_pending_count`
  - `remaining_pending_count`
  - `planned_pending_ratio`
- Coverage is computed from the selected `batches`.
- Ratio fields are only exposed when the pending distribution is complete.

## Validation
```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vector_migration_plan.py
flake8 src/api/v1/vectors.py tests/unit/test_vector_migration_plan.py --max-line-length=100
pytest -q \
  tests/unit/test_vector_migration_plan.py \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/unit/test_api_route_uniqueness.py
```

## Result
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `8 passed`

## Covered Scenarios
- Memory backend reports partial coverage when `max_batches` truncates the plan.
- Filtered memory plan reports full coverage when one batch covers all matching pending vectors.
- Partial-scan Qdrant plan keeps coverage counts but hides exact remaining ratio.

## Notes
- This PR is read-only.
- No migration execution behavior changed.
