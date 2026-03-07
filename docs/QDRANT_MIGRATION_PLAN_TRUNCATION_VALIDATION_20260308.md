# Qdrant Migration Plan Truncation Validation

## Goal
- Extend `GET /api/v1/vectors/migrate/plan` with truncation-aware summary fields.
- Make it explicit when the current `max_batches` window does not cover all pending source versions.

## Changes
- Added:
  - `coverage_complete`
  - `truncated_by_max_batches`
  - `unplanned_from_versions`
  - `suggested_next_max_batches`
- These fields summarize plan coverage at the top level without requiring clients to diff the batch list themselves.

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
- Memory backend shows truncation when `max_batches` excludes some source versions.
- Fully covered plans report `coverage_complete=true` and no unplanned source versions.
- Partial-scan Qdrant plans remain non-ready because of scan completeness, not batch truncation.

## Notes
- This PR only changes planning metadata.
- No migration execution path changed.
