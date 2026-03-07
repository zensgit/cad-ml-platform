# Qdrant Migration Plan Estimates Validation 2026-03-08

## Goal
Add run-count estimates to `GET /api/v1/vectors/migrate/plan` so operators can see how many executions are needed for each source version and in total.

## Files
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_migration_plan.py`

## Behavior
- Adds `estimated_runs_by_version`.
- Adds `estimated_total_runs`.
- Uses `default_run_limit` to compute the number of required runs.
- Keeps the endpoint read-only.

## Validation
```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vector_migration_plan.py
flake8 src/api/v1/vectors.py tests/unit/test_vector_migration_plan.py --max-line-length=100
pytest -q tests/unit/test_vector_migration_plan.py
python3 scripts/ci/generate_openapi_schema_snapshot.py --output config/openapi_schema_snapshot.json
pytest -q tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/unit/test_api_route_uniqueness.py
```

## Result
- Unit tests: `3 passed`
- OpenAPI contract tests: `5 passed`
- Snapshot regenerated successfully
