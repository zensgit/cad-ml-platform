# Qdrant Migration Plan Validation 2026-03-08

## Goal
Add a read-only `GET /api/v1/vectors/migrate/plan` endpoint that turns pending-summary data into an executable migration batch plan.

## Files
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_migration_plan.py`
- `config/openapi_schema_snapshot.json`

## Behavior
- Reuses existing pending-candidate scan path.
- Returns ranked batches by `from_version`.
- Exposes `suggested_run_limit` for each batch.
- Marks `allow_partial_scan_required=true` when Qdrant scan is partial.
- Supports `from_version_filter`, `max_batches`, and `default_run_limit`.

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
- Snapshot refreshed: `paths=178`, `operations=183`
