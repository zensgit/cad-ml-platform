# Qdrant Migration Plan Runbook Validation 2026-03-08

## Goal
Extend `GET /api/v1/vectors/migrate/plan` so each recommended batch includes a ready-to-run request payload and operational notes.

## Files
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_migration_plan.py`
- `config/openapi_schema_snapshot.json`

## Behavior
- Adds `request_payload` to each batch for `POST /api/v1/vectors/migrate/pending/run`.
- Adds `notes` per batch:
  - `single_batch_ready`
  - `split_batch_required`
  - `partial_scan_override_required`
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
- Snapshot refreshed: `paths=178`, `operations=183`
