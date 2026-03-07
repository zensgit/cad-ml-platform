# Qdrant Migration Plan Advisories Validation

## Goal
- Extend `GET /api/v1/vectors/migrate/plan` with execution-oriented advisories.
- Expose whether the current plan can be run safely without extra operator overrides.
- Surface the first recommended batch and request payload directly at the top level.

## Changes
- Added top-level response fields:
  - `plan_ready`
  - `blocking_reasons`
  - `recommended_first_batch`
  - `recommended_first_request_payload`
- Preserved existing batch-level `request_payload` and `notes`.
- Reused the existing partial-scan logic to drive top-level readiness and blocking output.

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
- Memory backend returns `plan_ready=true` and exposes the first executable batch.
- Qdrant partial scan returns `plan_ready=false` with `blocking_reasons=["partial_scan_override_required"]`.
- `from_version_filter` continues to narrow both batch selection and advisory output.

## Notes
- This is a read-only contract enhancement.
- No migration write path changed in this PR.
