# Qdrant Migration Pending Summary Validation - 2026-03-08

## Goal

Add a compact observability surface for pending migrations, grouped by source feature version.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-summary-20260308-000900/src/api/v1/vectors.py) with:

- `GET /api/v1/vectors/migrate/pending/summary`

Response fields:

- `target_version`
- `observed_by_from_version`
- `total_pending`
- `pending_ratio`
- `backend`
- `scanned_vectors`
- `scan_limit`
- `distribution_complete`

Behavior:

- exact `total_pending` and `pending_ratio` are returned only when distribution is complete
- under partial scan, only `observed_by_from_version` is exposed and exact totals remain `null`

Added coverage in:

- [test_vector_migration_pending_summary.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-summary-20260308-000900/tests/unit/test_vector_migration_pending_summary.py)
- [test_vector_migration_pending.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-summary-20260308-000900/tests/unit/test_vector_migration_pending.py)
- [test_vector_migration_pending_run.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-summary-20260308-000900/tests/unit/test_vector_migration_pending_run.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vector_migration_pending_summary.py
flake8 src/api/v1/vectors.py tests/unit/test_vector_migration_pending_summary.py --max-line-length=100
pytest -q tests/unit/test_vector_migration_pending_summary.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `9 passed`
