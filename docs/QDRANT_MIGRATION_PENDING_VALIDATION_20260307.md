# Qdrant Migration Pending Validation - 2026-03-07

## Goal

Add an operational endpoint to list vectors that have not yet reached the migration readiness target version.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-20260307-234800/src/api/v1/vectors.py) with:

- `GET /api/v1/vectors/migrate/pending`

Response fields:

- `target_version`
- `items[{id, from_version, to_version}]`
- `listed_count`
- `total_pending`
- `backend`
- `scanned_vectors`
- `scan_limit`
- `distribution_complete`

Behavior:

- memory backend returns exact `total_pending`
- qdrant backend returns exact `total_pending` only when the scanned distribution is complete
- under partial scan, `total_pending` is `null` to avoid false precision

Added coverage in:

- [test_vector_migration_pending.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-20260307-234800/tests/unit/test_vector_migration_pending.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py tests/unit/test_vector_migration_pending.py
flake8 src/api/v1/vectors.py tests/unit/test_vector_migration_pending.py --max-line-length=100
pytest -q tests/unit/test_vector_migration_pending.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `3 passed`
