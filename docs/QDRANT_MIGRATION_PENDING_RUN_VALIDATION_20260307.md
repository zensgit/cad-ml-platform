# Qdrant Migration Pending Run Validation - 2026-03-07

## Goal

Add an operational write path that migrates vectors discovered by the pending-list surface, while keeping Qdrant partial-scan execution safe by default.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-run-20260307-235600/src/api/v1/vectors.py) with:

- shared helper: pending candidate collection for memory and Qdrant backends
- `POST /api/v1/vectors/migrate/pending/run`

Request fields:

- `limit`
- `dry_run`
- `allow_partial_scan`

Behavior:

- default target version comes from `VECTOR_MIGRATION_TARGET_VERSION` or falls back to `v4`
- Qdrant partial-scan runs return `409 CONSTRAINT_VIOLATION` unless `allow_partial_scan=true`
- when partial scan is explicitly allowed, only scanned pending IDs are migrated

Added coverage in:

- [test_vector_migration_pending.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-run-20260307-235600/tests/unit/test_vector_migration_pending.py)
- [test_vector_migration_pending_run.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-run-20260307-235600/tests/unit/test_vector_migration_pending_run.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py

flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py \
  --max-line-length=100

pytest -q tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `6 passed`
