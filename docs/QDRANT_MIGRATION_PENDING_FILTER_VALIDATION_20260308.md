# Qdrant Migration Pending Filter Validation - 2026-03-08

## Goal

Allow phased migration by filtering pending vectors by source feature version across list, summary, and run surfaces.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-filter-20260308-001800/src/api/v1/vectors.py) with:

- `GET /api/v1/vectors/migrate/pending?from_version_filter=...`
- `GET /api/v1/vectors/migrate/pending/summary?from_version_filter=...`
- `POST /api/v1/vectors/migrate/pending/run` request field `from_version_filter`

Behavior:

- list, summary, and run now share one pending-candidate collection path
- filtering is applied before pending counts, summaries, and migration execution
- response payloads echo `from_version_filter` on read surfaces for operator clarity

Added coverage in:

- [test_vector_migration_pending.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-filter-20260308-001800/tests/unit/test_vector_migration_pending.py)
- [test_vector_migration_pending_run.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-filter-20260308-001800/tests/unit/test_vector_migration_pending_run.py)
- [test_vector_migration_pending_summary.py](/private/tmp/cad-ml-platform-vectors-migrate-pending-filter-20260308-001800/tests/unit/test_vector_migration_pending_summary.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vector_migration_pending_summary.py

flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vector_migration_pending_summary.py \
  --max-line-length=100

pytest -q tests/unit/test_vector_migration_pending.py \
  tests/unit/test_vector_migration_pending_run.py \
  tests/unit/test_vector_migration_pending_summary.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `12 passed`
