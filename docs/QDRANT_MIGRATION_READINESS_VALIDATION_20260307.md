# Qdrant Migration Readiness Validation - 2026-03-07

## Goal

Expose migration readiness signals on vector migration observability surfaces so operators can tell whether the current vector estate is fully migrated to the target feature version.

## Changes

Updated [src/api/v1/vectors.py](/private/tmp/cad-ml-platform-vectors-migrate-readiness-20260307-233600/src/api/v1/vectors.py) to add readiness fields on:

- `GET /api/v1/vectors/migrate/status`
- `GET /api/v1/vectors/migrate/summary`
- `GET /api/v1/vectors/migrate/trends`

Added fields:

- `target_version`
- `target_version_vectors`
- `target_version_ratio`
- `pending_vectors`
- `migration_ready`

Design choice:

- readiness numbers are returned only when version distribution is complete
- when Qdrant distribution is partial due to scan limiting, `target_version_vectors`, `target_version_ratio`, and `pending_vectors` return `null`
- `migration_ready` remains `false` under partial scans to avoid false certainty

Added coverage in:

- [tests/unit/test_vector_migration_status.py](/private/tmp/cad-ml-platform-vectors-migrate-readiness-20260307-233600/tests/unit/test_vector_migration_status.py)
- [tests/unit/test_migration_preview_trends.py](/private/tmp/cad-ml-platform-vectors-migrate-readiness-20260307-233600/tests/unit/test_migration_preview_trends.py)

## Validation

Commands:

```bash
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py

flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  --max-line-length=100

pytest -q tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py
```

Result:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `29 passed`

## Notes

- Default readiness target remains `v4`
- Operators can override the target with `VECTOR_MIGRATION_TARGET_VERSION`
- Partial-scan readiness intentionally avoids estimated pending counts
