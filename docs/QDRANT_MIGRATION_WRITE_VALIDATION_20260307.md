# Qdrant Migration Write Validation 2026-03-07

## Scope
- Add Qdrant-native support for `POST /api/v1/vectors/migrate`.
- Preserve the existing in-memory migration path.
- Keep migration history, counters, and response contract unchanged.

## Changes
- `migrate_vectors()` now resolves Qdrant via `_get_qdrant_store_or_none()`.
- When Qdrant is enabled:
  - source vectors are read from `qdrant_store.get_vector()`;
  - migrated vectors are written back through `qdrant_store.register_vector()`;
  - dry-run mode does not write.
- Added Qdrant execution and dry-run tests in `tests/unit/test_vector_migrate_api.py`.

## Validation
```bash
pytest -q tests/unit/test_vector_migrate_api.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py
flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migrate_api.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py \
  --max-line-length=100
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migrate_api.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py
```

## Results
- `pytest`: `34 passed`
- `flake8`: passed
- `py_compile`: passed

## Notes
- Qdrant path updates vectors via upsert semantics; delete/reinsert is not required.
- `_VECTOR_MIGRATION_HISTORY` remains shared and backend-agnostic.
