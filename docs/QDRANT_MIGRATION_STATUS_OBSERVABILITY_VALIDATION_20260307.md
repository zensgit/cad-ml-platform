# Qdrant Migration Status Observability Validation 2026-03-07

## Scope
- Extend `GET /api/v1/vectors/migrate/status` with backend-aware scan observability.
- Preserve existing migration history and feature version output.

## Changes
- `VectorMigrationStatusResponse` now includes:
  - `backend`
  - `current_total_vectors`
  - `scanned_vectors`
- `migrate_status()` now:
  - reports `backend=qdrant` and uses Qdrant scan statistics when Qdrant is enabled;
  - reports `backend=memory` and exact in-memory totals otherwise.

## Validation
```bash
pytest -q tests/unit/test_vector_migration_status.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py
flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py \
  --max-line-length=100
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py
```

## Results
- `pytest`: `21 passed`
- `flake8`: passed
- `py_compile`: passed

## Notes
- `scanned_vectors` may be lower than `current_total_vectors` in future if scan limits are applied.
- This is an observability-only change; migration behavior is unchanged.
