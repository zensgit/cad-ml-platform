# Qdrant Migration Summary Validation 2026-03-07

## Scope
- Extend `GET /api/v1/vectors/migrate/summary` with backend-aware current version observability.
- Preserve existing migration aggregate counts and status list.

## Changes
- `VectorMigrationSummaryResponse` now includes:
  - `backend`
  - `current_version_distribution`
  - `current_total_vectors`
- `migrate_summary()` now:
  - reads current version distribution from Qdrant when `VECTOR_STORE_BACKEND=qdrant`;
  - falls back to in-memory `_VECTOR_STORE/_VECTOR_META` otherwise.

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
- This is an observability-only change; migration write behavior is unchanged.
- Qdrant summary fields intentionally mirror the semantics already used in `status` and `trends`.
