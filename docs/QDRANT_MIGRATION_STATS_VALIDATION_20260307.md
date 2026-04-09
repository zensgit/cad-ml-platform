# Qdrant Migration Stats Validation 2026-03-07

## Scope
- Add Qdrant-backed version distribution support to:
  - `GET /api/v1/vectors/migrate/status`
  - `GET /api/v1/vectors/migrate/trends`
- Keep migration write path unchanged
- Keep migration summary/history endpoints unchanged

## Changes
- `src/api/v1/vectors.py`
  - add `_collect_qdrant_feature_versions()`
  - `migrate_status()` now reads feature version counts from Qdrant when
    `VECTOR_STORE_BACKEND=qdrant`
  - `migrate_trends()` now reads version distribution and v4 adoption rate from Qdrant
- tests
  - add Qdrant status coverage
  - add Qdrant trends coverage

## Validation
### Static
```bash
python3 -m py_compile \
  src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py

flake8 \
  src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  --max-line-length=100
```

### Tests
```bash
pytest -q \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  -k 'qdrant or status_flow or trends_v4_adoption_rate or trends_with_history or trends_empty_history'
```

## Results
- Targeted migration status/trends suite: `6 passed`
- No lint failures
- No compile failures

## Notes
- This is a read-only observability change.
- `migrate/preview` and `POST /migrate` still use existing in-memory vector source logic.
