# Qdrant Migration Preview Validation 2026-03-07

## Scope
- Add Qdrant-native support for `GET /api/v1/vectors/migrate/preview`.
- Keep the in-memory preview path unchanged.
- Preserve preview response contract: `total_vectors`, `by_version`, `preview_items`, and dimension delta statistics.

## Changes
- Added `_collect_qdrant_preview_samples()` in `src/api/v1/vectors.py`.
- Updated `preview_migration()` to:
  - use Qdrant when `VECTOR_STORE_BACKEND=qdrant` and a managed store is available;
  - collect `by_version` from Qdrant metadata;
  - only materialize vectors for the requested preview window;
  - continue scanning metadata-only pages to finish version distribution.
- Added Qdrant tests in `tests/unit/test_migration_preview_trends.py`.

## Validation
```bash
pytest -q tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py
flake8 src/api/v1/vectors.py \
  tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py \
  --max-line-length=100
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_migration_preview_stats.py \
  tests/unit/test_migration_preview_response.py \
  tests/unit/test_migration_preview_trends.py
```

## Results
- `pytest`: `28 passed`
- `flake8`: passed
- `py_compile`: passed

## Notes
- Qdrant path is read-only preview support; actual migration write path is unchanged.
- Preview intentionally loads vectors only for the preview sample and counts the remaining version distribution through metadata pages.
