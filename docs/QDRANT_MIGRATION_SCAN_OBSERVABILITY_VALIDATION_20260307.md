# Qdrant Migration Scan Observability Validation 2026-03-07

## Scope
- Expose whether Qdrant version-distribution results are complete or scan-limited.
- Cover `status`, `summary`, and `trends` read endpoints.

## Changes
- Added `_resolve_vector_migration_scan_limit()`.
- Extended responses with scan observability:
  - `scan_limit`
  - `distribution_complete`
- Extended where relevant:
  - `status`: `current_total_vectors`, `scanned_vectors`
  - `summary`: `current_total_vectors`, `scanned_vectors`
  - `trends`: `current_total_vectors`, `scanned_vectors`
- Added partial-scan Qdrant coverage in `tests/unit/test_migration_preview_trends.py`.

## Validation
```bash
pytest -q tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py
flake8 src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py \
  --max-line-length=100
python3 -m py_compile src/api/v1/vectors.py \
  tests/unit/test_vector_migration_status.py \
  tests/unit/test_migration_preview_trends.py \
  tests/unit/test_vector_migrate_v4.py \
  tests/unit/test_vector_migrate_downgrade_chain.py
```

## Results
- `pytest`: `46 passed`
- `flake8`: passed
- `py_compile`: passed

## Notes
- This is an observability-only change.
- Partial distributions are now explicit when `VECTOR_MIGRATION_SCAN_LIMIT` truncates Qdrant scans.
