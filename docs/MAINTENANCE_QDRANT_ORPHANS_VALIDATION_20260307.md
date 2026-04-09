# Maintenance Qdrant Orphans Validation 2026-03-07

## Scope
- Route `DELETE /api/v1/maintenance/orphans`
- Qdrant backend compatibility for orphan-vector cleanup

## Changes
- Added Qdrant vector ID scanning for orphan detection.
- Added Qdrant delete path for orphan cleanup.
- Preserved existing Redis/cache lookup logic and legacy in-memory cleanup path.

## Validation Commands
```bash
python3 -m py_compile \
  src/api/v1/maintenance.py \
  tests/unit/test_maintenance_endpoint_coverage.py

flake8 \
  src/api/v1/maintenance.py \
  tests/unit/test_maintenance_endpoint_coverage.py \
  --max-line-length=100

pytest -q tests/unit/test_maintenance_endpoint_coverage.py -k "cleanup_orphans_qdrant"
```

## Coverage
- Qdrant path when cache client is unavailable
- Qdrant path when one vector still has a live cache entry
- Legacy in-memory orphan cleanup remains unchanged
