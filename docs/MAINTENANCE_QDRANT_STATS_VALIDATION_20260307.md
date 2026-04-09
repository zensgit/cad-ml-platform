# Maintenance Qdrant Stats Validation 2026-03-07

## Scope
- Route `GET /api/v1/maintenance/stats`
- Qdrant backend compatibility for vector-store statistics

## Changes
- Added a local Qdrant store resolver in `src/api/v1/maintenance.py`.
- `maintenance/stats` now reads vector counts from Qdrant when
  `VECTOR_STORE_BACKEND=qdrant`.
- Added `vector_store.backend` to the response so the caller can see whether
  stats came from `memory` or `qdrant`.

## Validation Commands
```bash
python3 -m py_compile \
  src/api/v1/maintenance.py \
  tests/unit/test_maintenance_endpoint_coverage.py

flake8 \
  src/api/v1/maintenance.py \
  tests/unit/test_maintenance_endpoint_coverage.py \
  --max-line-length=100

pytest -q tests/unit/test_maintenance_endpoint_coverage.py -k "get_stats"
```

## Coverage
- Legacy memory stats path
- Qdrant stats path
- Cache unavailable path
- Vector store exception path
