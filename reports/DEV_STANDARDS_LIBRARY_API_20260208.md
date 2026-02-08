# DEV_STANDARDS_LIBRARY_API_20260208

## Goal
Expose a lightweight HTTP API for the existing standard-parts knowledge base (`src/core/knowledge/standards`) so non-assistant clients can query:

- Metric thread specs (ISO 261/262)
- Bearing specs (ISO 15)
- O-ring specs (ISO 3601)

## Implementation

### New Router
Added `src/api/v1/standards.py` and registered it in `src/api/__init__.py`:

- Base path: `/api/v1/standards`
- Auth: `X-API-Key` dependency (same as other v1 endpoints)

### Endpoints (MVP)
- `GET /api/v1/standards/status`
  - returns counts and example inputs (does not dump the full databases)
- `GET /api/v1/standards/thread?designation=M10`
  - thread geometry + tap drill suggestion
- `GET /api/v1/standards/bearing?designation=6205`
  - basic dimensions + load/speed/weight metadata
- `GET /api/v1/standards/bearing/by-bore?bore_mm=25`
  - search bearings by bore diameter
- `GET /api/v1/standards/oring?designation=20x3`
  - o-ring dimensions + tolerances + groove reference dimensions
- `GET /api/v1/standards/oring/by-id?inner_diameter_mm=20`
  - search o-rings by inner diameter

## Verification

### Tests
Added `tests/integration/test_standards_api.py` and ran:

```bash
.venv/bin/pytest -q tests/integration/test_standards_api.py
```

Result: passed.

### Example Calls

```bash
curl -sS "http://127.0.0.1:8000/api/v1/standards/thread?designation=M10" -H "X-API-Key: test"
curl -sS "http://127.0.0.1:8000/api/v1/standards/bearing?designation=6205" -H "X-API-Key: test"
curl -sS "http://127.0.0.1:8000/api/v1/standards/oring?designation=20x3" -H "X-API-Key: test"
```

