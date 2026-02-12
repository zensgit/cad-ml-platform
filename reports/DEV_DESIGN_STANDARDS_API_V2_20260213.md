# DEV_DESIGN_STANDARDS_API_V2_20260213

## Goal
Extend the Design Standards API with practical "design features" helpers for chamfers/fillets and add list endpoints for surface-finish grades and preferred diameters.

This is intended to support deterministic downstream use (rules/fusion) without requiring manual drawing review.

## Changes
- `src/api/v1/design_standards.py`
  - Added endpoints:
    - `GET /api/v1/design-standards/surface-finish/grades`
    - `GET /api/v1/design-standards/design-features/preferred-diameters`
    - `GET /api/v1/design-standards/design-features/chamfer`
    - `GET /api/v1/design-standards/design-features/fillet`
- `tests/integration/test_design_standards_api.py`
  - Added integration coverage for the new endpoints.
- `config/openapi_schema_snapshot.json`
  - Updated snapshot baseline after adding routes.

## Validation
- `.venv/bin/python -m pytest tests/integration/test_design_standards_api.py -v` (`10 passed`)
- `make openapi-snapshot-update`
- `make validate-core-fast` (passed)

## Notes
- The chamfer/fillet endpoints are "closest-standard" selectors backed by:
  - `src/core/knowledge/design_standards/design_features.py`
- The list endpoints are deterministic enumeration helpers:
  - Surface finish grades are enumerated from `SurfaceFinishGrade` (N1..N12) and `SURFACE_FINISH_TABLE`.
  - Preferred diameters range filter delegates to `list_preferred_diameters(...)`.

