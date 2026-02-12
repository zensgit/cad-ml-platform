# DEV_DESIGN_STANDARDS_API_20260213

## Goal
Expose deterministic mechanical design-guideline knowledge (surface finish, general tolerances, preferred diameters) through a small, stable HTTP API surface and register it in the provider framework for health/readiness visibility.

## Changes
- API router:
  - Added `src/api/v1/design_standards.py`
  - Registered in `src/api/__init__.py` with prefix `/api/v1/design-standards`
  - Endpoints:
    - `GET /api/v1/design-standards/status`
    - `GET /api/v1/design-standards/surface-finish/grade`
    - `GET /api/v1/design-standards/surface-finish/application`
    - `GET /api/v1/design-standards/surface-finish/suggest`
    - `GET /api/v1/design-standards/general-tolerances/linear`
    - `GET /api/v1/design-standards/general-tolerances/angular`
    - `GET /api/v1/design-standards/general-tolerances/table`
    - `GET /api/v1/design-standards/preferred-diameter`
- Provider framework:
  - Added `knowledge/design_standards` provider adapter in `src/core/providers/knowledge.py`
  - Extended provider-bridge tests to cover the new knowledge provider
- OpenAPI:
  - Updated schema snapshot baseline: `config/openapi_schema_snapshot.json`
- Tests:
  - Added `tests/integration/test_design_standards_api.py`

## Validation
- `.venv/bin/python -m pytest tests/integration/test_design_standards_api.py -v` (`6 passed`)
- `.venv/bin/python -m pytest tests/unit/test_provider_knowledge_providers.py tests/unit/test_provider_framework_knowledge_bridge.py tests/unit/test_knowledge_provider_coverage.py -v` (`24 passed`)
- `make openapi-snapshot-update`
- `make validate-core-fast` (passed)

## Notes
- Knowledge sources are the built-in tables and helpers under `src/core/knowledge/design_standards` (surface finish, ISO 2768-1 style general tolerances, and preferred diameters/chamfers/fillets).
- The provider registration is intentionally lightweight (health probes + summary counts) and does not replace the dedicated query endpoints.

