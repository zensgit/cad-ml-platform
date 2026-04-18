# Phase 3 Analyze Legacy Redirect Router Extraction Verification

## Implemented
- added split router module: `src/api/v1/analyze_legacy_redirects.py`
- updated `src/api/v1/analyze.py` to include the split router instead of hosting the redirect routes inline
- updated integration patch coverage to hook `src.api.v1.analyze_legacy_redirects.raise_legacy_redirect`
- added route ownership smoke coverage: `tests/unit/test_analyze_legacy_redirect_router.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/api/v1/analyze.py src/api/v1/analyze_legacy_redirects.py tests/unit/test_analyze_legacy_redirect_router.py tests/unit/test_deprecated_endpoints_410.py tests/unit/test_deprecated_health_endpoints.py tests/integration/test_analyze_legacy_redirect_pipeline.py tests/unit/test_api_route_uniqueness.py tests/unit/test_legacy_redirect_pipeline.py`
- `.venv311/bin/flake8 src/api/v1/analyze.py src/api/v1/analyze_legacy_redirects.py tests/unit/test_analyze_legacy_redirect_router.py tests/unit/test_deprecated_endpoints_410.py tests/unit/test_deprecated_health_endpoints.py tests/integration/test_analyze_legacy_redirect_pipeline.py tests/unit/test_api_route_uniqueness.py tests/unit/test_legacy_redirect_pipeline.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analyze_legacy_redirect_router.py tests/unit/test_deprecated_endpoints_410.py tests/unit/test_deprecated_health_endpoints.py tests/integration/test_analyze_legacy_redirect_pipeline.py tests/unit/test_api_route_uniqueness.py tests/unit/test_legacy_redirect_pipeline.py`

## Result
- legacy redirect route ownership is split out of `src/api/v1/analyze.py` without changing endpoint behavior
- deprecated `/api/v1/analyze/...` endpoints still return the same 410 structured migration errors
- targeted pytest result: `22 passed, 7 warnings`
