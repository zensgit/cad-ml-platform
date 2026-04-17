# Phase 3 Analyze Legacy Redirects Extraction Verification

## Implemented
- added shared helper: `src/core/legacy_redirect_pipeline.py`
- updated the deprecated analyze auxiliary endpoints to delegate to the shared helper
- added regression coverage:
  - `tests/unit/test_legacy_redirect_pipeline.py`
  - `tests/integration/test_analyze_legacy_redirect_pipeline.py`

## Validation
- `python3 -m py_compile src/core/legacy_redirect_pipeline.py src/api/v1/analyze.py tests/unit/test_legacy_redirect_pipeline.py tests/integration/test_analyze_legacy_redirect_pipeline.py`
- `.venv311/bin/flake8 src/core/legacy_redirect_pipeline.py src/api/v1/analyze.py tests/unit/test_legacy_redirect_pipeline.py tests/integration/test_analyze_legacy_redirect_pipeline.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_legacy_redirect_pipeline.py tests/integration/test_analyze_legacy_redirect_pipeline.py tests/unit/test_deprecated_endpoints_410.py tests/unit/test_deprecated_health_endpoints.py tests/unit/test_deprecated_vector_endpoints.py tests/unit/test_feature_cache.py`

## Result
- static validation passed
- deprecated endpoint regression suite passed
- analyze route layer lost duplicate 410 redirect boilerplate without changing endpoint behavior
