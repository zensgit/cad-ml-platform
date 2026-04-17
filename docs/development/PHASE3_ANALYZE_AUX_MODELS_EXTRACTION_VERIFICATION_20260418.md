# Phase 3 Analyze Auxiliary Models Extraction Verification

## Implemented
- added shared schema module: `src/api/v1/analyze_aux_models.py`
- updated `src/api/v1/analyze.py` to import the auxiliary/legacy endpoint models
- updated `src/api/v1/process.py` to reuse the shared `ProcessRulesAuditResponse`
- added smoke coverage: `tests/unit/test_analyze_aux_models.py`

## Validation
- `python3 -m py_compile src/api/v1/analyze_aux_models.py src/api/v1/analyze.py src/api/v1/process.py tests/unit/test_analyze_aux_models.py`
- `.venv311/bin/flake8 src/api/v1/analyze_aux_models.py src/api/v1/analyze.py src/api/v1/process.py tests/unit/test_analyze_aux_models.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analyze_aux_models.py tests/unit/test_process_api_coverage.py tests/unit/test_deprecated_endpoints_410.py tests/unit/test_deprecated_health_endpoints.py tests/unit/test_deprecated_vector_endpoints.py tests/unit/test_feature_cache.py`

## Result
- static validation passed
- process audit and deprecated endpoint regressions passed
- analyze route layer lost a large block of auxiliary schema definitions without changing route behavior
