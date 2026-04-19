# Phase 3 Analyze Vector Compat Router Extraction Verification

## Implemented
- added split router module: `src/api/v1/analyze_vector_compat.py`
- updated `src/api/v1/analyze.py` to include the split vector compat router instead of hosting those routes inline
- updated integration patch coverage to hook `src.api.v1.analyze_vector_compat`
- added route ownership smoke coverage: `tests/unit/test_analyze_vector_compat_router.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/api/v1/analyze.py src/api/v1/analyze_vector_compat.py tests/unit/test_analyze_vector_compat_router.py tests/integration/test_analyze_legacy_admin_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_faiss_rebuild.py tests/unit/test_vector_migration_status.py`
- `.venv311/bin/flake8 src/api/v1/analyze.py src/api/v1/analyze_vector_compat.py tests/unit/test_analyze_vector_compat_router.py tests/integration/test_analyze_legacy_admin_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_faiss_rebuild.py tests/unit/test_vector_migration_status.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analyze_vector_compat_router.py tests/integration/test_analyze_legacy_admin_pipeline.py tests/integration/test_analyze_vector_update_pipeline.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_faiss_rebuild.py tests/unit/test_vector_migration_status.py`

## Result
- legacy analyze vector admin/update/migration endpoints are split out of `src/api/v1/analyze.py` without changing behavior
- targeted delegation and route ownership regressions passed
- targeted pytest result: `11 passed, 7 warnings`
