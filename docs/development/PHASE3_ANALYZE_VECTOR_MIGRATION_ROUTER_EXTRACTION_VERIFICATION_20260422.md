# Phase 3 Analyze Vector Migration Router Extraction Verification

## Local Validation

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/api/v1/analyze_vector_migration_router.py src/api/v1/analyze.py src/api/v1/analyze_vector_compat.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_analyze_vector_compat_router.py tests/unit/test_analyze_vector_migration_router.py`
- `.venv311/bin/flake8 src/api/v1/analyze_vector_migration_router.py src/api/v1/analyze.py src/api/v1/analyze_vector_compat.py tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_analyze_vector_compat_router.py tests/unit/test_analyze_vector_migration_router.py`
- `.venv311/bin/python -m pytest -q tests/integration/test_analyze_legacy_vector_migration_pipeline.py tests/unit/test_analyze_vector_compat_router.py tests/unit/test_analyze_vector_migration_router.py`

## Expected Outcome

- `/api/v1/analyze/vectors/migrate*` routes are owned by the new split router
- Existing integration monkeypatch tests still pass against the new module
- `analyze_vector_compat.py` keeps only the remaining vector compat update route
