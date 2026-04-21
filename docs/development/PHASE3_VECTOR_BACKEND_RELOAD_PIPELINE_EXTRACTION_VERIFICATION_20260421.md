# Phase 3 Vector Backend Reload Pipeline Extraction Verification

## Local Validation

- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/vector_backend_reload_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_backend_reload_pipeline.py tests/unit/test_vectors_backend_reload_delegation.py`
- `.venv311/bin/flake8 src/core/vector_backend_reload_pipeline.py src/api/v1/vectors.py tests/unit/test_vector_backend_reload_pipeline.py tests/unit/test_vectors_backend_reload_delegation.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_vector_backend_reload_pipeline.py tests/unit/test_vectors_backend_reload_delegation.py tests/unit/test_vector_backend_reload_failure.py tests/unit/test_vectors_backend_reload_admin_token.py tests/unit/test_vector_store_reload.py`

## Expected Outcome

- Shared helper owns backend reload execution logic
- `vectors.py` remains a thin route wrapper
- Existing API behavior, metrics, env updates, and auth handling remain unchanged
