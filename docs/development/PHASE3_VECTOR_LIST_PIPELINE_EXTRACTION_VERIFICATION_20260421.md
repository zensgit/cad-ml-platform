# Phase 3 Vector List Pipeline Extraction Verification

## Summary
- The `/api/v1/vectors` listing orchestration body was moved into `src/core/vector_list_pipeline.py`.
- `vectors.py` now keeps the route signature and delegates to the shared helper.
- Existing module-level patch coverage remains valid because the route still passes `_get_qdrant_store_or_none`, `_resolve_list_source`, `_list_vectors_redis`, `_list_vectors_memory`, and `get_client` into the helper.

## Files
- `src/core/vector_list_pipeline.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_list_pipeline.py`
- `tests/unit/test_vectors_list_delegation.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No request/response contract change for `/api/v1/vectors`
- Existing qdrant, redis, and memory vector list regressions continue to pass
- `vectors.py` becomes thinner while preserving module-level monkeypatch compatibility
