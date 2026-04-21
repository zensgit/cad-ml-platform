# Phase 3 Vector Delete Pipeline Extraction Verification

## Summary
- The `/api/v1/vectors/delete` orchestration body was moved into `src/core/vector_delete_pipeline.py`.
- `vectors.py` now keeps the route signature and delegates to the shared helper.
- Existing module-level patch coverage remains valid because the route still passes `_get_qdrant_store_or_none` and `get_client` into the helper.

## Files
- `src/core/vector_delete_pipeline.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_delete_pipeline.py`
- `tests/unit/test_vectors_delete_delegation.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No request/response contract change for `/api/v1/vectors/delete`
- Existing qdrant and memory/faiss/redis vector delete regressions continue to pass
- `vectors.py` becomes thinner while preserving module-level monkeypatch compatibility
