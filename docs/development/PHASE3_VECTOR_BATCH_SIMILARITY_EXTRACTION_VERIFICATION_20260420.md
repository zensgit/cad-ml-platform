# Phase 3 Vector Batch Similarity Extraction Verification

## Summary
- The `/api/v1/vectors/similarity/batch` orchestration body was moved into `src/core/vector_batch_similarity.py`.
- `vectors.py` now keeps the route signature and delegates to the shared helper.
- Existing `src.api.v1.vectors._get_qdrant_store_or_none` patch coverage remains valid because the route still passes the module-level alias into the helper.

## Files
- `src/core/vector_batch_similarity.py`
- `src/api/v1/vectors.py`
- `tests/unit/test_vector_batch_similarity_pipeline.py`
- `tests/unit/test_vectors_batch_similarity_delegation.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No request/response contract change for `/api/v1/vectors/similarity/batch`
- Existing qdrant and fallback batch similarity regressions continue to pass
- `vectors.py` becomes thinner while preserving module-level monkeypatch compatibility
