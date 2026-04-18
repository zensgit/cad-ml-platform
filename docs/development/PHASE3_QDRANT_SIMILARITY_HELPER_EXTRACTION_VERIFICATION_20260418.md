# Phase 3 Qdrant Similarity Helper Extraction Verification

## Implemented
- added shared helper: `src/core/qdrant_similarity_helper.py`
- updated `src/api/v1/analyze.py` to pass `compute_qdrant_cosine_similarity` into `run_vector_pipeline`
- added helper smoke coverage: `tests/unit/test_qdrant_similarity_helper.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile src/core/qdrant_similarity_helper.py src/api/v1/analyze.py tests/unit/test_qdrant_similarity_helper.py tests/integration/test_analyze_vector_pipeline.py`
- `.venv311/bin/flake8 src/core/qdrant_similarity_helper.py src/api/v1/analyze.py tests/unit/test_qdrant_similarity_helper.py tests/integration/test_analyze_vector_pipeline.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_qdrant_similarity_helper.py tests/unit/test_vector_pipeline.py tests/integration/test_analyze_vector_pipeline.py`

## Result
- Qdrant cosine similarity logic is now centralized in a shared helper
- `analyze.py` keeps the same vector pipeline call contract while dropping the local helper body
- helper coverage and analyze/vector pipeline regressions passed
- targeted pytest result: `10 passed, 7 warnings`
