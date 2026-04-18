# Phase 3 Qdrant Store Helper Extraction Verification

## Implemented
- added shared helper: `src/core/qdrant_store_helper.py`
- updated these route modules to reuse the shared resolver through local alias imports:
  - `src/api/v1/analyze.py`
  - `src/api/v1/compare.py`
  - `src/api/v1/features.py`
  - `src/api/v1/maintenance.py`
  - `src/api/v1/vectors.py`
  - `src/api/v1/vectors_stats.py`
- added helper smoke coverage: `tests/unit/test_qdrant_store_helper.py`

## Validation
- `python3 -m py_compile src/core/qdrant_store_helper.py src/api/v1/analyze.py src/api/v1/compare.py src/api/v1/features.py src/api/v1/maintenance.py src/api/v1/vectors.py src/api/v1/vectors_stats.py tests/unit/test_qdrant_store_helper.py`
- `.venv311/bin/flake8 src/core/qdrant_store_helper.py src/api/v1/analyze.py src/api/v1/compare.py src/api/v1/features.py src/api/v1/maintenance.py src/api/v1/vectors.py src/api/v1/vectors_stats.py tests/unit/test_qdrant_store_helper.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_qdrant_store_helper.py tests/unit/test_compare_endpoint.py tests/unit/test_features_diff_endpoint.py tests/unit/test_vector_stats.py tests/unit/test_vector_distribution_endpoint.py tests/unit/test_maintenance_endpoint_coverage.py tests/unit/test_vectors_module_endpoints.py tests/unit/test_similarity_topk.py tests/unit/test_similarity_error_codes.py`

## Result
- shared Qdrant resolver extraction passed static checks
- helper test and route-level qdrant alias regressions passed
- duplicated resolver logic is now centralized without breaking existing patch targets
