# Phase 3 Analyze Live Models Extraction Verification

## Implemented
- added shared live schema module: `src/api/v1/analyze_live_models.py`
- updated `src/api/v1/analyze.py` to import the main analyze, batch classify, and similarity models
- added smoke coverage: `tests/unit/test_analyze_live_models.py`

## Validation
- `python3 -m py_compile src/api/v1/analyze_live_models.py src/api/v1/analyze.py tests/unit/test_analyze_live_models.py`
- `.venv311/bin/flake8 src/api/v1/analyze_live_models.py src/api/v1/analyze.py tests/unit/test_analyze_live_models.py`
- `.venv311/bin/python -m pytest -q tests/unit/test_analyze_live_models.py tests/unit/test_analyze_history_sequence_resolution.py tests/unit/test_similarity_endpoint.py tests/unit/test_similarity_error_codes.py tests/unit/test_similarity_topk.py tests/unit/test_similarity_complexity_filter.py tests/unit/test_similarity_filters.py tests/unit/test_similarity_topk_pagination.py tests/unit/test_v16_classifier_endpoints.py tests/integration/test_analyze_batch_classify_pipeline.py`

## Result
- static validation passed
- history sequence, similarity, and batch classify regressions passed
- analyze route file lost the remaining live schema block without changing route behavior
