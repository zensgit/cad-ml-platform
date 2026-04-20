# Phase 3 Analyze Batch Router Extraction Verification

## Summary
- The live `/api/v1/analyze/batch` and `/api/v1/analyze/batch-classify` routes were moved into `src/api/v1/analyze_batch_router.py`.
- `analyze.py` now includes the split batch router and no longer owns these two live endpoint implementations.

## Files
- `src/api/v1/analyze_batch_router.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analyze_batch_router.py`
- `tests/integration/test_analyze_batch_pipeline.py`
- `tests/integration/test_analyze_batch_classify_pipeline.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No path or response contract change
- Existing batch integration monkeypatch coverage continues to pass against the new router module
- Route ownership is explicit and covered
