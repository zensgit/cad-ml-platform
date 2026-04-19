# Phase 3 Analyze Vector Attachment Extraction Verification

## Summary
- The `analyze.py` wrapper around `run_vector_pipeline` and similarity stage timing was moved into `src/core/analysis_vector_attachment.py`.
- `analyze.py` now delegates similarity write-back and stage timing while still passing the route-local vector pipeline alias.

## Files
- `src/core/analysis_vector_attachment.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_vector_attachment.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No route contract change
- Existing integration monkeypatch coverage continues to pass
- Similarity stage timing is still recorded only when similarity exists
