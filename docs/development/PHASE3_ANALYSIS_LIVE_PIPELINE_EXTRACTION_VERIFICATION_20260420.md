# Phase 3 Analysis Live Pipeline Extraction Verification

## Summary
- The main `analyze_cad_file` orchestration body was moved into `src/core/analysis_live_pipeline.py`.
- `analyze.py` now keeps the route signature and exception handling, then delegates the live flow to the shared helper.
- Existing monkeypatch points in `src.api.v1.analyze` remain intact because the route still passes module-level helpers into the extracted pipeline.

## Files
- `src/core/analysis_live_pipeline.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_live_pipeline.py`
- `tests/integration/test_analyze_live_pipeline.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No request/response contract change for `POST /api/v1/analyze/`
- Existing analyze monkeypatch coverage keeps working against `src.api.v1.analyze`
- The shared helper owns the live orchestration body and `analyze.py` becomes thinner
