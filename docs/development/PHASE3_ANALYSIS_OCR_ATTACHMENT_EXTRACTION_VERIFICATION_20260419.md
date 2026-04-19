# Phase 3 Analyze OCR Attachment Extraction Verification

## Summary
- The `analyze.py` wrapper around `run_analysis_ocr_pipeline` was moved into `src/core/analysis_ocr_attachment.py`.
- `analyze.py` now delegates OCR payload write-back while still passing the route-local OCR pipeline alias.

## Files
- `src/core/analysis_ocr_attachment.py`
- `src/api/v1/analyze.py`
- `tests/unit/test_analysis_ocr_attachment.py`

## Validation
- `PYTHONPYCACHEPREFIX=/tmp/cad_ml_platform_pycache python3 -m py_compile ...`
- `.venv311/bin/flake8 ...`
- `.venv311/bin/python -m pytest -q ...`

## Expected Outcome
- No route contract change
- Existing integration monkeypatch coverage continues to pass
- `None` OCR payload still leaves `results` untouched
