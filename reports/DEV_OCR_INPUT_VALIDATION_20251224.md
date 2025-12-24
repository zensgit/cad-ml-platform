# DEV_OCR_INPUT_VALIDATION_20251224

## Scope
- Implement OCR input validation for MIME, file size, PDF page limits, and PDF forbidden tokens.
- Enable OCR rejection tests previously skipped.

## Changes
- `src/security/input_validator.py`
  - Added OCR-specific MIME resolution and size limits.
  - Added PDF page counting and forbidden token checks.
- `tests/test_ocr_invalid_mime.py`
  - Removed skip to exercise invalid MIME handling.
- `tests/test_ocr_pdf_rejections.py`
  - Removed skips for PDF page limit and forbidden token checks.

## Validation
- Command: `.venv/bin/python -m pytest tests/test_ocr_invalid_mime.py tests/test_ocr_pdf_rejections.py -v`
  - Result: 3 passed.

## Notes
- Tests executed in project venv to ensure OCR dependencies are available.
