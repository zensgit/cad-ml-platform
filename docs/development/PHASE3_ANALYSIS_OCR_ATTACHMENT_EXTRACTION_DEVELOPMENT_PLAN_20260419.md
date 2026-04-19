# Phase 3 Analyze OCR Attachment Extraction Development Plan

## Goal
- Extract the `analyze.py` OCR post-processing wrapper into a shared helper.
- Keep route behavior and analyze-level OCR pipeline patch points unchanged.

## Scope
- Add `src/core/analysis_ocr_attachment.py`
- Update `src/api/v1/analyze.py` to delegate OCR payload attachment
- Add focused unit coverage for the helper

## Constraints
- Preserve `src.api.v1.analyze.run_analysis_ocr_pipeline`
- Do not move the real OCR strategy/manager logic
- Keep `None` payload behavior unchanged

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper and existing analyze OCR integration coverage
