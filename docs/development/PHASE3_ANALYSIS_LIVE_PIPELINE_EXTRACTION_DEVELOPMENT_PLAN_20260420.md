# Phase 3 Analysis Live Pipeline Extraction Development Plan

## Goal
- Extract the live `analyze_cad_file` orchestration body out of `src/api/v1/analyze.py`.
- Keep request/response behavior unchanged while shrinking the main analyze route module.

## Scope
- Add `src/core/analysis_live_pipeline.py`
- Update `src/api/v1/analyze.py` to delegate the main analysis flow to the shared helper
- Add direct helper coverage
- Add a thin route delegation regression test

## Constraints
- Preserve the existing FastAPI route signature for `POST /api/v1/analyze/`
- Preserve all existing `src.api.v1.analyze.*` monkeypatch points for document/classification/quality/process/vector/OCR/finalize helpers
- Keep exception handling in `analyze.py`
- Keep changes limited to orchestration extraction only

## Validation Plan
- `py_compile` on touched files
- `flake8` on touched files
- Focused `pytest` for the new helper, route delegation, and existing analyze integration regressions
