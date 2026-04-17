# Phase 3 Analysis OCR Pipeline Extraction Development Plan

## Goal

Extract the optional OCR enrichment block from `src/api/v1/analyze.py` into a
shared helper while preserving:

- `enable_ocr` gate behavior
- provider registry bootstrap behavior
- provider registration for `paddle` and `deepseek_hf`
- `preview_image_bytes`-based no-op behavior
- OCR response payload contract written to `results["ocr"]`

## Scope

### In

- add `src/core/ocr/analysis_ocr_pipeline.py`
- move OCR manager wiring and provider registration
- move preview-image guard and OCR result serialization
- keep `analyze.py` as a thin caller that writes `results["ocr"]`

### Out

- OCR provider implementation logic
- OCR manager fallback strategy
- non-OCR analyze flows
- route path or response schema changes

## Design

Create `run_analysis_ocr_pipeline(...)` with:

- `enable_ocr`
- `ocr_provider_strategy`
- `unified_data`
- injectable bootstrap function, provider getter, and manager factory for tests

Return:

- `None` when OCR is disabled
- `{"status": "no_preview_image"}` when no preview bytes exist
- serialized OCR payload when OCR extraction runs

`analyze.py` keeps:

- one helper call
- `results["ocr"] = payload` when helper returns non-`None`

## Risk Controls

- preserve default `confidence_fallback=0.85`
- keep provider names and serialized field names unchanged
- validate route-level use of the shared helper via integration monkeypatch

## Validation Plan

1. `python3 -m py_compile src/core/ocr/analysis_ocr_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py`
2. `.venv311/bin/flake8 src/core/ocr/analysis_ocr_pipeline.py src/api/v1/analyze.py tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py`
3. `.venv311/bin/python -m pytest -q tests/unit/test_analysis_ocr_pipeline.py tests/integration/test_analyze_ocr_pipeline.py tests/unit/test_analysis_result_envelope.py tests/integration/test_analyze_result_envelope.py tests/unit/test_analysis_preflight.py tests/unit/test_document_pipeline.py tests/test_api_integration.py`
