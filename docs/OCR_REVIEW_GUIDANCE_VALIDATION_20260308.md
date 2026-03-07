# OCR Review Guidance Validation 20260308

## Goal
- Upgrade `review_hints` from a passive readiness payload into actionable guidance.
- Keep the change low-conflict by limiting it to OCR/drawing response surfaces.

## Design
- Added stable action-oriented fields to `review_hints`:
  - `primary_gap`
  - `review_priority`
  - `automation_ready`
  - `recommended_actions`
- Decision rules:
  - missing critical title-block fields or low readiness => `review_priority=high`
  - any remaining review reason => `review_priority=medium`
  - no review reason and high readiness => `automation_ready=true`
- `recommended_actions` are machine-readable strings suitable for downstream review queueing.

## Files
- `src/core/ocr/response_summary.py`
- `src/core/vision/base.py`
- `tests/ocr/test_response_summary.py`
- `tests/ocr/test_drawing_endpoint.py`
- `tests/unit/test_ocr_endpoint_coverage.py`
- `tests/vision/test_vision_ocr_integration.py`

## Validation
```bash
python3 -m py_compile \
  src/core/ocr/response_summary.py \
  src/core/vision/base.py \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py

flake8 \
  src/core/ocr/response_summary.py \
  src/core/vision/base.py \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py \
  --max-line-length=100

pytest -q \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py
```

## Result
- Local validation passed.
- OCR and drawing responses now expose actionable review guidance without changing route shape.
