# OCR Review Hints Validation

Date: 2026-03-08
Branch: `feat/ocr-review-hints`
Status: validated locally

## Goal

Add stable `review_hints` outputs to OCR-, drawing-, and vision-embedded
responses so downstream review and benchmark workflows can reason about:

- missing critical title-block fields
- evidence completeness
- readiness score / band
- whether manual review is recommended

## Scope

Updated:

- `src/core/ocr/response_summary.py`
- `src/api/v1/ocr.py`
- `src/api/v1/drawing.py`
- `src/core/vision/base.py`
- `src/core/vision/manager.py`
- `tests/ocr/test_drawing_endpoint.py`
- `tests/unit/test_ocr_endpoint_coverage.py`
- `tests/vision/test_vision_ocr_integration.py`

Added:

- `tests/ocr/test_response_summary.py`

## Review Hints Contract

`review_hints` now contains:

- `critical_fields`
- `present_critical_fields`
- `missing_critical_fields`
- `has_identifiers`
- `has_dimensions`
- `has_symbols`
- `has_process_requirements`
- `has_standards_candidates`
- `review_recommended`
- `review_reasons`
- `readiness_score`
- `readiness_band`

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/core/ocr/response_summary.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py
```

```bash
flake8 \
  src/core/ocr/response_summary.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py \
  --max-line-length=100
```

### Targeted tests

```bash
pytest -q \
  tests/ocr/test_response_summary.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py
```

Result:

- `32 passed`

### OpenAPI contract

```bash
make validate-openapi
```

Result:

- `5 passed`

## Outcome

- Drawing and OCR surfaces now expose benchmark-oriented review readiness hints.
- Vision OCR passthrough keeps the same structure, so review logic can reuse one
  contract across OCR, drawing, and vision responses.
- The added helper tests lock the scoring and missing-field behavior in place.
