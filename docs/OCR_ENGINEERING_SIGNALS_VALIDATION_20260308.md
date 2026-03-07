# OCR Engineering Signals Validation 2026-03-08

## Goal

Bring the OCR API up to the same product surface as the drawing endpoint by
exposing title-block coverage and engineering-oriented summaries directly from
`/api/v1/ocr`.

## Design

This change introduces a shared helper module:

- `src/core/ocr/response_summary.py`

Shared builders:
- `build_field_coverage()`
- `extract_standard_candidates()`
- `build_engineering_signals()`

Consumers:
- `src/api/v1/ocr.py`
- `src/api/v1/drawing.py`
- `src/core/vision/manager.py`

New OCR response fields:
- `field_coverage`
- `engineering_signals`

The same helper is reused by drawing and embedded vision OCR results so all
three surfaces report the same summary contract.

## Changed Files

- `src/core/ocr/response_summary.py`
- `src/api/v1/ocr.py`
- `src/api/v1/drawing.py`
- `src/core/vision/base.py`
- `src/core/vision/manager.py`
- `tests/unit/test_ocr_endpoint_coverage.py`
- `tests/vision/test_vision_ocr_integration.py`
- `config/openapi_schema_snapshot.json`

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/core/ocr/response_summary.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py

flake8 \
  src/core/ocr/response_summary.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py \
  --max-line-length=100
```

### Functional tests

```bash
pytest -q \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py \
  tests/ocr/test_drawing_endpoint.py
```

Result:
- `30 passed`

### OpenAPI contract

```bash
python3 scripts/ci/generate_openapi_schema_snapshot.py \
  --output config/openapi_schema_snapshot.json
pytest -q \
  tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/unit/test_api_route_uniqueness.py
```

Result:
- `5 passed`

## Outcome

OCR consumers now receive:
- `identifiers[]`
- `field_evidence{}`
- `field_coverage{}`
- `engineering_signals{}`

This closes a major product gap between low-level OCR extraction and the richer
drawing-recognition surface while keeping summary logic centralized.
