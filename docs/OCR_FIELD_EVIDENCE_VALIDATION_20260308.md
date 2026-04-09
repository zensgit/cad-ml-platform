# OCR Field Evidence Validation 2026-03-08

## Goal

Build a direct per-field evidence surface on top of `identifiers[]` so clients do
not need to reconstruct title-block provenance themselves.

## Design

This change is intentionally additive and stacked on top of the
`feat/ocr-identifiers-evidence` branch.

New helper:
- `build_field_evidence(identifiers)` in
  `src/core/ocr/parsing/identifier_parser.py`

It converts the first identifier per field into a stable map:

```json
{
  "drawing_number": {
    "label": "Drawing Number",
    "value": "DWG-001",
    "normalized_value": "DWG-001",
    "source_text": "图号: DWG-001",
    "bbox": [10, 10, 50, 10],
    "confidence": 0.92,
    "source": "ocr_line"
  }
}
```

Exposed surfaces:
- `src/api/v1/ocr.py`
- `src/api/v1/drawing.py`
- `src/core/vision/base.py`
- `src/core/vision/manager.py`

## Changed Files

- `src/core/ocr/parsing/identifier_parser.py`
- `src/api/v1/ocr.py`
- `src/api/v1/drawing.py`
- `src/core/vision/base.py`
- `src/core/vision/manager.py`
- `tests/ocr/test_identifier_parser.py`
- `tests/ocr/test_drawing_endpoint.py`
- `tests/unit/test_ocr_endpoint_coverage.py`
- `tests/vision/test_vision_ocr_integration.py`
- `config/openapi_schema_snapshot.json`

## Validation

### Static checks

```bash
python3 -m py_compile \
  src/core/ocr/parsing/identifier_parser.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/ocr/test_identifier_parser.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py

flake8 \
  src/core/ocr/parsing/identifier_parser.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py \
  tests/ocr/test_identifier_parser.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py \
  --max-line-length=100
```

### Functional tests

```bash
pytest -q \
  tests/ocr/test_identifier_parser.py \
  tests/ocr/test_drawing_endpoint.py \
  tests/unit/test_ocr_endpoint_coverage.py \
  tests/vision/test_vision_ocr_integration.py
```

Result:
- `34 passed`

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

OCR, drawing, and embedded vision OCR payloads now provide both:
- `identifiers[]` for list-style benchmark compatibility
- `field_evidence{}` for direct keyed consumption

This reduces client-side glue code and makes provenance immediately available
for UI review and reporting surfaces.
