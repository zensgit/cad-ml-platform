# OCR Identifiers Evidence Validation 2026-03-08

## Goal

Add `identifiers[]` to OCR and drawing responses so downstream consumers receive
structured title-block identifiers with evidence, not only flattened field values.

This change is benchmark-driven: competitor systems commonly expose identifier
objects with provenance. The implementation here keeps scope low-risk by
focusing on title-block style identifiers that are already parsed by the OCR
 stack.

## Design

### New core model

`src/core/ocr/base.py`

- Added `IdentifierInfo`
- Added `OcrResult.identifiers`

Each identifier now carries:
- `identifier_type`
- `label`
- `value`
- `normalized_value`
- `source_text`
- `bbox`
- `confidence`
- `source`

### New parser

`src/core/ocr/parsing/identifier_parser.py`

Capabilities:
- same-line label/value extraction from OCR lines
- split-line caption/value extraction
- fallback extraction from full OCR text
- de-duplication
- bbox/confidence propagation

Sources used:
- `ocr_line`
- `regex_text`
- `provider_json`

### Provider wiring

- `src/core/ocr/providers/paddle.py`
  - builds identifiers from OCR lines + parsed title-block values
- `src/core/ocr/providers/deepseek_hf.py`
  - builds identifiers from provider JSON title-block values
  - attaches OCR-line evidence when paddle alignment lines are available

### API / integration surfaces

- `src/api/v1/ocr.py`
  - exposes `identifiers`
- `src/api/v1/drawing.py`
  - exposes `identifiers`
- `src/core/vision/base.py`
  - extends embedded OCR result with `identifiers`
- `src/core/vision/manager.py`
  - passes identifiers through OCR-to-vision conversion

## Changed Files

- `src/core/ocr/base.py`
- `src/core/ocr/parsing/identifier_parser.py`
- `src/core/ocr/providers/paddle.py`
- `src/core/ocr/providers/deepseek_hf.py`
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
  src/core/ocr/base.py \
  src/core/ocr/parsing/identifier_parser.py \
  src/core/ocr/providers/paddle.py \
  src/core/ocr/providers/deepseek_hf.py \
  src/api/v1/ocr.py \
  src/api/v1/drawing.py \
  src/core/vision/base.py \
  src/core/vision/manager.py

flake8 \
  src/core/ocr/base.py \
  src/core/ocr/parsing/identifier_parser.py \
  src/core/ocr/providers/paddle.py \
  src/core/ocr/providers/deepseek_hf.py \
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
- `33 passed`

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

OCR, drawing, and vision responses now expose structured identifiers with basic
evidence. This gives downstream review/reporting layers a stable substrate for
future provenance UI and benchmark-style metadata rendering without forcing a
large OCR provider rewrite.

## Limitations

- Current identifier extraction focuses on title-block style identifiers
- `provider_json` evidence is shallow; it does not yet carry provider-native
  reference ids
- bbox evidence depends on available OCR lines / paddle alignment
