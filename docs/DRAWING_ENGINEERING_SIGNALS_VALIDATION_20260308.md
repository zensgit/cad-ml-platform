# Drawing Engineering Signals Validation 2026-03-08

## Goal

Strengthen `/api/v1/drawing` so it exposes engineering-facing structure instead of only
title block fields and raw dimensions/symbols. This is part of the benchmark-surpass track
 against drawing intelligence products that already expose manufacturing-relevant signals.

## Design

The OCR pipeline already produces `process_requirements` inside `OcrResult`, but
`src/api/v1/drawing.py` did not return it. This change promotes existing structured OCR
 output into the public response and adds two lightweight summaries:

1. `field_coverage`
   - recognized field count
   - total tracked fields
   - coverage ratio
   - recognized keys
   - missing keys

2. `engineering_signals`
   - dimension and symbol counts
   - detected symbol types
   - GD&T presence
   - surface-finish presence
   - process requirement counts
   - materials detected
   - standards candidates

`standards_candidates` is derived from:
- explicit `surface_treatments[].standard`
- regex extraction from `general_notes` / `raw_text`

This keeps the implementation low-risk:
- no OCR provider changes
- no new route
- no new parsing dependency
- only promotes already available structured data

## Changed Files

- `src/api/v1/drawing.py`
- `tests/ocr/test_drawing_endpoint.py`
- `config/openapi_schema_snapshot.json`

## Validation

### Static checks

```bash
python3 -m py_compile src/api/v1/drawing.py tests/ocr/test_drawing_endpoint.py
flake8 src/api/v1/drawing.py tests/ocr/test_drawing_endpoint.py --max-line-length=100
```

### Targeted tests

```bash
pytest -q tests/ocr/test_drawing_endpoint.py tests/ocr/test_process_parser.py
```

Result:
- `32 passed`

### OpenAPI contract

```bash
python3 scripts/ci/generate_openapi_schema_snapshot.py --output config/openapi_schema_snapshot.json
pytest -q tests/contract/test_openapi_operation_ids.py \
  tests/contract/test_openapi_schema_snapshot.py \
  tests/unit/test_api_route_uniqueness.py
make validate-openapi
```

Result:
- contract tests `5 passed`
- `make validate-openapi` passed

## Outcome

`/api/v1/drawing/recognize` and `/api/v1/drawing/recognize-base64` now return:

- `process_requirements`
- `field_coverage`
- `engineering_signals`

This improves product-facing drawing understanding without changing provider internals and
gives downstream review/UI/reporting layers direct access to manufacturing hints.

## Limitations

- `standards_candidates` is regex-based and intentionally conservative
- `materials_detected` currently reflects title-block material only
- no new provenance bbox layer was added in this change
