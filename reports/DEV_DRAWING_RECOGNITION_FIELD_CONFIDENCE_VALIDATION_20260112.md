# DEV_DRAWING_RECOGNITION_FIELD_CONFIDENCE_VALIDATION_20260112

## Scope
Validate per-field confidence mapping and expanded title block aliases.

## Test Runs
1. Title block parser tests (including aliases):

```bash
pytest tests/ocr/test_title_block_parser.py -v
```

Result: **4 passed**

2. Drawing recognition endpoint smoke test (field confidence mapping):

```bash
pytest tests/ocr/test_drawing_endpoint.py -v
```

Result: **1 passed**

3. Route registration check:

```bash
pytest tests/test_routes_smoke.py -k drawing_recognize_route_exists -v
```

Result: **1 passed** (2 deselected)
