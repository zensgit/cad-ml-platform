# DEV_DRAWING_RECOGNITION_VALIDATION_20260112

## Scope
Run the drawing recognition tests and route smoke validation.

## Test Runs
1. Title block parsing tests:

```bash
pytest tests/ocr/test_title_block_parser.py -v
```

Result: **3 passed**

2. Drawing endpoint smoke test (dummy manager):

```bash
pytest tests/ocr/test_drawing_endpoint.py -v
```

Result: **1 passed**

3. Route smoke test (main app routing):

```bash
pytest tests/test_routes_smoke.py -k drawing_recognize_route_exists -v
```

Result: **1 passed** (2 deselected)

## Notes
- The route smoke test uses the main app; OCR providers run in stub mode for fake image data.
