# DEV_OCR_PROVIDER_HEALTH_20260113

## Scope
Validate provider catalog and health endpoints for OCR and drawing recognition.

## Changes
- Added `/api/v1/ocr/providers` and `/api/v1/ocr/health`.
- Added `/api/v1/drawing/providers` and `/api/v1/drawing/health`.

## Tests
Command:
```
pytest tests/test_routes_smoke.py -v
```
Result:
- `tests/test_routes_smoke.py::test_vision_health_route` passed
- `tests/test_routes_smoke.py::test_ocr_extract_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_recognize_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_fields_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_recognize_base64_route_exists` passed
- `tests/test_routes_smoke.py::test_ocr_providers_route_exists` passed
- `tests/test_routes_smoke.py::test_ocr_health_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_providers_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_health_route_exists` passed
