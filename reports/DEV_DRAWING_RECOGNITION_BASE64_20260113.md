# DEV_DRAWING_RECOGNITION_BASE64_20260113

## Scope
Validate base64 input support for drawing recognition and updated input validation helper.

## Changes
- Added `/api/v1/drawing/recognize-base64` for JSON base64 payloads.
- Added `validate_bytes` to reuse OCR input validation for raw byte payloads.

## Tests
Command:
```
pytest tests/ocr/test_drawing_endpoint.py tests/test_routes_smoke.py -v
```
Result:
- `tests/ocr/test_drawing_endpoint.py::test_drawing_recognize_smoke` passed
- `tests/ocr/test_drawing_endpoint.py::test_drawing_fields_catalog` passed
- `tests/ocr/test_drawing_endpoint.py::test_drawing_recognize_base64_smoke` passed
- `tests/test_routes_smoke.py::test_vision_health_route` passed
- `tests/test_routes_smoke.py::test_ocr_extract_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_recognize_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_fields_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_recognize_base64_route_exists` passed
