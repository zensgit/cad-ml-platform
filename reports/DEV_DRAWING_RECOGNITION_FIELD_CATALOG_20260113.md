# DEV_DRAWING_RECOGNITION_FIELD_CATALOG_20260113

## Scope
Validate drawing recognition schema enhancements: title block map in response and field catalog endpoint.

## Changes
- Added `title_block` map to `/api/v1/drawing/recognize` response.
- Added `GET /api/v1/drawing/fields` to expose field keys and labels.

## Tests
Command:
```
pytest tests/ocr/test_drawing_endpoint.py tests/test_routes_smoke.py -v
```
Result:
- `tests/ocr/test_drawing_endpoint.py::test_drawing_recognize_smoke` passed
- `tests/ocr/test_drawing_endpoint.py::test_drawing_fields_catalog` passed
- `tests/test_routes_smoke.py::test_vision_health_route` passed
- `tests/test_routes_smoke.py::test_ocr_extract_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_recognize_route_exists` passed
- `tests/test_routes_smoke.py::test_drawing_fields_route_exists` passed

## Notes
- Field catalog relies on `FIELD_LABELS` as the source of truth.
