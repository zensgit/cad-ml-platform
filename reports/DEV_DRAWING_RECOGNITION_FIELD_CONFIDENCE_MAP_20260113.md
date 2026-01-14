# DEV_DRAWING_RECOGNITION_FIELD_CONFIDENCE_MAP_20260113

## Scope
Validate the new `field_confidence` map in drawing recognition responses.

## Changes
- Added `field_confidence` map to `/api/v1/drawing/recognize` and `/api/v1/drawing/recognize-base64` responses.

## Tests
Command:
```
pytest tests/ocr/test_drawing_endpoint.py -v
```
Result:
- `tests/ocr/test_drawing_endpoint.py::test_drawing_recognize_smoke` passed
- `tests/ocr/test_drawing_endpoint.py::test_drawing_fields_catalog` passed
- `tests/ocr/test_drawing_endpoint.py::test_drawing_recognize_base64_smoke` passed
