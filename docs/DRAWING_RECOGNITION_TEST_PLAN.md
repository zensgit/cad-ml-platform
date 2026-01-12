# Drawing Recognition Test Plan

## Coverage Goals
- Validate title block parsing for English and Chinese label patterns.
- Ensure drawing recognition endpoint returns a normalized response structure.
- Confirm route registration under `/api/v1/drawing`.

## Test Cases
1. `tests/ocr/test_title_block_parser.py`
   - English title block extraction (drawing number, revision, scale, sheet, date, etc.).
   - Chinese title block extraction (图号, 材料, 比例, 页, 日期, 重量, 投影).
   - Confidence capture from OCR line scores.

2. `tests/ocr/test_drawing_endpoint.py`
   - Uses a dummy OCR manager to verify response fields and dimensions without external dependencies.

3. `tests/test_routes_smoke.py`
   - Route existence check for `/api/v1/drawing/recognize` in the main FastAPI app.

## Notes
- Endpoint tests avoid real OCR providers to keep runtime deterministic.
- Validation steps and results are recorded in the verification report.
