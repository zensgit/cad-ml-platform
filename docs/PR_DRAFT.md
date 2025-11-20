# PR Draft: Observability, Metrics, Lint Phase 1, and Tests

## Summary
- Standardized error model (HTTP 200 + `{success, code, error}`) across Vision/OCR
- Expanded metrics: EMA error rates, input size histograms, detailed rejection reasons
- Health `/health` includes runtime + EMA + config
- Added dev tooling, CI non-blocking `lint-all` report, and Grafana/PromQL examples
- Phase 1 lint clean for `src/`, tests expanded; all tests pass

## Key Changes
- Metrics: `vision_image_size_bytes`, `ocr_image_size_bytes`, `*_error_rate_ema`
- Rejection reasons: Vision base64/url; OCR mime/size/pdf
- Config: `VISION_MAX_BASE64_BYTES`, `ERROR_EMA_ALPHA`, `OCR_MAX_PDF_PAGES`, `OCR_MAX_FILE_MB`
- Tests: Base64 cases, PDF rejections, provider down
- Docs: README, ALERT_RULES, runbooks, Grafana dashboard JSON

## How to Test
1. `uvicorn src.main:app --reload`
2. `GET /health` to see EMA and config
3. Hit `/api/v1/vision/analyze` with invalid base64 and check `/metrics`
4. Upload a big PDF to `/api/v1/ocr/extract` and check `ocr_input_rejected_total`

## Screenshots / Artifacts
- Grafana: `docs/grafana/observability_dashboard.json`
- Alert rules: `docs/ALERT_RULES.md`
- Runbooks: `docs/runbooks/*.md`

## CI
- `lint-type` job lints `src` and runs mypy
- `lint-all-report` uploads full flake8 output (non-blocking)
- Tests run on 3.10/3.11; 145 tests pass locally

