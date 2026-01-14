# OCR_PROVIDER_HEALTH_DESIGN

## Goal
- Expose provider catalog and health status for OCR and drawing recognition clients.

## Changes
- Add `GET /api/v1/ocr/providers` to list registered OCR providers and default selection.
- Add `GET /api/v1/ocr/health` to report provider readiness status.
- Add `GET /api/v1/drawing/providers` and `GET /api/v1/drawing/health` as drawing-facing aliases.

## Approach
- Reuse the OCR manager registry to list providers.
- Aggregate provider `health_check()` results into an overall status: healthy, degraded, unhealthy, or unavailable.
