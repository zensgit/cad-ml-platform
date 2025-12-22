# Day 3 - Observability

Date: 2025-12-21

## Scope

- Add metrics to the CAD render service.
- Track CAD preview success/failure in Athena.

## Changes

- Render service metrics:
  - `cad-ml-platform/scripts/cad_render_server.py`
  - Added `/metrics` endpoint and counters/histograms.
- Athena metrics:
  - `Athena/ecm-core/src/main/java/com/ecm/core/preview/PreviewService.java`
  - Added `cad_preview_total{status,reason}` via Micrometer.
- Docs:
  - `cad-ml-platform/docs/CAD_RENDER_SERVER.md`
  - `Athena/docs/INTEGRATION_CAD_PREVIEW_RENDERER.md`

## Verification

- Render service metrics:
  - `GET http://localhost:18002/metrics`
  - Observed `cad_render_requests_total`, `cad_render_duration_seconds`, `cad_render_input_bytes`.
- Athena metrics:
  - Ran `Athena/scripts/smoke_test_cad_preview.sh`.
  - `GET http://localhost:7700/actuator/prometheus` shows `cad_preview_total{status="ok",reason="rendered"}`.
- Smoke test report:
  - `Athena/docs/SMOKE_CAD_PREVIEW_20251221_215017.md`
