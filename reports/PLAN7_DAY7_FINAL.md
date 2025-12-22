# Day 7 - Final E2E Validation

Date: 2025-12-21

## Scope

- Full E2E regression for DWG and DXF.
- Final summary and residual risks.

## Verification

- DWG smoke test:
  - `Athena/docs/SMOKE_CAD_PREVIEW_20251221_215545.md`
  - Result: `supported=true`, `pageCount=1`, PNG thumbnail OK.
- DXF smoke test:
  - `Athena/docs/SMOKE_CAD_PREVIEW_20251221_215550.md`
  - Result: `supported=true`, `pageCount=1`, PNG thumbnail OK.

## Residual risks

- Render service currently runs as a host process; production should use a
  dedicated service host with explicit lifecycle management.
- DWG conversion depends on installed tools; ensure licensing and updates are
  managed per environment.

## Deliverables

- Scripts:
  - `cad-ml-platform/scripts/cad_render_server.py`
  - `cad-ml-platform/scripts/run_cad_render_server.sh`
  - `cad-ml-platform/scripts/benchmark_cad_render.py`
  - `Athena/scripts/smoke_test_cad_preview.sh`
- Docs:
  - `cad-ml-platform/docs/CAD_RENDER_SERVER.md`
  - `cad-ml-platform/docs/CAD_RENDER_PRODUCTION.md`
  - `cad-ml-platform/docs/DWG_CONVERTERS.md`
  - `Athena/docs/INTEGRATION_CAD_PREVIEW_RENDERER.md`
