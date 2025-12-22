# CAD Render Autostart + Token Rotation Report

## Goal
- Move the CAD render service to a LaunchAgent-safe runtime path (outside Downloads).
- Rotate auth token and verify end-to-end preview flow in Athena.

## Changes Applied
- Created runtime copy at `/Users/huazhou/Library/Application Support/dedupcad/cad-render` and installed dependencies in `.venv`.
- Updated LaunchAgent to run the runtime copy and new token:
  - `/Users/huazhou/Library/LaunchAgents/com.dedupcad.cad-render.plist`
  - `CAD_RENDER_AUTH_TOKEN=GVrzfzYVSHKcTJSv2zLt8ZRJdpUB3VuwuIo-Vvt932A`
  - `PYTHON_BIN=/Users/huazhou/Library/Application Support/dedupcad/cad-render/.venv/bin/python`
- Restarted LaunchAgent with `launchctl bootout` + `launchctl bootstrap`.
- Rebuilt `athena-ecm-core` so it picks up the rotated token in `.env`.

## Verification
- LaunchAgent status:
  - `launchctl print gui/501/com.dedupcad.cad-render` shows running and points to runtime path.
- Render service health:
  - `GET http://localhost:18002/health` -> `200`.
- Auth enforcement:
  - Missing token -> `401` on `/api/v1/render/cad`.
  - Valid token -> `200` and PNG returned.
- Athena end-to-end:
  - `scripts/smoke_test_cad_preview.sh` completed OK.
  - Report: `/Users/huazhou/Downloads/Github/Athena/docs/SMOKE_CAD_PREVIEW_20251222_083614.md`.

## Latest Update Runs
- `CAD_RENDER_UPDATE_RUN_20251222_093219.md` (health check OK, no Athena smoke).
- `CAD_RENDER_UPDATE_RUN_20251222_114125.md` (health check OK + Athena smoke OK).
  - Smoke report: `/Users/huazhou/Downloads/Github/Athena/docs/SMOKE_CAD_PREVIEW_20251222_114125.md`.
- `CAD_RENDER_UPDATE_RUN_20251222_130125.md` (health check OK + Athena smoke OK, auto-rollback enabled).
  - Smoke report: `/Users/huazhou/Downloads/Github/Athena/docs/SMOKE_CAD_PREVIEW_20251222_130125.md`.

## Logs
- Render service logs:
  - `/Users/huazhou/Library/Logs/cad_render_server.log`
  - `/Users/huazhou/Library/Logs/cad_render_server.err`

## Notes
- This resolves the macOS TCC restriction caused by running LaunchAgents from `~/Downloads`.
- If the token is rotated again, update both LaunchAgent env and Athena `.env`, then rebuild `athena-ecm-core`.
