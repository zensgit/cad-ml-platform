# CAD Render Sync + Autostart Verification Report

## Scope
- Validate LaunchAgent state after sync script addition.
- Verify render service health.
- Run Athena end-to-end smoke test for CAD preview.

## Checks

### LaunchAgent
```
launchctl print gui/$(id -u)/com.dedupcad.cad-render
```
Result: **running**
- Program: `/Users/huazhou/Library/Application Support/dedupcad/cad-render/scripts/run_cad_render_server.sh`
- Working dir: `/Users/huazhou/Library/Application Support/dedupcad/cad-render`
- Token: `CAD_RENDER_AUTH_TOKEN` set

### Render Service Health
```
curl http://localhost:18002/health
```
Result: **200**

### Athena Smoke Test
```
RENDER_LOG_PATH=/Users/huazhou/Library/Logs/cad_render_server.log \
  /Users/huazhou/Downloads/Github/Athena/scripts/smoke_test_cad_preview.sh
```
Result: **OK**
- Report: `/Users/huazhou/Downloads/Github/Athena/docs/SMOKE_CAD_PREVIEW_20251222_092650.md`

## Notes
- Sync script path: `scripts/sync_cad_render_runtime.sh`
- LaunchAgent remains stable after sync.
