# CAD Render Service - Production Deployment Guide

This document outlines a production deployment pattern for the CAD render service
used by Athena previews (DWG/DXF -> PNG).

## Recommended topology

- **Renderer** runs as a dedicated service on a host with DWG conversion tools.
- **Athena ecm-core** calls the renderer via internal HTTP.
- **DedupCAD Vision** remains unchanged.

```
Athena ecm-core ---> CAD Render Service ---> ODA / CAD tool
```

## Option A: Windows host + ODA (recommended)

1) Install ODAFileConverter (Windows).
2) Deploy the render service (Python + FastAPI) on the same host.
3) Configure a firewall rule to allow internal access from Athena.

### Environment

```
DWG_CONVERTER=oda
ODA_FILE_CONVERTER_EXE=C:\\Program Files\\ODA\\ODAFileConverter\\ODAFileConverter.exe
CAD_RENDER_PORT=18002
CAD_RENDER_AUTH_TOKEN=<shared-secret>
```

### Start command

```
set PYTHONPATH=C:\\path\\to\\cad-ml-platform
python -m uvicorn scripts.cad_render_server:app --host 0.0.0.0 --port 18002
```

## Option B: Command-based conversion (fallback)

If you use AutoCAD/BricsCAD/DraftSight, configure a command template:

```
DWG_CONVERTER=autocad
DWG_AUTOCAD_CMD="C:\\Program Files\\Autodesk\\AutoCAD 2024\\accoreconsole.exe /i {input} /o {output} /s dwg2dxf.scr"
```

See `docs/DWG_CONVERTERS.md` for command templates and tool-specific notes.

## Option C: macOS LaunchAgent (local/edge)

macOS LaunchAgents cannot reliably execute binaries from `~/Downloads` due to TCC.
Use a runtime directory under `~/Library/Application Support` and sync the
renderer files into it.

### Sync runtime files

```
RUNTIME_DIR="$HOME/Library/Application Support/dedupcad/cad-render" \
  INSTALL_DEPS=1 \
  ./scripts/sync_cad_render_runtime.sh
```

### One-command update + restart + health check

```
./scripts/update_cad_render_runtime.sh
```

To include a full Athena smoke test:

```
RUN_ATHENA_SMOKE=1 ./scripts/update_cad_render_runtime.sh
```

The update script creates a temporary backup and rolls back automatically if
the health check fails. Set `KEEP_BACKUP=1` to preserve the backup directory.

### LaunchAgent plist (example)

Save as `~/Library/LaunchAgents/com.dedupcad.cad-render.plist`:

```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.dedupcad.cad-render</string>
    <key>ProgramArguments</key>
    <array>
      <string>/Users/<user>/Library/Application Support/dedupcad/cad-render/run_cad_render_server.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/<user>/Library/Application Support/dedupcad/cad-render</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>CAD_RENDER_PORT</key>
      <string>18002</string>
      <key>CAD_RENDER_AUTH_TOKEN</key>
      <string>change-me</string>
      <key>DWG_CONVERTER</key>
      <string>auto</string>
      <key>ODA_FILE_CONVERTER_EXE</key>
      <string>/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter</string>
      <key>PYTHON_BIN</key>
      <string>/Users/<user>/Library/Application Support/dedupcad/cad-render/.venv/bin/python</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/<user>/Library/Logs/cad_render_server.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/<user>/Library/Logs/cad_render_server.err</string>
  </dict>
</plist>
```

### Enable and check

```
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.dedupcad.cad-render.plist || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dedupcad.cad-render.plist
launchctl enable gui/$(id -u)/com.dedupcad.cad-render
launchctl kickstart -k gui/$(id -u)/com.dedupcad.cad-render
launchctl print gui/$(id -u)/com.dedupcad.cad-render
curl http://localhost:18002/health
```

## Athena configuration

```
ECM_PREVIEW_CAD_ENABLED=true
ECM_PREVIEW_CAD_RENDER_URL=http://<renderer-host>:18002/api/v1/render/cad
ECM_PREVIEW_CAD_AUTH_TOKEN=
ECM_PREVIEW_CAD_TIMEOUT_MS=30000
```

If you enable `CAD_RENDER_AUTH_TOKEN`, set the same value in
`ECM_PREVIEW_CAD_AUTH_TOKEN`.

## Operational checklist

- Renderer `/health` returns 200.
- ODA conversion produces DXF within SLA.
- Render service latency within limits (DWG < 10s recommended).
- Logs and metrics collected.
