# CAD Render Server (Dev)

This document describes the standalone CAD render service used in development
for DWG/DXF -> PNG preview generation.

## Requirements

- Python venv created in the repo (`.venv`)
- CAD conversion tool (recommended: ODAFileConverter)

## Start the service

```
cd /Users/huazhou/Downloads/Github/cad-ml-platform
./scripts/run_cad_render_server.sh
```

The default port is `18002`.

## Configuration

```
CAD_RENDER_PORT=18002
DWG_CONVERTER=auto   # auto|oda|cmd|autocad|bricscad|draftsight
ODA_FILE_CONVERTER_EXE=/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter
ODA_OUTPUT_VERSION=ACAD2018
DWG_TO_DXF_CMD=
DWG_AUTOCAD_CMD=
DWG_BRICSCAD_CMD=
DWG_DRAFTSIGHT_CMD=
CAD_RENDER_AUTH_TOKEN=
```

## Use as cad-ml fallback

cad-ml-api can forward `/api/v1/render/cad` to this service when local DWG conversion
is unavailable.

```
CAD_RENDER_FALLBACK_URL=http://host.docker.internal:18002
CAD_RENDER_FALLBACK_TOKEN=
```

## Metrics

- Endpoint: `GET /metrics`
- Key metrics:
  - `cad_render_requests_total{status="ok|error"}`
  - `cad_render_duration_seconds`
  - `cad_render_input_bytes`

## Example request

```
curl -X POST -F "file=@/path/to/file.dwg" \
  http://localhost:18002/api/v1/render/cad \
  --output preview.png
```

If `CAD_RENDER_AUTH_TOKEN` is set, include the bearer token:

```
curl -X POST -H "Authorization: Bearer <token>" -F "file=@/path/to/file.dwg" \
  http://localhost:18002/api/v1/render/cad \
  --output preview.png
```

## Benchmark

```
python scripts/benchmark_cad_render.py --file /path/to/file.dwg --requests 5 --concurrency 1
```

## macOS auto-start (LaunchAgent)

```
cp /Users/huazhou/Library/LaunchAgents/com.dedupcad.cad-render.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.dedupcad.cad-render.plist
launchctl enable gui/$(id -u)/com.dedupcad.cad-render
launchctl kickstart -k gui/$(id -u)/com.dedupcad.cad-render
```

Logs:
- `/Users/huazhou/Library/Logs/cad_render_server.log`
- `/Users/huazhou/Library/Logs/cad_render_server.err`

## Notes

- The response is raw PNG bytes without a `Content-Disposition` filename header
  to avoid unicode header encoding issues.
- For DWG conversion, the service uses ODA if available; otherwise it falls back
  to a configured command template.
