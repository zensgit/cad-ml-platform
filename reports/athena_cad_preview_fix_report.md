# Athena CAD Preview Fix Report

Date: 2025-12-20

## Scope

- Add a CAD render API in `cad-ml-platform` for DWG/DXF → PNG.
- Wire Athena `ecm-core` preview/thumbnail flow to call the render API for CAD files.

## Changes

### cad-ml-platform

- Added CAD render endpoint:
  - `src/api/v1/render.py` (`POST /api/v1/render/cad`, returns `image/png`).
- Registered router:
  - `src/api/__init__.py` (`/v1/render`).

### Athena (ecm-core)

- Added CAD preview integration and safe MIME handling:
  - `ecm-core/src/main/java/com/ecm/core/preview/PreviewService.java`
    - CAD preview uses external render service.
    - CAD thumbnail uses render result + thumbnailator with fallback to default.
    - MIME null-safe normalization.
    - CAD detection via MIME **or** file extension.
    - Result metadata is preserved (`documentId`, `mimeType`).
- New configuration keys:
  - `ecm-core/src/main/resources/application.yml`
  - `docker-compose.yml` (pass-through env vars)

## New Configuration

```
ECM_PREVIEW_CAD_ENABLED=true
ECM_PREVIEW_CAD_RENDER_URL=http://<cad-ml-host>:8000/api/v1/render/cad
ECM_PREVIEW_CAD_AUTH_TOKEN=
ECM_PREVIEW_CAD_TIMEOUT_MS=30000
```

## Verification

### Automated

- Attempted import sanity check for the new render endpoint.
- Result: **failed due to local environment dependencies**
  - `ModuleNotFoundError: arq`
  - `scipy`/`numpy` version mismatch in local Python 3.9 environment

No runtime services were started for this check.

### Manual Verification (recommended)

1. Start cad-ml-platform API (ensure `arq`, `ezdxf`, `matplotlib` are installed):

```
cd /Users/huazhou/Downloads/Github/cad-ml-platform
uvicorn src.main:app --reload --port 8000
```

2. Configure Athena to call the render endpoint:

```
export ECM_PREVIEW_CAD_RENDER_URL=http://localhost:8000/api/v1/render/cad
```

3. Hit Athena thumbnail endpoint for a DWG/DXF document:

```
curl -H "Authorization: Bearer <token>" \
  http://localhost:7700/api/v1/documents/<doc_id>/thumbnail --output preview.png
```

4. Confirm `preview.png` is a rendered CAD image (not a gray placeholder).

## Notes / Risks

- DWG rendering depends on `ODA_FILE_CONVERTER_EXE` or `DWG_TO_DXF_CMD` in
  `cad-ml-platform`. Without it, DWG → DXF conversion fails.
- If CAD render service is unavailable, Athena falls back to a default thumbnail.

## Verification (2025-12-21)

### Runtime setup

- Started a local cad-ml-platform container with the updated code mounted:
  - Container: `cad-ml-api-local`
  - Port: `http://localhost:18001`
- Configured Athena to call the renderer:
  - `ECM_PREVIEW_CAD_RENDER_URL=http://host.docker.internal:18001/api/v1/render/cad`

### Results

- **Render API (DXF)**
  - Request: `POST /api/v1/render/cad` with `test_left.dxf`
  - Status: `200`
  - Output: PNG (`834x960`)
- **Athena preview (DXF)**
  - Upload: `test_left.dxf` via `/api/v1/documents/upload-legacy`
  - Preview: `/api/v1/documents/{id}/preview`
  - Result: `supported=true`, `pageCount=1`
  - Thumbnail: `/api/v1/documents/{id}/thumbnail` → PNG (`174x200`)
- **Athena preview (DWG)**
  - Upload: `BTJ01231501522-00短轴承座(盖)v2.dwg`
  - Preview result: `supported=false`
  - Error: `DWG conversion unavailable: set ODA_FILE_CONVERTER_EXE or DWG_TO_DXF_CMD`

### Open Items

- DWG requires a converter. Configure one of:
  - `ODA_FILE_CONVERTER_EXE=/path/to/ODAFileConverter`
  - `DWG_TO_DXF_CMD` (custom conversion command)

Once configured, repeat the DWG preview test to validate full DWG → PNG flow.

## Multi-tool DWG conversion support

- Added `DWG_CONVERTER` selection: `auto|oda|cmd|autocad|bricscad|draftsight`.
- ODA auto-detect now checks default macOS path:
  - `/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`
- Command-based converters are configured via:
  - `DWG_TO_DXF_CMD`, `DWG_AUTOCAD_CMD`, `DWG_BRICSCAD_CMD`, `DWG_DRAFTSIGHT_CMD`

## Verification (Local ODA + render)

- ODA auto-detect:
  - `resolve_oda_exe_from_env()` resolved to `/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`.
- DWG → DXF conversion:
  - Input: `BTJ01231501522-00短轴承座(盖)v2.dwg`
  - Output: `/tmp/athena_dwg_convert_test.dxf` (size: 836,901 bytes)
- DXF → PNG render (local .venv):
  - Output: `/tmp/athena_dwg_render.png` (size: 198,186 bytes)

Note: Full Athena → render API validation for DWG still requires running the render API on the host
(where ODA is available). If needed, I can start a local render-only service and re-test the Athena
thumbnail/preview endpoints end-to-end.

## End-to-end DWG verification (host render service)

- Local render service started on host:
  - `uvicorn cad_render_server:app --port 18002`
  - ODA: `/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`
- Athena config:
  - `ECM_PREVIEW_CAD_RENDER_URL=http://host.docker.internal:18002/api/v1/render/cad`

### Results

- Render API (DWG):
  - `POST http://localhost:18002/api/v1/render/cad` with `BTJ01231501522-00短轴承座(盖)v2.dwg`
  - Status: `200`
  - Output: PNG (`1357x960`)
- Athena preview (DWG):
  - Upload: `BTJ01231501522-00短轴承座(盖)v2.dwg`
  - Preview: `supported=true`, `pageCount=1`
  - Thumbnail: PNG (`200x141`)

### Notes

- The render endpoint no longer sets a `Content-Disposition` filename header to avoid
  Unicode header encoding errors for non-ASCII CAD filenames.
- Host render service PID is stored in `/tmp/cad_render_server.pid` (stop with `kill $(cat /tmp/cad_render_server.pid)`).

## Dev workflow artifacts

- Scripted render service:
  - `scripts/cad_render_server.py`
  - `scripts/run_cad_render_server.sh`
- Docs:
  - `docs/CAD_RENDER_SERVER.md`
  - Athena: `docs/INTEGRATION_CAD_PREVIEW_RENDERER.md`

## Base64 line-break fix (Athena)

- Added Jackson configuration to force `Base64Variants.MIME_NO_LINEFEEDS`:
  - `Athena/ecm-core/src/main/java/com/ecm/core/config/JacksonConfig.java`
- Verification:
  - Preview JSON parsed successfully with Python `json.loads`
  - No newline bytes detected in the `content` base64 segment
  - Result: `supported=true`, `pageCount=1`

## Smoke test script

- Added: `Athena/scripts/smoke_test_cad_preview.sh`
- Verified output:
  - `supported=true`, `pageCount=1`
  - thumbnail PNG generated
