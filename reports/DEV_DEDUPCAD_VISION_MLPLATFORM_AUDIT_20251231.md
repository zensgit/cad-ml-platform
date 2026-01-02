# DedupCAD Vision ↔ CAD ML Platform ML Integration Audit (2025-12-31)

## Scope

- Validate ML integration flow assumptions between dedupcad-vision and cad-ml-platform.
- Confirm file-type compatibility for `/api/v1/analyze` used by dedupcad-vision L3.
- Update contract doc to clarify L3 prerequisites.

## Findings

- dedupcad-vision L3 (`ProgressiveSearchEngine`) only runs when `ml_input_path` is available; otherwise it skips L3 and emits a warning.
- `ml_input_path` is set when the input contains a CAD source (e.g., DXF/DWG) and `enable_ml=true`; image-only inputs (PNG/JPG/PDF) do not provide a CAD path.
- cad-ml-platform `/api/v1/analyze` only accepts CAD formats (`dxf/dwg/step/stp/iges/igs/stl`), so PNG/JPG inputs are rejected with `UNSUPPORTED_FORMAT`.

## Changes

- Updated `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md` to explicitly note L3 prerequisites and supported analyze formats.

## Tests

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

png_path = Path("data/dxf_fixtures_subset_out/mixed.png")
dxf_path = Path("data/dxf_fixtures_subset/mixed.dxf")

with png_path.open("rb") as fh:
    resp_png = client.post(
        "/api/v1/analyze/",
        files={"file": (png_path.name, fh, "image/png")},
    )

with dxf_path.open("rb") as fh:
    resp_dxf = client.post(
        "/api/v1/analyze/",
        files={"file": (dxf_path.name, fh, "application/dxf")},
    )

print("PNG status:", resp_png.status_code)
print("PNG detail code:", (resp_png.json().get("detail") or {}).get("code"))
print("DXF status:", resp_dxf.status_code)
PY
```

Result:
- PNG status: 400 (`UNSUPPORTED_FORMAT`)
- DXF status: 200
