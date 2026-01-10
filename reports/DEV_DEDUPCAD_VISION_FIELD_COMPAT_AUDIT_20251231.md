# DedupCAD Vision Field Compatibility Audit (2025-12-31)

## Scope

- Field-level compatibility check for dedupcad-vision ↔ cad-ml-platform ML call paths.
- Focus: `/api/v1/analyze`, `/api/v1/vectors/register`, `/api/v1/vectors/search`, optional fallback paths.

## Findings

- `/api/v1/analyze` expects `options` JSON string; dedupcad-vision sends `extract_features`, `classify_parts`, `enable_ocr` fields correctly.
- `/api/v1/analyze` accepts CAD formats only (`dxf/dwg/step/stp/iges/igs/stl`); PNG/JPG inputs are rejected (`UNSUPPORTED_FORMAT`).
- `/api/v1/vectors/register` payload fields (`id`, `vector`, `meta`) align with dedupcad-vision usage.
- `/api/v1/vectors/search` payload fields (`vector`, `k`, `material_filter`, `complexity_filter`) align with dedupcad-vision usage.
- `dedupcad-vision` L3 compare path falls back to `POST /api/compare` if vector lookup misses; cad-ml-platform does not expose `/api/compare`, so that branch returns no result (non-blocking but reduces L3 coverage unless vectors are registered by hash).

## Changes

- Updated `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md` with vector register/search payloads and fallback caveat.

## Tests

```bash
.venv/bin/python - <<'PY'
import uuid
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)
vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
vid = f"ml-audit-{uuid.uuid4().hex[:8]}"

register_resp = client.post(
    "/api/v1/vectors/register",
    json={
        "id": vid,
        "vector": vector,
        "meta": {"material": "steel", "complexity": "low", "format": "dxf"},
    },
)
search_resp = client.post(
    "/api/v1/vectors/search",
    json={"vector": vector, "k": 5, "material_filter": "steel"},
)

print("register status:", register_resp.json().get("status"))
print("search total:", search_resp.json().get("total"))
print("found:", any(item.get("id") == vid for item in search_resp.json().get("results", [])))
client.post("/api/v1/vectors/delete", json={"id": vid})
PY
```

Result:
- `register status: accepted`
- `search total: 1`
- `found: True`
