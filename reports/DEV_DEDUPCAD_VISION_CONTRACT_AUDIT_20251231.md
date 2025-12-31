# DedupCAD Vision Contract Audit (2025-12-31)

## Scope

- Cross-check `cad-ml-platform` ↔ `dedupcad-vision` request/response contracts for health, search, and indexing.
- Update contract documentation to reflect current implementations.

## Code References

- cad-ml-platform
  - `src/core/dedupcad_vision.py`
  - `src/api/v1/dedup.py`
  - `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md`
- dedupcad-vision
  - `src/caddedup_vision/api/routes_progressive.py`
  - `src/caddedup_vision/api/routes_index.py`
  - `src/caddedup_vision/api/main.py`

## Findings

- `/health` contract matches: `status` is mandatory and present; extra fields are tolerated.
- `/api/v2/search` contract matches: form fields (`mode`, `max_results`, `compute_diff`, `enable_ml`, `enable_geometric`) and response fields are aligned with cad-ml-platform expectations.
- `exclude_self` is supported by dedupcad-vision (default `true`) but is not explicitly sent by cad-ml-platform; documented the default behavior.
- dedupcad-vision accepts `pdf/dxf/dwg` inputs in addition to `png/jpg` (conversion pipeline confirmed); documented the supported inputs.
- Indexing endpoints used by cad-ml-platform (`/api/index/add`, `/api/v2/index/rebuild`) were added to the contract doc with required fields.

## Changes

- Updated `docs/DEDUP2D_VISION_INTEGRATION_CONTRACT.md` to include:
  - Supported file types and `exclude_self` default.
  - Index add and index rebuild endpoint contracts.
  - Pass-through query parameters in `cad-ml-platform` search API.

## Tests

```bash
API_BASE_URL=http://localhost:8001 \
DEDUPCAD_VISION_URL=http://localhost:58001 \
.venv/bin/python -m pytest \
  tests/integration/test_dedupcad_vision_contract.py \
  tests/integration/test_e2e_api_smoke.py -q
```

Result: 4 passed.

## Notes

- Test run used local dedupcad-vision (`start_server.py --port 58001`) and local cad-ml-platform (`uvicorn src.main:app --port 8001`).
- Both services were stopped after validation.
