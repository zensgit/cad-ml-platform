# DEV DedupCAD-Vision Integration Audit (2025-12-22)

## Scope
- Read `/Users/huazhou/Downloads/Github/dedupcad-vision` integration paths
- Compare to cad-ml-platform endpoints and contract
- Add compatibility endpoints/tests where missing

## Findings
- dedupcad-vision has two ML clients:
  - `caddedup_vision/ml/client.py` (aiohttp) calls `/api/analyze`, `/api/compare`, `/health/ready`.
  - `caddedup_vision/integrations/ml_platform.py` (httpx) calls `/api/v1/analyze`, `/api/v1/vectors/search`, `/api/v1/vectors/register`, `/api/v1/ocr/extract`.
- Runtime L3 path (progressive engine) uses `ml/client.py` and expects `/api/analyze` + `/health/ready` (not present in cad-ml-platform).
- Enhanced features path expects `results.features.combined` from `/api/v1/analyze`.

## Fixes Applied (cad-ml-platform)
- Added `/api/v1/vectors/register` and `/api/v1/vectors/search` compatibility endpoints.
- Added `results.features.combined` to `/api/v1/analyze` response.
- Updated integration contract doc to mention new fields/endpoints.

## Notes
- `/api/v1/analyze` still does not accept PNG/JPG; dedupcad-vision must supply CAD files for L3 semantic features.
- The `/api/analyze` and `/api/compare` paths used by `ml/client.py` remain incompatible; keep `ML_PLATFORM_ENABLED=false` unless aligned.

## Tests
- `.venv/bin/python -m pytest tests/unit/test_vectors_module_endpoints.py tests/unit/test_feature_slots.py -q`
- Result: `9 passed in 7.04s`
