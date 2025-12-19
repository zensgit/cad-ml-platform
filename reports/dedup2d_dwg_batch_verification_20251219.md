# Dedup2D DWG Batch Verification Report (2025-12-19)

## Scope
- Repository: cad-ml-platform
- Dataset: 110 DWG files under `/Users/huazhou/Downloads/训练图纸/训练图纸`
- Pipeline: DWG -> DXF (ODA) -> PNG + v2 geom JSON -> index add -> rebuild -> async search
- Runtime: Docker `cad-ml-api` container (API), `dedupcad-vision` service for indexing/search

## Fixes Applied
- `src/core/dedupcad_precision/cad_pipeline.py`:
  - Text rendering policy now falls back to `TextPolicy.FILLING` when `ACCURATE` is unavailable.
  - Added `DEDUPCAD2_RENDER_HATCH` env flag to skip hatch rendering when disabled.
  - Rendering filter skips hatch/text/proxy entities when disabled to avoid render OOM.

## Execution Summary
- DXF conversion: 110 files produced under `/app/data/dedupcad_batch/input`.
- Indexing:
  - All 110 files indexed successfully.
  - Index rebuild triggered once after indexing.
- Search:
  - Batch search completed with `--skip-index`.
  - Results persisted to JSON + CSV artifacts.

## Results Summary
- Total files: 110
- Search success: 110
- Errors: 0
- Empty matches: 0
- Match count range: 1-4 (avg 2.1)
- Highest match group:
  - `J0525002-04-02过滤托架v1/v2/v3/v4.dxf` (4 matches each)

## Special Handling
- The following files required hatch/text rendering to be disabled to avoid OOM:
  - `BTJ01231301522-00蜗轮蜗杆传动出料机构v1.dxf`
  - `BTJ01231301522-00蜗轮蜗杆传动出料机构v2.dxf`
  - `BTJ01231301522-00蜗轮蜗杆传动出料机构v3.dxf`

## Artifacts
- `reports/dedup2d_batch_search_results_20251219.json`
- `reports/dedup2d_batch_search_results_20251219.csv`

## Reproduce (Container)
```
# Index-only (rerun as needed)
DEDUPCAD2_RENDER_TEXT=0 DEDUPCAD2_RENDER_HATCH=0 \
python /app/scripts/dedup_2d_batch_search_cad.py /app/data/dedupcad_batch/input \
  --base-url http://localhost:8000 --api-key test --user-name batch \
  --skip-search --no-rebuild-index --work-dir /app/data/dedupcad_batch/work

# Rebuild index
python - <<'PY'
import requests
resp = requests.post('http://localhost:8000/api/v1/dedup/2d/index/rebuild', headers={'X-API-Key':'test'}, timeout=120)
print(resp.status_code, resp.text)
PY

# Search-only
DEDUPCAD2_RENDER_TEXT=0 DEDUPCAD2_RENDER_HATCH=0 \
python /app/scripts/dedup_2d_batch_search_cad.py /app/data/dedupcad_batch/input \
  --base-url http://localhost:8000 --api-key test --user-name batch \
  --skip-index --mode balanced --max-results 5 \
  --work-dir /app/data/dedupcad_batch/work \
  --output-json /app/data/dedupcad_batch/results_full_20251219.json \
  --output-csv /app/data/dedupcad_batch/results_full_20251219.csv
```
