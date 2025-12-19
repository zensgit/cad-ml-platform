# Dedup2D DWG Sample Validation Report

Date: 2025-12-19

## Scope
Test async 2D search with provided DWG files via cad-ml-platform -> dedupcad-vision.

## Environment
- cad-ml-platform: http://localhost:18000
- dedupcad-vision: http://localhost:58001
- Endpoint: `POST /api/v1/dedup/2d/search?async=true&mode=balanced&max_results=5`
- Header: `X-API-Key: test`

## Input Files
- /Users/huazhou/Downloads/训练图纸/训练图纸/BTJ01231501522-00短轴承座(盖)v2.dwg (163,715 bytes)
- /Users/huazhou/Downloads/训练图纸/训练图纸/BTJ01231501522-00短轴承座(盖)v3.dwg (156,451 bytes)
- /Users/huazhou/Downloads/训练图纸/训练图纸/BTJ01231301522-01轴向定位轴承v1.dwg (121,394 bytes)

## Results
### File 1
- job_id: 73fb84b4-8ecc-4574-b7eb-94f0af597982
- status: completed
- success: false
- error: cannot identify image file '/tmp/tmptr2g1v06.dwg'

### File 2
- job_id: 445577ba-c855-4701-aeda-8f7bf3956add
- status: completed
- success: false
- error: cannot identify image file '/tmp/tmp9cxd457m.dwg'

### File 3
- job_id: c048b77c-61e1-48a2-80bb-ce1d9dce7b63
- status: completed
- success: false
- error: cannot identify image file '/tmp/tmpco21cr1z.dwg'

## Conclusion
The 2D dedup pipeline expects raster inputs (PNG/JPG/PDF). Raw DWG files cannot be processed directly by dedupcad-vision, resulting in image decode errors.

## Recommended Next Step
Convert DWG -> DXF -> PNG (and optionally geom JSON) before calling the 2D endpoints. The repo includes a helper script:

```bash
python3 scripts/dedup_2d_batch_ingest_cad.py \
  "/Users/huazhou/Downloads/训练图纸/训练图纸" \
  --base-url http://localhost:18000 \
  --api-key test \
  --user-name batch \
  --dwg-to-dxf auto \
  --work-dir data/dedupcad_batch \
  --max-files 3
```

Notes:
- DWG conversion requires ODA File Converter or a custom `DWG_TO_DXF_CMD`.
- The script renders PNG + geom JSON and calls `/api/v1/dedup/2d/index/add` for indexing.
