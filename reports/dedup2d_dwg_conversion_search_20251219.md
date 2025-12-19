# Dedup2D DWG Direct Conversion + Search Report

Date: 2025-12-19

## Goal
Validate direct DWG conversion workflow for many DWG files: DWG -> DXF -> PNG/JSON -> index -> search.

## Environment
- ODA converter: `/Applications/ODAFileConverter.app/Contents/MacOS/ODAFileConverter`
- cad-ml-platform API: http://localhost:18000
- dedupcad-vision API: http://localhost:58001

## Input DWG Files
- BTJ01231501522-00短轴承座(盖)v2.dwg
- BTJ01231501522-00短轴承座(盖)v3.dwg
- BTJ01231301522-01轴向定位轴承v1.dwg

## Steps Executed
1) Convert DWG -> DXF using ODA converter (host).
2) Render DXF -> PNG and extract v2 JSON in cad-ml-platform container.
3) Index PNG + JSON via `/api/v1/dedup/2d/index/add`.
4) Run async search via `/api/v1/dedup/2d/search` using PNG + v2 JSON.

## Index Results
- v1: drawing_id=1
- v2: drawing_id=2
- v3: drawing_id=3

## Search Results (Summary)
### Query: v1
- `total_matches`: 1
- Duplicates: self only

### Query: v2
- `total_matches`: 2
- Duplicates:
  - v2 (self, similarity=1.0)
  - v3 (similarity=0.995, precision_score=0.993)

### Query: v3
- `total_matches`: 2
- Duplicates:
  - v3 (self, similarity=1.0)
  - v2 (similarity=0.995, precision_score=0.993)

## Notes
- The conversion + search pipeline works; DWG cannot be sent directly to the 2D endpoint without conversion.
- Self-matches appear because items are indexed before search; you may want to filter out identical file_hash/file_name in post-processing.
- Rendering produced warnings: `no default font found` inside the cad-ml-platform container. For better text fidelity, install fonts (e.g., DejaVu) in the image.

## Recommendation for Many DWG Files
Use a batch pipeline:
1) Convert all DWG -> DXF (ODA)
2) Render PNG + v2 JSON + index
3) Search each PNG + JSON and collect results

The existing script `scripts/dedup_2d_batch_ingest_cad.py` covers step 2 for DXF inputs. A small batch search script can be added to automate step 3.
