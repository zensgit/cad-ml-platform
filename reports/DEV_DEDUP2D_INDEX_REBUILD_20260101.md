# DEV_DEDUP2D_INDEX_REBUILD_20260101

## Scope
- Validate dedupcad-vision index rebuild call via cad-ml API.

## Request
```bash
curl -sS -H 'X-API-Key: test' -X POST \
  http://localhost:18000/api/v1/dedup/2d/index/rebuild | jq '.'
```
Response:
```json
{"success":true,"message":"Indexes rebuilt successfully"}
```

## Post-check
```bash
curl -sS http://localhost:8100/health | jq '.indexes'
```
Observed:
- `l2_faiss.ready=true` (was false before rebuild in this session).

Metrics signal:
```bash
curl -sSf --max-time 10 http://localhost:18000/metrics/ \
  | rg "dedupcad_vision_requests_total" | rg "rebuild" 
```
Observed:
- `dedupcad_vision_requests_total{endpoint="rebuild_indexes",status="success"}=1`

## Result
- Index rebuild request completed and dedupcad-vision reported updated readiness.
