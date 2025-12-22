#!/usr/bin/env markdown
# Dev Service Smoke (cad-ml-api)

## Scope
- Basic health, metrics, analyze, similarity endpoints against the running dev service.

## Environment
- Base URL: `http://localhost:18000`
- Container: `cad-ml-api` (port mapped 8000 -> 18000)

## Requests
- GET `/health` -> `200`
- GET `/health/extended` -> `200`
- GET `/api/v1/health/features/cache` -> `200`
- GET `/metrics/` -> `200`
- POST `/api/v1/analyze/` (DXF stub, `X-API-Key: test`) -> `200`
- POST `/api/v1/analyze/` (second sample) -> `200`
- POST `/api/v1/analyze/similarity` -> `200`
- POST `/api/v1/analyze/similarity/topk` (`target_id`, `k`, `exclude_self`) -> `200`

## Notes
- Endpoints with trailing slash (`/metrics/`, `/api/v1/analyze/`) avoid 307 redirects.
