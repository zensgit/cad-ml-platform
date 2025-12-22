#!/usr/bin/env markdown
# Dev Service Smoke (Extended)

## Scope
- Validate health, metrics, analyze, similarity, vectors stats, drift status, and batch similarity on running dev service.

## Environment
- Base URL: `http://localhost:8000`
- Container: `cad-ml-api` (port mapped 8000 -> 8000)
- ENV: `WORKERS=1`, `VECTOR_STORE_BACKEND=redis`, `REDIS_URL=redis://redis:6379/0`
- Redis running: `cad-ml-redis` (host port 16379)

## Results
- GET `/health` -> `200`
- GET `/health/extended` -> `200`
- GET `/api/v1/health/features/cache` -> `200`
- GET `/metrics/` -> `200`
- POST `/api/v1/analyze/` (DXF stub) -> `200`
- POST `/api/v1/analyze/` (second sample) -> `200`
- POST `/api/v1/analyze/similarity` -> `200`
- POST `/api/v1/analyze/similarity/topk` -> `200`
- GET `/api/v1/vectors/` -> `200`
- GET `/api/v1/vectors_stats/stats` -> `200`
- GET `/api/v1/vectors_stats/distribution` -> `200`
- GET `/api/v1/faiss/health` -> `200` (available=false, status=unavailable)
- GET `/api/v1/health/faiss/health` -> `200` (same as above)
- GET `/api/v1/analyze/drift` -> `200`
- GET `/api/v1/analyze/drift/baseline/status` -> `200`
- POST `/api/v1/analyze/drift/baseline/export` -> `200` (status=empty)
- POST `/api/v1/vectors/similarity/batch` -> `200`

## Observations
- Vector stats report `backend=redis`, `total=8`, `average_dimension=7.0` (persisted Redis view).
- `/api/v1/vectors/` still reports in-memory vectors for the current process only.
- Redis holds 8 `vector:*` keys after repeated smoke runs; top-k can include older entries.
- Redis vector metadata now includes `feature_version` + `vector_layout` after backfill.
- Baseline export is a POST endpoint; GET returns `405` if used.
- Drift status is `baseline_pending` because min count not reached.

## Notes
- Use trailing slashes (`/metrics/`, `/api/v1/analyze/`, `/api/v1/vectors/`) to avoid 307 redirects.
- Use unique filenames/content in smoke requests to avoid Redis cache hits.
