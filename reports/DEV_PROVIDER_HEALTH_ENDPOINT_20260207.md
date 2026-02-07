# DEV_PROVIDER_HEALTH_ENDPOINT_20260207

## Summary

Added a core providers health-check endpoint that runs best-effort, timeout-bounded checks for all providers registered in `src/core/providers/ProviderRegistry`.

## Why

`/api/v1/providers/registry` (and `/api/v1/health/providers/registry`) exposes the provider registry snapshot but does not indicate which providers are currently *ready*.

This endpoint fills that gap so operations can see optional dependency availability (for example: Graph2D disabled when `torch` is missing) without digging through logs.

## Changes

- `src/api/v1/health.py`
  - Added `GET /api/v1/providers/health`
  - Added alias `GET /api/v1/health/providers/health`
  - Adds response models: `ProviderHealthItem`, `ProviderHealthResponse`
  - Uses `timeout_seconds` query param (default `0.75`, capped at `10.0`)
  - Returns stable-sorted results by `(domain, provider)`

- `tests/unit/test_provider_health_endpoint.py`
  - Covers response shape + stable sorting
  - Uses patching to avoid optional dependency variance

## API

Request:

```bash
curl -H "X-API-Key: test-key" "http://127.0.0.1:8000/api/v1/providers/health?timeout_seconds=0.5"
```

Response shape:

```json
{
  "status": "ok",
  "total": 3,
  "ready": 2,
  "timeout_seconds": 0.5,
  "results": [
    {"domain":"classifier","provider":"hybrid","ready":true,"snapshot":{...}},
    {"domain":"classifier","provider":"graph2d","ready":false,"snapshot":{...},"error":"..."},
    {"domain":"ocr","provider":"paddle","ready":true,"snapshot":{...}}
  ]
}
```

## Validation

Executed:

```bash
python3 -m pytest -q tests/unit/test_provider_health_endpoint.py
```

Result: PASS

