#!/usr/bin/env markdown
# Dev Restart Persistence Smoke

## Scope
- Verify Redis-backed vectors persist across `cad-ml-api` restart.

## Environment
- Base URL: `http://localhost:8000`
- Vector backend: `redis`
- Redis host port: `16379`
- Container: `cad-ml-api` (restart only)

## Steps
- Baseline stats: `total=4`
- Insert 2 unique DXF stubs (cache_hit=false)
- After insert stats: `total=6`
- Restart `cad-ml-api`
- Post-restart stats: `total=6`

## Result
- ✅ Passed: vector totals preserved across restart (Redis persistence confirmed).
