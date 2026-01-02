# Dedup2D Staging Smoke Attempt (2026-01-01)

## Scope

- Attempt staging smoke checks from `docs/DEDUP2D_STAGING_RUNBOOK.md`.

## Attempts

- `curl -sS --max-time 3 http://localhost:58001/health`

## Result

- Blocked: `dedupcad-vision` not running on localhost:58001 (connection refused).
- No further API/worker smoke checks executed in this environment.
