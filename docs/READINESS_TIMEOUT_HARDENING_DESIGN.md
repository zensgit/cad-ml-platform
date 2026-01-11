# Readiness Timeout Hardening Design

## Scope
- Add timeouts to readiness dependency checks to avoid hanging `/ready`.
- Return structured readiness payloads with per-check details.
- Log failures with clear reasons and return 503 when not ready.

## Problem Statement
- `/ready` could block indefinitely when a dependency check stalled.
- Failures returned generic 503 responses without structured detail.

## Design
- Add `ReadinessResponse` and `ReadinessCheck` schemas in `src/api/health_models.py`.
- Implement `_run_readiness_check` using `asyncio.wait_for` and `READINESS_CHECK_TIMEOUT_SECONDS`.
- Return a consistent payload and set the response status to 503 when any required check fails.
- Emit structured logs listing failing checks and reasons.

## Impact
- Readiness checks return promptly even when dependencies are slow or unavailable.
- Operators receive structured diagnostics in the response payload.

## Validation
- `python3 -m pytest tests/unit/test_main_coverage.py -k readiness -v`
