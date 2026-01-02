# DedupCAD Vision Resilience & Observability Report

- Date: 2025-12-27
- Scope: dedupcad-vision client resilience (retry + circuit breaker) and metrics

## Changes
- Added retry/backoff and circuit breaker handling for dedupcad-vision calls.
- Added Prometheus metrics for dedupcad-vision requests, errors, retries, latency, and circuit state.

## Commands
- pytest tests/integration/test_dedupcad_vision_contract.py -q
- pytest tests/integration/test_e2e_api_smoke.py::test_e2e_dedup_search_smoke -q

## Result
- PASS

## Summary
- Contract tests: 2 passed in 1.52s
- E2E dedup search smoke: 1 passed in 0.66s
