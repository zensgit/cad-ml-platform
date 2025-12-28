# Dedup2D Load Test Report (High QPS)

- Date: 2025-12-28
- Scope: `/api/v1/dedup/2d/search` with higher rate limit

## Setup
- cad-ml-platform API: `http://localhost:8001`
- dedupcad-vision: `python start_server.py --port 58001`
- Rate limit: `RATE_LIMIT_QPS=200`, `RATE_LIMIT_BURST=400`
- Fixture: `data/dxf_fixtures_subset_out/mixed.png`

## Changes Verified
- Map dedupcad-vision circuit open to HTTP 503 with `Retry-After`.
- Unit test coverage for circuit-open handling.

## Commands
- `.venv/bin/python -m pytest tests/test_dedup_2d_proxy.py -k circuit_open -q`
- 5-minute async load (concurrency=4), `max_results=5`, `compute_diff=false`

## Result
- PASS (circuit breaker engaged under sustained overload)

## Summary
- Duration: 300.0s
- Concurrency: 4
- Total requests: 226,239
- Success (200): 2,009
- Circuit open (503): 224,222
- Throughput: 754.13 rps (all), 6.7 rps (success-only)
- Latency (200 responses):
  - avg: 26.94 ms
  - p50: 24.69 ms
  - p95: 28.15 ms
  - p99: 58.75 ms
  - min/max: 8.57 ms / 967.22 ms

## Notes
- 503s indicate the dedupcad-vision circuit opened under sustained load; raise capacity or reduce
  concurrency for higher success ratios.
