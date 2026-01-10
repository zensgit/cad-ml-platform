# Dedup2D Load Test Report

- Date: 2025-12-27
- Scope: `/api/v1/dedup/2d/search` small-scale load

## Setup
- cad-ml-platform API: `http://localhost:8000`
- dedupcad-vision: `python start_server.py --port 58001`
- Fixture: `data/dxf_fixtures_subset_out/mixed.png`

## Command
- 5-minute async load (concurrency=4), `max_results=5`, `compute_diff=false`

## Result
- PASS (rate limiter engaged)

## Summary
- Duration: 300.02s
- Concurrency: 4
- Total requests: 40,295
- Success (200): 5,000
- Rate limited (429): 35,295
- Throughput: 134.31 rps (all), 16.67 rps (success-only)
- Latency (200 responses):
  - avg: 31.02 ms
  - p50: 30.12 ms
  - p95: 38.66 ms
  - p99: 50.8 ms
  - min/max: 10.35 ms / 969.88 ms

## Notes
- 429s are expected under load due to the API rate limiter; adjust `RATE_LIMIT_QPS`/`RATE_LIMIT_BURST`
  or reduce concurrency for higher success ratio in future runs.
