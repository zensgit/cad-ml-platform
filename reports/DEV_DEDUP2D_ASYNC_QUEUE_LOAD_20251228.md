# Dedup2D Async Queue Load Report

- Date: 2025-12-28
- Scope: `/api/v1/dedup/2d/search?async=true` backpressure behavior

## Setup
- cad-ml-platform API: `http://localhost:8002`
- dedupcad-vision: `python start_server.py --port 58001`
- Rate limit: `RATE_LIMIT_QPS=1000`, `RATE_LIMIT_BURST=2000`
- Async config: `DEDUP2D_ASYNC_MAX_CONCURRENCY=2`, `DEDUP2D_ASYNC_MAX_JOBS=200`
- Fixture: `data/dxf_fixtures_subset_out/mixed.png`

## Command
- 5-minute async load (concurrency=4), `max_results=5`

## Result
- PASS (queue backpressure observed)

## Summary
- Duration: 300.0s
- Concurrency: 4
- Total requests: 269,053
- Accepted (200): 5,200
- Queue full (429): 263,853
- Throughput: 896.84 rps (all), 17.33 rps (accepted-only)
- Latency (200 responses):
  - avg: 5.61 ms
  - p50: 5.4 ms
  - p95: 7.29 ms
  - p99: 9.33 ms
  - min/max: 2.24 ms / 48.67 ms

## Notes
- 429 responses were all `JOB_QUEUE_FULL`, confirming queue backpressure behavior under load.
