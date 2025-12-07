# Performance Baseline (Phase 2)

Record environment, commit SHA, and baseline performance.

- Date:
- Commit: `git rev-parse --short HEAD`
- Python: `python --version`
- Host: CPU/RAM

## Workloads

1) v4 Feature Extraction
- Load tool: hey/ab/locust
- Parameters: concurrency, duration
- Metrics:
  - p50/p95/p99 latency
  - RPS
  - Error rate

2) Vector Operations (add/query)
- Synthetic vectors (24 dims)
- Metrics: throughput, p95

3) Memory Snapshot
- Process RSS, VSZ before/after load

## Results

| Metric | Value | Notes |
|-------|-------|-------|
| v4 p95 |       | histogram_quantile(0.95, feature_extraction_latency_seconds_bucket) |
| RPS    |       | |
| RSS    |       | ps output |

## Notes
- Capture `/metrics` snapshots and attach.
- Keep reproducible command lines used for tests.
