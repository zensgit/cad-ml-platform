# Memory Stability Load Test (1 Hour)

- Date: 2025-12-27
- Scope: Sustained load + memory sampling on `cad-ml-api`
- Duration: 3600s (1 hour)

## Command
- .venv/bin/python scripts/memory_stability_check.py --duration-seconds 3600 --interval-seconds 60 --concurrency 10 --output-csv reports/memory_stability_samples_20251227_1h.csv --summary-json reports/memory_stability_summary_20251227_1h.json

## Result
- PASS

## Summary
- Requests: 35,240
- Error rate: 0.00%
- Latency p50/p95/p99 (ms): 14.84 / 21.60 / 25.61
- Memory MiB min/avg/max: 280.4 / 286.84 / 292.3
- Memory % min/avg/max: 3.58 / 3.66 / 3.73

## Artifacts (ignored by git)
- reports/memory_stability_samples_20251227_1h.csv
- reports/memory_stability_summary_20251227_1h.json
