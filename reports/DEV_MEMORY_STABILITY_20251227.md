# Memory Stability Load Test (Dev Sample)

- Date: 2025-12-27
- Scope: Sustained load + memory sampling on `cad-ml-api`
- Duration: 600s (10 minutes) sample run

## Command
- .venv/bin/python scripts/memory_stability_check.py --duration-seconds 600 --interval-seconds 60 --concurrency 10 --output-csv reports/memory_stability_samples_20251227.csv --summary-json reports/memory_stability_summary_20251227.json

## Result
- PASS (sample run)

## Summary
- Requests: 5,190
- Error rate: 1.93% (100 errors)
- Latency p50/p95/p99 (ms): 12.39 / 25.64 / 3002.10
- Memory MiB min/avg/max: 286.4 / 292.3 / 311.4
- Memory % min/avg/max: 3.65 / 3.73 / 3.97

## Artifacts
- reports/memory_stability_samples_20251227.csv
- reports/memory_stability_summary_20251227.json

## Note
- For production verification, extend to 1 hour as per PRODUCTION_VERIFICATION_PLAN.
