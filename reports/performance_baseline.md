# Performance Baseline Report

Generated: 2026-04-09 12:20:49 UTC

**9/9 PASS** | 0 FAIL | 0 SKIP

## Results

| Module | Operation | Target (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Iters | Status |
|--------|-----------|-------------|----------|----------|----------|-----------|-------|--------|
| CostEstimator | estimate() | 100 | 0.01 | 0.01 | 0.02 | 0.01 | 200 | PASS |
| GraphQueryEngine | query() | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 200 | PASS |
| GraphQueryEngine | find_optimal_process() | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 200 | PASS |
| MetricsAnomalyDetector | detect() | 10 | 1.11 | 1.23 | 1.34 | 1.13 | 200 | PASS |
| HybridIntelligence | analyze_ensemble_uncertainty() | 5 | 0.00 | 0.00 | 0.01 | 0.00 | 200 | PASS |
| SmartSampler | combined_sampling(1000 -> 10) | 50 | 9.30 | 9.79 | 10.03 | 9.26 | 200 | PASS |
| GeometryDiff | compare() (small DXF) | 500 | 10.11 | 11.59 | 22.11 | 10.80 | 20 | PASS |
| PointCloudPreprocessor | normalize(2048 pts) | 5 | 0.04 | 0.04 | 0.05 | 0.04 | 200 | PASS |
| FunctionCallingEngine | __init__(offline) | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 200 | PASS |
