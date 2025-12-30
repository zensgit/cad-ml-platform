# Performance Baseline Comparison

- Date: 2025-12-30
- Day 0 baseline: 2025-12-30 19:59:44
- Day 6 baseline: 2025-12-22 14:47:30

## Summary
- Comparison uses p95 latency where available.
- Baseline scripts use synthetic timers; interpret deltas as relative trends only.

## Benchmarks
| Benchmark | Day 0 p95 | Day 6 p95 | Delta |
| --- | --- | --- | --- |
| feature_extraction_v3 | 1.29ms | 1.26ms | -2.5% |
| feature_extraction_v4 | 1.53ms | 1.51ms | -0.9% |
| batch_similarity_5ids | 6.31ms | 6.28ms | -0.5% |
| batch_similarity_20ids | 25.03ms | 25.03ms | -0.0% |
| batch_similarity_50ids | 55.02ms | 55.03ms | +0.0% |
| model_cold_load | 55.03ms | 55.03ms | -0.0% |
