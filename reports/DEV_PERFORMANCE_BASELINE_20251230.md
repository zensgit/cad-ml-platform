# Performance Baseline Report

- Date: 2025-12-30
- Source: reports/performance_baseline_day0.json

## Benchmarks (p50 / p95)

| Benchmark | p50 | p95 |
| --- | --- | --- |
| feature_extraction_v3 | 1.27ms | 1.29ms |
| feature_extraction_v4 | 1.52ms | 1.53ms |
| batch_similarity_5ids | 5.95ms | 6.31ms |
| batch_similarity_20ids | 24.48ms | 25.03ms |
| batch_similarity_50ids | 53.89ms | 55.02ms |
| model_cold_load | 53.84ms | 55.03ms |

## Notes
- v4 overhead (p95): +18.4%
