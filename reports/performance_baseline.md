# Performance Baseline Report (Day 6)

## Scope
- Capture current baseline metrics using `scripts/performance_baseline.py` logic.
- Compare with Day 0 baseline in `reports/performance_baseline_day0.json`.

## Artifacts
- Day 0 baseline: `reports/performance_baseline_day0.json`
- Day 6 baseline: `reports/performance_baseline_day6.json`

## Run
- Command (programmatic reuse of baseline functions):
  - `.venv/bin/python - <<'PY' ...` (generated `performance_baseline_day6.json`)

## Comparison (p95)
- feature_extraction_v3: 1.28ms → 1.26ms (‑1.5%)
- feature_extraction_v4: 1.54ms → 1.51ms (‑1.9%)
- batch_similarity_5ids: 6.30ms → 6.28ms (‑0.3%)
- batch_similarity_20ids: 27.61ms → 25.03ms (‑9.4%)
- batch_similarity_50ids: 55.05ms → 55.03ms (‑0.0%)
- model_cold_load: 55.04ms → 55.03ms (‑0.0%)

## Notes
- `scripts/performance_baseline.py` uses stubbed delays to simulate workload; results reflect relative comparisons, not production latency.
