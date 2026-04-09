# Benchmark Knowledge Real-Data Correlation Validation

## Scope

This delivery adds a standalone benchmark exporter that correlates domain-level
knowledge readiness and application status with real-data validation coverage.

The goal is to move beyond synthetic/operator-only benchmark summaries and show
whether `tolerance`, `standards`, and `gdt` are actually backed by DXF hybrid,
history-sequence, and STEP/B-Rep evidence.

## Changes

- Added reusable benchmark core module:
  - `src/core/benchmark/knowledge_realdata_correlation.py`
- Added standalone exporter:
  - `scripts/export_benchmark_knowledge_realdata_correlation.py`
- Exporter now reports:
  - `status`
  - `ready_domain_count`
  - `partial_domain_count`
  - `blocked_domain_count`
  - per-domain `readiness_status`
  - per-domain `application_status`
  - per-domain `realdata_status`
  - per-domain mapped real-data component statuses
  - domain-specific recommendations
- Added unit coverage for:
  - ready case
  - partial / blocked case
  - CLI output generation
  - Markdown rendering

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_realdata_correlation.py \
  scripts/export_benchmark_knowledge_realdata_correlation.py \
  tests/unit/test_benchmark_knowledge_realdata_correlation.py

flake8 \
  src/core/benchmark/knowledge_realdata_correlation.py \
  scripts/export_benchmark_knowledge_realdata_correlation.py \
  tests/unit/test_benchmark_knowledge_realdata_correlation.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_realdata_correlation.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: passed

## Notes

- This layer only introduces the standalone correlation artifact.
- Companion, artifact bundle, release decision, release runbook, CI, and PR
  comment wiring are intentionally left for the next stacked deliveries.
