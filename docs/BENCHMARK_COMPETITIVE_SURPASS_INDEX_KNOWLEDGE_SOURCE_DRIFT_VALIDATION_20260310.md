# Benchmark Competitive Surpass Index Knowledge Source Drift Validation

## Scope
- Add `benchmark_knowledge_source_drift` as an explicit knowledge input to
  `competitive_surpass_index`.
- Make the knowledge pillar summary and details reflect source-drift state.
- Extend the standalone exporter CLI and artifact wiring to accept
  `--benchmark-knowledge-source-drift`.

## Key Changes
- `src/core/benchmark/competitive_surpass_index.py`
  - knowledge pillar now reads `knowledge_source_drift`
  - `component_statuses.source_drift` is included in pillar details
  - summary now includes `source_drift=...`
  - details now expose source-group regressions/improvements
- `scripts/export_benchmark_competitive_surpass_index.py`
  - new CLI flag: `--benchmark-knowledge-source-drift`
  - new artifact path passthrough
- `tests/unit/test_benchmark_competitive_surpass_index.py`
  - ready-case and degraded-case fixtures now include source-drift input
  - assertions cover `source_drift` propagation

## Validation
```bash
python3 -m py_compile \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_competitive_surpass_index.py

flake8 \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_competitive_surpass_index.py

git diff --check
```

## Result
- `competitive_surpass_index` can now distinguish stable vs regressed knowledge
  source state inside the knowledge pillar.
- The exporter CLI accepts source-drift JSON directly, so downstream benchmark
  surfaces can consume the same higher-level assessment without custom glue.
