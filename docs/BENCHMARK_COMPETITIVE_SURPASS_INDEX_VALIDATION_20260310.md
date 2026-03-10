# Benchmark Competitive Surpass Index Validation

## Scope

- add unified `competitive_surpass_index` benchmark component
- export standalone JSON/Markdown summary
- propagate status, gaps, and recommendations into:
  - benchmark artifact bundle
  - benchmark companion summary

## Design

The index summarizes five benchmark pillars:

- `engineering`
- `knowledge`
- `realdata`
- `operator_adoption`
- `release_alignment`

Each pillar is normalized to `ready`, `partial`, or `blocked`. The exporter emits:

- `status`
- `score`
- `ready_pillars`
- `partial_pillars`
- `blocked_pillars`
- `primary_gaps`
- `recommendations`

The first integration step intentionally stops at `bundle` and `companion` to avoid
introducing cyclic dependencies with release decision and release runbook.

## Files

- `src/core/benchmark/competitive_surpass_index.py`
- `scripts/export_benchmark_competitive_surpass_index.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_competitive_surpass_index.py`
- `tests/unit/test_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_companion_summary.py`

## Verification

```bash
python3 -m py_compile \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py

flake8 \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```

## Expected Result

- standalone exporter writes valid JSON/Markdown
- bundle exposes:
  - `competitive_surpass_index_status`
  - `competitive_surpass_primary_gaps`
  - `competitive_surpass_recommendations`
- companion exposes the same fields and renders a dedicated markdown section
