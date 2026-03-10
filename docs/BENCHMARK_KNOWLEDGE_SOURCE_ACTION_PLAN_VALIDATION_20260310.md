# Benchmark Knowledge Source Action Plan Validation

Date: 2026-03-10

## Scope

This change introduces a new benchmark control-plane component:
`knowledge_source_action_plan`.

It adds:

- a standalone exporter
- benchmark action synthesis from `knowledge_source_coverage`
- `competitive_surpass_index` integration so source coverage gaps and expansion
  opportunities affect the knowledge pillar

## Files

- `src/core/benchmark/knowledge_source_action_plan.py`
- `src/core/benchmark/__init__.py`
- `src/core/benchmark/competitive_surpass_index.py`
- `scripts/export_benchmark_knowledge_source_action_plan.py`
- `scripts/export_benchmark_competitive_surpass_index.py`
- `tests/unit/test_benchmark_knowledge_source_action_plan.py`
- `tests/unit/test_benchmark_competitive_surpass_index.py`

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_source_action_plan.py \
  scripts/export_benchmark_knowledge_source_action_plan.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py

flake8 \
  src/core/benchmark/knowledge_source_action_plan.py \
  scripts/export_benchmark_knowledge_source_action_plan.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py
```

Results:

- `py_compile` passed
- `flake8` passed
- `pytest` passed: `6 passed`

## Notes

- `knowledge_source_action_plan` turns source coverage into executable actions:
  - source-group coverage fixes
  - domain-level closure steps
  - ready manufacturing expansion promotions
- `competitive_surpass_index` now treats source action planning as part of the
  knowledge pillar, instead of only observing raw source coverage.
