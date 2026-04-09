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
- downstream benchmark surface integration for:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Files

- `src/core/benchmark/knowledge_source_action_plan.py`
- `src/core/benchmark/__init__.py`
- `src/core/benchmark/competitive_surpass_index.py`
- `scripts/export_benchmark_knowledge_source_action_plan.py`
- `scripts/export_benchmark_competitive_surpass_index.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_source_action_plan.py`
- `tests/unit/test_benchmark_competitive_surpass_index.py`
- `tests/unit/test_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_source_action_plan.py \
  scripts/export_benchmark_knowledge_source_action_plan.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_source_action_plan.py \
  scripts/export_benchmark_knowledge_source_action_plan.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_source_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile` passed
- `flake8` passed
- `pytest` passed: `34 passed`

## Notes

- `knowledge_source_action_plan` turns source coverage into executable actions:
  - source-group coverage fixes
  - domain-level closure steps
  - ready manufacturing expansion promotions
- `competitive_surpass_index` now treats source action planning as part of the
  knowledge pillar, instead of only observing raw source coverage.
- all downstream benchmark control-plane surfaces now carry:
  - status
  - action counts
  - priority domains
  - recommended first actions
  - compact recommendations
