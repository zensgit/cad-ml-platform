# Benchmark Knowledge Focus Areas Validation

## Goal

Upgrade benchmark knowledge readiness from a single readiness status into an
actionable gap surface that can drive companion, bundle, release-decision, and
runbook outputs.

## Design

- Extend `knowledge_readiness` with `focus_areas_detail`.
- Each focus area now carries:
  - `component`
  - `status`
  - `priority`
  - `ready_metric_count`
  - `missing_metric_count`
  - `total_metric_count`
  - `total_reference_items`
  - `missing_metrics`
  - `action`
- Preserve the detailed focus-area payload through:
  - benchmark scorecard
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Files

- `src/core/benchmark/knowledge_readiness.py`
- `src/core/benchmark/__init__.py`
- `scripts/generate_benchmark_scorecard.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_readiness.py`
- `tests/unit/test_generate_benchmark_scorecard.py`
- `tests/unit/test_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_readiness.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/generate_benchmark_scorecard.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_readiness.py \
  src/core/benchmark/__init__.py \
  scripts/export_benchmark_knowledge_readiness.py \
  scripts/generate_benchmark_scorecard.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_knowledge_readiness.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `25 passed, 1 warning`

## Outcome

- Benchmark knowledge readiness now exposes a stable, actionable gap structure.
- Companion, release, and runbook surfaces can point to concrete knowledge
  backlog areas instead of only showing `knowledge_foundation_partial`.
