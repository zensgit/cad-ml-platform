# Benchmark Knowledge Outcome Correlation Validation

## Goal

Add a benchmark component that correlates `knowledge_domain_matrix` readiness with
actual `realdata_scorecard` outcomes, then propagate that signal into:

- companion summary
- artifact bundle
- release decision
- release runbook

## Implementation

Primary files:

- `src/core/benchmark/knowledge_outcome_correlation.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_outcome_correlation.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_outcome_correlation.py`
- `tests/unit/test_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Design

The component maps knowledge domains to the real-data surfaces that matter most:

- `tolerance` -> `hybrid_dxf`, `history_h5`, `step_dir`
- `standards` -> `hybrid_dxf`, `history_h5`
- `gdt` -> `hybrid_dxf`, `step_smoke`, `step_dir`

Each domain now exposes:

- `matrix_status`
- `best_surface`
- `best_surface_score`
- `surface_scores`
- `surface_statuses`
- `ready_surfaces`
- `partial_surfaces`
- `weak_surfaces`
- `missing_surfaces`
- `focus_components`
- `missing_metrics`
- `action`

Top-level outputs:

- `knowledge_outcome_correlation_status`
- `knowledge_outcome_correlation_domains`
- `knowledge_outcome_correlation_priority_domains`
- `knowledge_outcome_correlation_recommendations`

## Validation Commands

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_outcome_correlation.py \
  scripts/export_benchmark_knowledge_outcome_correlation.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_outcome_correlation.py

flake8 \
  src/core/benchmark/knowledge_outcome_correlation.py \
  scripts/export_benchmark_knowledge_outcome_correlation.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_outcome_correlation.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_outcome_correlation.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Validation Result

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `21 passed`

## Outcome

`knowledge_outcome_correlation` is now a first-class benchmark component across the
core release-support surfaces. It is ready for the next stacked steps:

1. CI wiring
2. PR comment / signal light wiring
