# Benchmark Knowledge Outcome Drift Surfaces Validation

## Goal

Extend `benchmark_knowledge_outcome_drift` beyond its standalone exporter so the
same signal is available in downstream benchmark surfaces:

- artifact bundle
- companion summary
- release decision
- release runbook

The new surface keeps the existing benchmark pattern:

- component status row
- structured payload passthrough
- summary text
- recommendations passthrough
- artifact presence tracking

## Design

The implementation mirrors the existing `knowledge_drift` and
`knowledge_outcome_correlation` surfaces.

Added support includes:

- new component status key: `knowledge_outcome_drift`
- new artifact key: `benchmark_knowledge_outcome_drift`
- new payload fields:
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift`
  - `knowledge_outcome_drift_summary`
  - `knowledge_outcome_drift_domain_regressions`
  - `knowledge_outcome_drift_domain_improvements`
  - `knowledge_outcome_drift_resolved_priority_domains`
  - `knowledge_outcome_drift_new_priority_domains`
  - `knowledge_outcome_drift_recommendations`

Decision semantics:

- `release_decision` and `release_runbook` expose outcome drift as a review /
  guidance signal
- `knowledge_outcome_drift` is not added as a blocking release component in the
  same way as core execution blockers
- runbook now treats missing `benchmark_knowledge_outcome_drift` as a genuine
  missing artifact, so tests and CLI fixtures were updated accordingly

## Files

- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Validation

Commands:

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: passed
- `flake8`: passed
- `pytest`: `21 passed`

## Outcome

`knowledge outcome drift` is now a first-class benchmark signal across bundle,
companion, release decision, and release runbook surfaces, with unit coverage
for each layer and CLI fixture updates where artifact completeness matters.
