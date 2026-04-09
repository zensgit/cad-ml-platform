# Benchmark Knowledge Drift Bundle Validation

Date: 2026-03-08

## Scope

This change wires `scripts/export_benchmark_knowledge_drift.py` output into:

- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`

No workflow, release decision, or release runbook files were modified.

## Contract Added

Both exporters now accept:

- `--benchmark-knowledge-drift`

Both outputs now expose knowledge drift through:

- `component_statuses.knowledge_drift`
- `artifacts.benchmark_knowledge_drift`
- `knowledge_drift`
- `knowledge_drift_summary`
- `knowledge_drift_recommendations`
- `knowledge_drift_component_changes`

Behavior notes:

- artifact bundle keeps existing overall-status precedence unchanged
- artifact bundle uses knowledge-drift recommendations only as a fallback after release, companion, operational, and scorecard recommendations are absent
- companion summary preserves existing review-surface behavior, but now surfaces drift in `primary_gap` when the drift status is `baseline_missing`, `regressed`, or `mixed`
- companion summary uses drift recommendations as a fallback after bundle and operational recommendations are absent

## Validation

Commands run in `/private/tmp/cad-ml-platform-benchmark-knowledge-drift-bundle`:

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```

Result: passed

```bash
flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  --max-line-length=100
```

Result: passed

```bash
pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```

Result: `10 passed, 1 warning in 1.43s`

Known warning:

- `starlette.formparsers` emits a `PendingDeprecationWarning` for `multipart`; this is pre-existing and out of scope for this change
