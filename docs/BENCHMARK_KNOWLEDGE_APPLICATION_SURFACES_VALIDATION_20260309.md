# Benchmark Knowledge Application Surfaces Validation

## Scope

Wire `benchmark_knowledge_application` into benchmark release surfaces:

- companion summary
- artifact bundle
- release decision
- release runbook

## Design

`knowledge_application` is treated as a first-class benchmark component.

Downstream exporters now receive:

- `benchmark_knowledge_application`

and expose:

- `knowledge_application_status`
- `knowledge_application_domains`
- `knowledge_application_priority_domains`
- `knowledge_application_recommendations`

Release-facing behavior:

- companion summary includes `knowledge_application` in `component_statuses`
- artifact bundle includes `benchmark_knowledge_application` in artifacts and component rollup
- release decision includes `knowledge_application` in review signals
- release runbook includes `knowledge_application` in operator guidance and missing artifact checks

## Validation

Commands:

```bash
python3 -m py_compile \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `18 passed`

## Outcome

Benchmark knowledge readiness is no longer only a foundation metric. The new
knowledge-application signal now propagates into the companion, bundle,
decision, and runbook surfaces that operators actually read.
