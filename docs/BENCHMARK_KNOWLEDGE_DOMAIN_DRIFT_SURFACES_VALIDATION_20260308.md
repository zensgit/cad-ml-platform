# Benchmark Knowledge Domain Drift Surfaces Validation

## Scope

Promote knowledge domain drift from nested benchmark internals to stable
top-level fields in benchmark artifact surfaces.

## Delivered

- `export_benchmark_artifact_bundle.py`
- `export_benchmark_companion_summary.py`
- `export_benchmark_release_decision.py`
- `export_benchmark_release_runbook.py`

Each now emits:

- `knowledge_drift_domain_regressions`
- `knowledge_drift_domain_improvements`
- `knowledge_drift_resolved_priority_domains`
- `knowledge_drift_new_priority_domains`

The markdown outputs for these artifacts now also render the new domain-drift
fields directly.

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

Result:

- `py_compile` passed
- `flake8` passed
- `pytest` passed: `18 passed`
