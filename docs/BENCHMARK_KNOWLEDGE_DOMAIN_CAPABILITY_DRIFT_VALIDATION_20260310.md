# Benchmark Knowledge Domain Capability Drift Validation

## Scope

- Added standalone `knowledge_domain_capability_drift` benchmark component.
- Compared current and previous `knowledge_domain_capability_matrix` baselines.
- Passed drift status into bundle, companion, release decision, and release runbook surfaces.

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_capability_drift.py \
  scripts/export_benchmark_knowledge_domain_capability_drift.py \
  tests/unit/test_benchmark_knowledge_domain_capability_drift.py

flake8 \
  src/core/benchmark/knowledge_domain_capability_drift.py \
  scripts/export_benchmark_knowledge_domain_capability_drift.py \
  tests/unit/test_benchmark_knowledge_domain_capability_drift.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_capability_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Expected Outcome

- `knowledge_domain_capability_drift.status` reports `baseline_missing / stable / improved / regressed / mixed`.
- Drift payload exposes domain-level regressions and provider/surface gap deltas.
- Bundle, companion, release decision, and release runbook all expose the new drift status and recommendations.

## Result

- `python3 -m py_compile` passed
- `flake8 --max-line-length=100` passed
- `pytest -q ...` passed: `39 passed, 1 warning`
