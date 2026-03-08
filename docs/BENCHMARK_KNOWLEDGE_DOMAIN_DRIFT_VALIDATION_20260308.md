# Benchmark Knowledge Domain Drift Validation

## Scope

Extend benchmark knowledge drift from component-only changes to domain-level
signals that are meaningful to release and benchmark operators.

## Delivered

- `build_knowledge_drift_status()` now exposes:
  - `domain_regressions`
  - `domain_improvements`
  - `resolved_priority_domains`
  - `new_priority_domains`
  - `domain_changes`
- drift recommendations now mention regressed/improved domains and resolved/new
  priority domains
- markdown rendering now includes a `Domain Changes` section

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_drift.py \
  scripts/export_benchmark_knowledge_drift.py \
  tests/unit/test_benchmark_knowledge_drift.py

flake8 \
  src/core/benchmark/knowledge_drift.py \
  scripts/export_benchmark_knowledge_drift.py \
  tests/unit/test_benchmark_knowledge_drift.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_drift.py
```

Result:

- `py_compile` passed
- `flake8` passed
- `pytest` passed: `3 passed`

## Notes

- Empty `priority_domains` is preserved as an empty operator signal and no
  longer falls back to all known domains.
- Domain drift is derived from `knowledge_readiness.domains` when present and
  falls back to computed domain statuses when only component data exists.
