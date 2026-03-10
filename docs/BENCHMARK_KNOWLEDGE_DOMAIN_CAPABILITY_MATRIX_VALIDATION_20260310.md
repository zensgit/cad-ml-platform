# Benchmark Knowledge Domain Capability Matrix Validation

## Goal

Add a benchmark control-plane component that exposes standards, tolerance, and
GD&T domain capability gaps across provider coverage, benchmark surfaces,
foundation/application readiness, and downstream release artifacts.

## Scope

Files changed in this layer:

- `src/core/benchmark/knowledge_domain_capability_matrix.py`
- `src/core/benchmark/__init__.py`
- `scripts/export_benchmark_knowledge_domain_capability_matrix.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_domain_capability_matrix.py`

## What Was Added

- New benchmark component: `knowledge_domain_capability_matrix`
- New exporter: `export_benchmark_knowledge_domain_capability_matrix.py`
- Downstream passthrough in:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Markdown rendering for domain-level capability status and recommendations
- Review-signal escalation when a domain remains provider/surface blocked

## Validation Commands

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_capability_matrix.py \
  scripts/export_benchmark_knowledge_domain_capability_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_capability_matrix.py

flake8 \
  src/core/benchmark/knowledge_domain_capability_matrix.py \
  scripts/export_benchmark_knowledge_domain_capability_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_capability_matrix.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_capability_matrix.py
```

## Expected Results

- New component reports:
  - `tolerance`
  - `standards`
  - `gdt`
- `gdt` remains benchmark-blocked when provider/public surface coverage is
  missing
- bundle / companion / release surfaces expose:
  - `knowledge_domain_capability_matrix_status`
  - domain-level rows
  - recommendations
- release decision and runbook include capability-matrix review signals

## Notes

- This layer intentionally exposes a current product gap:
  - `gdt` has assistant-side knowledge but no registered benchmark knowledge
    provider and no public benchmark-facing API surface
- That gap is now visible in benchmark outputs instead of staying implicit
