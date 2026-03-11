# Benchmark Knowledge Domain Validation Matrix Validation

## Goal
- Add a benchmark component that measures whether `tolerance`, `standards`, and `gdt`
  are actually validation-ready across provider, API, unit, integration, and assistant
  retrieval layers.
- Feed that component into bundle, companion, release decision, and release runbook
  surfaces so benchmark control-plane views stop treating knowledge coverage as a single
  abstract status.

## Design
- Source of truth is repository reality, not synthetic fixtures.
- Each domain row reports:
  - provider readiness
  - public API surface count
  - unit / integration / assistant test counts
  - missing layers
  - recommended action
- Status semantics:
  - `knowledge_domain_validation_ready`
  - `knowledge_domain_validation_partial`
  - `knowledge_domain_validation_blocked`
- Current repo-derived expectation:
  - `tolerance`: ready
  - `standards`: ready
  - `gdt`: blocked
    - missing provider
    - missing public API surface
    - missing integration coverage

## Files
- `src/core/benchmark/knowledge_domain_validation_matrix.py`
- `scripts/export_benchmark_knowledge_domain_validation_matrix.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_domain_validation_matrix.py`

## Verification
```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_validation_matrix.py \
  scripts/export_benchmark_knowledge_domain_validation_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_validation_matrix.py

flake8 \
  src/core/benchmark/knowledge_domain_validation_matrix.py \
  scripts/export_benchmark_knowledge_domain_validation_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_validation_matrix.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_domain_validation_matrix.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- `pytest`: pass

## Outcome
- Benchmark control-plane now exposes a concrete standards/tolerance/GD&T validation
  matrix instead of only coarse readiness labels.
- Release surfaces now get domain-specific validation blockers and recommendations,
  with `gdt` correctly highlighted as the primary gap in the current repository state.
