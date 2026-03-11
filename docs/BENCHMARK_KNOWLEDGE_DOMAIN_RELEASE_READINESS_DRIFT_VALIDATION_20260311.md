# Benchmark Knowledge Domain Release Readiness Drift Validation

## Goal
- Add a benchmark control-plane component that compares the current and previous
  `knowledge_domain_release_readiness_matrix` baselines.
- Surface regressions and improvements for `standards`, `tolerance`, and `gdt`
  through downstream benchmark release surfaces without breaking existing
  ready-to-freeze behavior.

## Design
- New core component:
  - `src/core/benchmark/knowledge_domain_release_readiness_drift.py`
- New exporter:
  - `scripts/export_benchmark_knowledge_domain_release_readiness_drift.py`
- Inputs:
  - `current_summary`
  - `previous_summary`
- Primary outputs:
  - `status`
  - `summary`
  - `current_status`
  - `previous_status`
  - `ready_domain_delta`
  - `partial_domain_delta`
  - `blocked_domain_delta`
  - `regressions`
  - `improvements`
  - `resolved_priority_domains`
  - `new_priority_domains`
  - `resolved_releasable_domains`
  - `new_releasable_domains`
  - `resolved_blocked_domains`
  - `new_blocked_domains`
  - `domain_regressions`
  - `domain_improvements`
  - `domain_changes`

## Downstream Wiring
- Artifact bundle:
  - `scripts/export_benchmark_artifact_bundle.py`
- Companion summary:
  - `scripts/export_benchmark_companion_summary.py`
- Release decision:
  - `scripts/export_benchmark_release_decision.py`
- Release runbook:
  - `scripts/export_benchmark_release_runbook.py`

Each downstream surface now carries:
- `knowledge_domain_release_readiness_drift_status`
- `knowledge_domain_release_readiness_drift_summary`
- `knowledge_domain_release_readiness_drift_domain_regressions`
- `knowledge_domain_release_readiness_drift_domain_improvements`
- `knowledge_domain_release_readiness_drift_recommendations`

## Tests
- New unit coverage:
  - `tests/unit/test_benchmark_knowledge_domain_release_readiness_drift.py`
- Extended downstream coverage:
  - `tests/unit/test_benchmark_artifact_bundle.py`
  - `tests/unit/test_benchmark_companion_summary.py`
  - `tests/unit/test_benchmark_release_decision.py`
  - `tests/unit/test_benchmark_release_runbook.py`

## Validation Commands
```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_release_readiness_drift.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_release_readiness_drift.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_drift.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Results
- `py_compile`: passed
- `flake8`: passed
- `pytest`: `46 passed, 1 warning`

## Notes
- The warning is the existing `PendingDeprecationWarning` from `starlette`
  multipart parsing and is unrelated to this feature.
- `knowledge_domain_release_readiness_drift` is treated as an optional release
  artifact. It propagates to release surfaces when present, but it does not
  block freeze readiness when omitted from an older ready baseline.
- This branch stays isolated in
  `/private/tmp/cad-ml-platform-knowledge-domain-release-readiness-drift-20260311`
  and does not modify the dirty main worktree.
