# Benchmark Knowledge Domain Release Readiness Matrix Validation

## Goal
- Add a benchmark control-plane component that turns knowledge validation, release gate,
  reference inventory, and release-surface alignment into a per-domain release readiness view.
- Expose the same readiness signal through downstream benchmark surfaces so release decisions and
  runbooks can reason about `standards`, `tolerance`, and `gdt` using one normalized component.

## Design
- New core component:
  - `src/core/benchmark/knowledge_domain_release_readiness_matrix.py`
- New exporter:
  - `scripts/export_benchmark_knowledge_domain_release_readiness_matrix.py`
- Inputs:
  - `benchmark_knowledge_domain_validation_matrix`
  - `benchmark_knowledge_domain_release_gate`
  - `benchmark_knowledge_reference_inventory`
  - optional `benchmark_knowledge_domain_release_surface_alignment`
- Primary outputs:
  - `status`
  - `summary`
  - `ready_domain_count`
  - `partial_domain_count`
  - `blocked_domain_count`
  - `ready_domains`
  - `partial_domains`
  - `blocked_domains`
  - `releasable_domains`
  - `priority_domains`
  - `focus_areas_detail`
  - `domains`
  - `gate_open`
- Per-domain details include:
  - `validation_status`
  - `inventory_status`
  - `release_gate_status`
  - `alignment_warning`
  - `blocking_reasons`
  - `warning_reasons`
  - `action`

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
- `knowledge_domain_release_readiness_matrix_status`
- `knowledge_domain_release_readiness_matrix_summary`
- `knowledge_domain_release_readiness_matrix_priority_domains`
- `knowledge_domain_release_readiness_matrix_releasable_domains`
- `knowledge_domain_release_readiness_matrix_blocked_domains`
- `knowledge_domain_release_readiness_matrix_focus_areas_detail`
- `knowledge_domain_release_readiness_matrix_recommendations`

## Tests
- New unit coverage:
  - `tests/unit/test_benchmark_knowledge_domain_release_readiness_matrix.py`
- Extended downstream coverage:
  - `tests/unit/test_benchmark_artifact_bundle.py`
  - `tests/unit/test_benchmark_companion_summary.py`
  - `tests/unit/test_benchmark_release_decision.py`
  - `tests/unit/test_benchmark_release_runbook.py`

## Validation Commands
```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_release_readiness_matrix.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_matrix.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_release_readiness_matrix.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_matrix.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_matrix.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_matrix.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Results
- `py_compile`: passed
- `pytest`: `46 passed, 1 warning`
- `flake8`: passed

## Notes
- The warning is an existing upstream `PendingDeprecationWarning` from `starlette` multipart
  parsing and is unrelated to this feature.
- This branch stays isolated in `/private/tmp/cad-ml-platform-knowledge-domain-release-readiness-20260311`
  and does not modify the dirty main worktree.
