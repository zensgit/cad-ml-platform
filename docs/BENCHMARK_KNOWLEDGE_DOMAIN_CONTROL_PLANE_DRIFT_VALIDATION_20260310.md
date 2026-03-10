# Benchmark Knowledge Domain Control Plane Drift Validation

## Goal
- Add `knowledge_domain_control_plane_drift` as a benchmark component.
- Propagate it through bundle, companion, release decision, and release runbook.
- Keep the new drift artifact ready for the next CI / PR-comment stacked layers.

## Implemented
- Added core drift helper:
  - `src/core/benchmark/knowledge_domain_control_plane_drift.py`
- Added standalone exporter:
  - `scripts/export_benchmark_knowledge_domain_control_plane_drift.py`
- Extended downstream surfaces:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Exported the new helper set from:
  - `src/core/benchmark/__init__.py`

## Drift Outputs
- `status`
- `current_status`
- `previous_status`
- `ready_domain_delta`
- `partial_domain_delta`
- `blocked_domain_delta`
- `missing_domain_delta`
- `total_action_delta`
- `high_priority_action_delta`
- `regressions`
- `improvements`
- `resolved_release_blockers`
- `new_release_blockers`
- `domain_regressions`
- `domain_improvements`
- `domain_changes`

## Surface Outputs
- Bundle / Companion:
  - `knowledge_domain_control_plane_drift_status`
  - `knowledge_domain_control_plane_drift`
  - `knowledge_domain_control_plane_drift_domain_regressions`
  - `knowledge_domain_control_plane_drift_domain_improvements`
  - `knowledge_domain_control_plane_drift_recommendations`
- Release Decision / Runbook also expose:
  - `knowledge_domain_control_plane_drift_new_release_blockers`
  - `knowledge_domain_control_plane_drift_resolved_release_blockers`

## Validation
Commands run:

```bash
cd /private/tmp/cad-ml-platform-knowledge-domain-control-plane-drift-20260310
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_control_plane_drift.py \
  scripts/export_benchmark_knowledge_domain_control_plane_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_control_plane_drift.py \
  scripts/export_benchmark_knowledge_domain_control_plane_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_control_plane_drift.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_control_plane.py \
  tests/unit/test_benchmark_knowledge_domain_capability_drift.py \
  tests/unit/test_benchmark_knowledge_domain_control_plane_drift.py

git diff --check
```

Expected validation result for this stack:
- existing control-plane and capability-drift tests still pass
- new control-plane-drift tests pass
- exporter stack remains importable and lint-clean

## Notes
- This branch intentionally stops at the exporter/surfaces layer.
- CI wiring and PR-comment wiring should be added as the next stacked PRs after this branch is green.
