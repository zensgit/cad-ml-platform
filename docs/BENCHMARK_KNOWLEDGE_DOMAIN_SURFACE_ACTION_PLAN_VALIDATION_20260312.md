# Benchmark Knowledge Domain Surface Action Plan Validation

## Scope
- Added `knowledge_domain_surface_action_plan` benchmark component and exporter.
- Wired the action-plan payload into:
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook

## Key Outputs
- `status`
- `total_subcapability_count`
- `domain_action_counts`
- `ready_domains`
- `partial_domains`
- `blocked_domains`
- `priority_domains`
- `recommended_first_actions`
- `actions`

## Validation
Executed in `/private/tmp/cad-ml-platform-knowledge-domain-surface-action-plan-20260312`.

```bash
pytest -q tests/unit/test_benchmark_knowledge_domain_surface_action_plan.py \
  tests/unit/test_benchmark_knowledge_domain_surface_action_plan_surfaces.py -q
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_surface_action_plan.py \
  scripts/export_benchmark_knowledge_domain_surface_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_surface_action_plan.py \
  tests/unit/test_benchmark_knowledge_domain_surface_action_plan_surfaces.py
flake8 \
  src/core/benchmark/knowledge_domain_surface_action_plan.py \
  scripts/export_benchmark_knowledge_domain_surface_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_domain_surface_action_plan.py \
  tests/unit/test_benchmark_knowledge_domain_surface_action_plan_surfaces.py \
  --max-line-length=100
```

## Results
- `pytest`: `5 passed`
- `py_compile`: passed
- `flake8`: passed

## Fixes Included
- Added missing `benchmark_knowledge_domain_surface_action_plan` parameter to `build_bundle()`.
- Passed the new action-plan payload through `_component_statuses(...)`.
- Added missing `knowledge_domain_surface_action_plan_component` extraction in `build_release_decision()`.
- Prevented release/runbook prioritization regressions caused by incomplete downstream passthrough.
