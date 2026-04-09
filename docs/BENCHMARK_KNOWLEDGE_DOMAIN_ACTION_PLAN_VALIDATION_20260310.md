# Benchmark Knowledge Domain Action Plan Validation

## Scope

- add a benchmark-level `knowledge_domain_action_plan` component
- export standalone JSON/Markdown action plans from `knowledge_domain_matrix`
- propagate action-plan status, actions, and recommendations into:
  - competitive surpass index
  - benchmark artifact bundle
  - benchmark companion summary
  - benchmark release decision
  - benchmark release runbook

## Design

The new action-plan layer converts `knowledge_domain_matrix` output into staged,
domain-specific execution steps. Each domain can emit actions for:

- `foundation`
- `application`
- `realdata`

Each action includes:

- `id`
- `domain`
- `stage`
- `priority`
- `status`
- `action`

The top-level exporter emits:

- `status`
- `ready_domains`
- `partial_domains`
- `blocked_domains`
- `domain_action_counts`
- `total_action_count`
- `high_priority_action_count`
- `medium_priority_action_count`
- `priority_domains`
- `recommended_first_actions`
- `recommendations`

The first integration step also promotes this new signal into the knowledge
pillar of `competitive_surpass_index`, so benchmark readiness can distinguish
between "we know the gap" and "we have an executable plan for the gap".

## Files

- `src/core/benchmark/knowledge_domain_action_plan.py`
- `src/core/benchmark/__init__.py`
- `src/core/benchmark/competitive_surpass_index.py`
- `scripts/export_benchmark_knowledge_domain_action_plan.py`
- `scripts/export_benchmark_competitive_surpass_index.py`
- `scripts/export_benchmark_artifact_bundle.py`
- `scripts/export_benchmark_companion_summary.py`
- `scripts/export_benchmark_release_decision.py`
- `scripts/export_benchmark_release_runbook.py`
- `tests/unit/test_benchmark_knowledge_domain_action_plan.py`
- `tests/unit/test_benchmark_competitive_surpass_index.py`
- `tests/unit/test_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_companion_summary.py`
- `tests/unit/test_benchmark_release_decision.py`
- `tests/unit/test_benchmark_release_runbook.py`

## Verification

Commands run in isolated worktree:
`/private/tmp/cad-ml-platform-domain-action-plan-main-20260310`

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_domain_action_plan.py \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_knowledge_domain_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_knowledge_domain_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/knowledge_domain_action_plan.py \
  src/core/benchmark/competitive_surpass_index.py \
  scripts/export_benchmark_knowledge_domain_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  scripts/export_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_knowledge_domain_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_action_plan.py \
  tests/unit/test_benchmark_competitive_surpass_index.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

Results:

- `py_compile`: pass
- `flake8`: pass
- `pytest`: `34 passed`

## Expected Result

- standalone exporter writes valid JSON and Markdown action plans
- competitive surpass index reflects action-plan readiness in the knowledge pillar
- bundle / companion / release decision / release runbook expose:
  - `knowledge_domain_action_plan_status`
  - `knowledge_domain_action_plan_actions`
  - `knowledge_domain_action_plan_priority_domains`
  - `knowledge_domain_action_plan_recommendations`
- release surfaces can distinguish between missing knowledge signals and
  missing executable plans

## Limitations

- This change only covers exporter and downstream surfaces
- CI wiring and PR comment exposure stay in stacked follow-up branches
- Domain coverage still depends on upstream `knowledge_domain_matrix` artifacts
