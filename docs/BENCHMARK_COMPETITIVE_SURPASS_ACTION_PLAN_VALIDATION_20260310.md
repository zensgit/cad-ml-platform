# Benchmark Competitive Surpass Action Plan Validation

## Scope

- add standalone `competitive_surpass_action_plan` exporter
- expose action-plan passthrough in:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook

## Key Outputs

- `competitive_surpass_action_plan.status`
- `competitive_surpass_action_plan_total_action_count`
- `competitive_surpass_action_plan_high_priority_action_count`
- `competitive_surpass_action_plan_medium_priority_action_count`
- `competitive_surpass_action_plan_priority_pillars`
- `competitive_surpass_action_plan_recommended_first_actions`
- `competitive_surpass_action_plan_recommendations`

## Validation

```bash
python3 -m py_compile \
  src/core/benchmark/competitive_surpass_action_plan.py \
  scripts/export_benchmark_competitive_surpass_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_competitive_surpass_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  src/core/benchmark/competitive_surpass_action_plan.py \
  scripts/export_benchmark_competitive_surpass_action_plan.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_competitive_surpass_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_competitive_surpass_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

git diff --check
```

## Notes

- action-plan recommendations are promoted into release decision and runbook review signals
- standalone exporter and downstream surfaces now share the same action-plan schema
