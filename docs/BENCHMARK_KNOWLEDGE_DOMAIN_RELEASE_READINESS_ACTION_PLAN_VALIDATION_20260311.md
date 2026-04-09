# Benchmark Knowledge Domain Release Readiness Action Plan Validation

## Scope

- Add a standalone `knowledge_domain_release_readiness_action_plan` benchmark component.
- Propagate the component into:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Expose stable fields and markdown sections for downstream CI and review surfaces.

## Key Changes

- Added:
  - `src/core/benchmark/knowledge_domain_release_readiness_action_plan.py`
  - `scripts/export_benchmark_knowledge_domain_release_readiness_action_plan.py`
  - `tests/unit/test_benchmark_knowledge_domain_release_readiness_action_plan.py`
- Extended downstream exporters:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
  - `scripts/export_benchmark_release_decision.py`
  - `scripts/export_benchmark_release_runbook.py`
- Added stable payload fields:
  - `knowledge_domain_release_readiness_action_plan_status`
  - `knowledge_domain_release_readiness_action_plan_total_action_count`
  - `knowledge_domain_release_readiness_action_plan_high_priority_action_count`
  - `knowledge_domain_release_readiness_action_plan_medium_priority_action_count`
  - `knowledge_domain_release_readiness_action_plan_gate_open`
  - `knowledge_domain_release_readiness_action_plan_actions`
  - `knowledge_domain_release_readiness_action_plan_priority_domains`
  - `knowledge_domain_release_readiness_action_plan_recommended_first_actions`
  - `knowledge_domain_release_readiness_action_plan_recommendations`

## Validation Commands

```bash
python3 -m py_compile \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_action_plan.py \
  src/core/benchmark/knowledge_domain_release_readiness_action_plan.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_knowledge_domain_release_readiness_action_plan.py \
  src/core/benchmark/knowledge_domain_release_readiness_action_plan.py \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_domain_release_readiness_action_plan.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

git diff --check
```

## Expected Results

- New action-plan exporter builds blocked / partial / ready outcomes from matrix, drift, and gate inputs.
- Bundle / companion / release decision / release runbook all expose the same action-plan status and recommendations.
- Rendered markdown includes `## Knowledge Domain Release Readiness Action Plan`.
- No syntax, lint, or changed-line whitespace issues.

## Notes

- This line is intentionally kept at the exporter/surfaces layer first.
- CI wiring and PR comment wiring should be stacked afterward to avoid repeated workflow conflict churn.
