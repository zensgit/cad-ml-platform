# Benchmark Knowledge Source Drift Validation

## Scope

- Added `knowledge_source_drift` benchmark component and exporter.
- Threaded `knowledge_source_drift` through:
  - artifact bundle
  - companion summary
  - release decision
  - release runbook
- Added passthrough unit coverage for each downstream surface.

## Key Outputs

- `status`
- `current_status`
- `previous_status`
- `source_group_regressions`
- `source_group_improvements`
- `resolved_priority_domains`
- `new_priority_domains`
- `recommendations`

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_knowledge_source_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py

flake8 \
  scripts/export_benchmark_knowledge_source_drift.py \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  scripts/export_benchmark_release_decision.py \
  scripts/export_benchmark_release_runbook.py \
  tests/unit/test_benchmark_knowledge_source_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_knowledge_source_drift.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  tests/unit/test_benchmark_release_decision.py \
  tests/unit/test_benchmark_release_runbook.py
```

## Expected Result

- `knowledge_source_drift` exports standalone JSON/markdown.
- All benchmark downstream surfaces preserve the drift payload and recommendations.
- Release decision and release runbook elevate regressed source coverage into review signals.
