# Benchmark Operator Adoption Bundle / Companion Surfaces Validation

## Scope
- Extend benchmark artifact bundle with scorecard / operational operator adoption passthrough
- Extend benchmark companion summary with scorecard / operational operator adoption passthrough
- Expose the new data in markdown sections for operator review

## Changes
- Added scorecard operator adoption summary fields:
  - `status`
  - `operator_mode`
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
- Added operational operator adoption summary fields:
  - `status`
  - `knowledge_outcome_drift_status`
  - `knowledge_outcome_drift_summary`
- Added markdown sections:
  - `## Scorecard Operator Adoption`
  - `## Operational Operator Adoption`
- Added fallback logic so downstream `component_statuses.operator_adoption` can use
  scorecard / operational data when standalone operator adoption JSON is absent.

## Validation
```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  scripts/export_benchmark_companion_summary.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_companion_summary.py
```
