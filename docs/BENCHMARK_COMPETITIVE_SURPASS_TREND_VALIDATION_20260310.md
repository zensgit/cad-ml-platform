# Benchmark Competitive Surpass Trend Validation

## Scope
- Added `competitive_surpass_trend` as a standalone benchmark exporter.
- Wired the trend into:
  - `benchmark artifact bundle`
  - `benchmark companion summary`
  - `benchmark release decision`
  - `benchmark release runbook`

## Output Contract
- `status`
- `score_delta`
- `pillar_improvements`
- `pillar_regressions`
- `resolved_primary_gaps`
- `new_primary_gaps`
- `recommendations`

## Validation
- `python3 -m py_compile src/core/benchmark/competitive_surpass_trend.py scripts/export_benchmark_competitive_surpass_trend.py scripts/export_benchmark_artifact_bundle.py scripts/export_benchmark_companion_summary.py scripts/export_benchmark_release_decision.py scripts/export_benchmark_release_runbook.py tests/unit/test_benchmark_competitive_surpass_trend.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_release_decision.py tests/unit/test_benchmark_release_runbook.py`
- `flake8 src/core/benchmark/competitive_surpass_trend.py scripts/export_benchmark_competitive_surpass_trend.py scripts/export_benchmark_artifact_bundle.py scripts/export_benchmark_companion_summary.py scripts/export_benchmark_release_decision.py scripts/export_benchmark_release_runbook.py tests/unit/test_benchmark_competitive_surpass_trend.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_release_decision.py tests/unit/test_benchmark_release_runbook.py --max-line-length=100`
- `pytest -q tests/unit/test_benchmark_competitive_surpass_trend.py tests/unit/test_benchmark_artifact_bundle.py tests/unit/test_benchmark_companion_summary.py tests/unit/test_benchmark_release_decision.py tests/unit/test_benchmark_release_runbook.py`
- `git diff --check`

## Expected Result
- Current and previous `competitive_surpass_index` baselines produce a stable trend payload.
- Bundle / companion / release decision / release runbook expose the same trend contract.
- Regressed and mixed trend states contribute review signals instead of being silent.
