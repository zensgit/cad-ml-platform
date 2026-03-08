# Benchmark Artifact Bundle Operator Adoption Validation

## Scope
- Extend benchmark artifact bundle with operator adoption input
- Track operator adoption artifact presence and readiness
- Allow operator adoption guidance to flow into bundle recommendations

## Files
- `scripts/export_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_artifact_bundle.py`

## Validation
```bash
python3 -m py_compile scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py
flake8 scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py --max-line-length=100
pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

## Result
- Artifact bundle now accepts `--benchmark-operator-adoption`
- `component_statuses.operator_adoption` is exported
- Operator adoption artifact is counted in available bundle artifacts
- Bundle recommendations can include operator adoption guidance
