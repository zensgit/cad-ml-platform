# Benchmark Operator Outcome Drift Bundle Companion Validation

## Goal

Expose `operator_adoption_knowledge_outcome_drift` in benchmark artifact bundle and
benchmark companion summary so downstream benchmark surfaces can consume operator outcome
regressions consistently.

## Changes

- Added `operator_adoption_knowledge_outcome_drift` helper + payload passthrough in:
  - `scripts/export_benchmark_artifact_bundle.py`
  - `scripts/export_benchmark_companion_summary.py`
- Extended markdown rendering for both outputs with a dedicated
  `Operator Adoption Knowledge Outcome Drift` section.
- Added and updated unit assertions in:
  - `tests/unit/test_benchmark_artifact_bundle.py`
  - `tests/unit/test_benchmark_companion_summary.py`

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

Result:
- `12 passed`
- `1 warning`

## Outcome

- Bundle and companion surfaces now carry both:
  - `operator_adoption_knowledge_drift`
  - `operator_adoption_knowledge_outcome_drift`
- Markdown outputs now make operator outcome regressions visible without opening raw JSON.
