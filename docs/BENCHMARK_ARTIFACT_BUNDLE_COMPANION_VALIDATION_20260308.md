# Benchmark Artifact Bundle Companion Validation

## Goal

Include `benchmark companion summary` as a first-class input and artifact row in the
benchmark artifact bundle export.

## Changes

- Added `--benchmark-companion-summary` input to
  `scripts/export_benchmark_artifact_bundle.py`.
- Added `benchmark_companion_summary` to exported artifact rows.
- Prefer companion summary when present for:
  - `overall_status`
  - `blockers`
  - `recommended_actions`
  - compact component statuses such as assistant, review queue, OCR, and Qdrant.

## Validation

```bash
python3 -m py_compile scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py
flake8 scripts/export_benchmark_artifact_bundle.py tests/unit/test_benchmark_artifact_bundle.py --max-line-length=100
pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

## Expected Outcome

- Benchmark artifact bundles can carry companion summary status alongside scorecard,
  operational summary, feedback, assistant evidence, review queue, and OCR review.
- Companion summary becomes part of the benchmark artifact chain rather than remaining
  a standalone side artifact only.
