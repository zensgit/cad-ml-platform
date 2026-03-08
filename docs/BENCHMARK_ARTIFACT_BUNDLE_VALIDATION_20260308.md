## Goal

Create a reusable benchmark artifact bundle so operators can hand off one manifest covering the
scorecard, operational summary, feedback flywheel, assistant evidence, review queue, and OCR
review artifacts.

## Files

- `scripts/export_benchmark_artifact_bundle.py`
- `tests/unit/test_benchmark_artifact_bundle.py`

## Design

- Accept optional JSON inputs for each benchmark artifact family.
- Prefer the operational summary as the primary overall status when present.
- Fall back to scorecard component statuses when no operational summary exists.
- Emit one JSON bundle manifest and one Markdown operator summary.

## Validation

```bash
python3 -m py_compile scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py
flake8 scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  --max-line-length=100
pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

## Result

Local validation should pass with JSON/Markdown bundle outputs and fallback behavior covered by
unit tests.
