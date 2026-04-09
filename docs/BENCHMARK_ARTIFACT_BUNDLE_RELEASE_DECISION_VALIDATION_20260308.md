# Benchmark Artifact Bundle Release Decision Validation

## Scope

Extend the benchmark artifact bundle so it can optionally include
`benchmark_release_decision` and prefer that signal when summarizing bundle
status for operators.

## Validation

```bash
python3 -m py_compile \
  scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py

flake8 \
  scripts/export_benchmark_artifact_bundle.py \
  tests/unit/test_benchmark_artifact_bundle.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_artifact_bundle.py
```

## Result

- artifact bundle accepts `--benchmark-release-decision`
- bundle artifact rows include release decision presence/path
- bundle summary prefers release decision status when present
