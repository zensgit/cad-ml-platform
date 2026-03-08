# Benchmark Knowledge Application Validation

## Scope

- add reusable `knowledge_application` benchmark helper
- add `export_benchmark_knowledge_application.py`
- validate JSON + Markdown export path

## Design

The exporter combines:

- applied evidence from `benchmark_engineering_signals`
- foundation readiness from `benchmark_knowledge_readiness`

It reports benchmark-facing application status for:

- `tolerance`
- `standards`
- `gdt`

Each domain carries:

- `status`
- `readiness_status`
- `evidence_status`
- `signal_count`
- `signal_breakdown`
- `missing_metrics`
- `action`

This keeps the benchmark line focused on whether knowledge is both:

1. present in the built-in foundation
2. actually surfacing in extracted benchmark evidence

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/knowledge_application.py \
  scripts/export_benchmark_knowledge_application.py \
  tests/unit/test_benchmark_knowledge_application.py

flake8 \
  src/core/benchmark/knowledge_application.py \
  scripts/export_benchmark_knowledge_application.py \
  tests/unit/test_benchmark_knowledge_application.py \
  --max-line-length=100

pytest -q tests/unit/test_benchmark_knowledge_application.py
```

Expected:

- unit tests pass
- CLI exporter writes JSON and Markdown outputs
- ready and partial domain states are both covered
