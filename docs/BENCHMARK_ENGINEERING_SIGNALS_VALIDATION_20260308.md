# Benchmark Engineering Signals Validation

## Goal

Promote standards / tolerance / GD&T style knowledge signals into the benchmark
stack so benchmark views compare engineering judgment, not only extraction or
classification coverage.

## Delivered

### 1. Reusable benchmark helper

- Added [src/core/benchmark/engineering_signals.py](../src/core/benchmark/engineering_signals.py)
- New reusable outputs:
  - `status`
  - `coverage_ratio`
  - `rows_with_checks`
  - `rows_with_violations`
  - `rows_with_standards_candidates`
  - `rows_with_hints`
  - `ocr_standard_signal_count`
  - `top_violation_categories`
  - `top_standard_types`
  - `top_hint_labels`

### 2. Standalone exporter

- Added [scripts/export_benchmark_engineering_signals.py](../scripts/export_benchmark_engineering_signals.py)
- Inputs:
  - `--hybrid-summary`
  - `--ocr-review-summary`
- Outputs:
  - JSON summary
  - Markdown summary

### 3. Scorecard integration

- Updated [scripts/generate_benchmark_scorecard.py](../scripts/generate_benchmark_scorecard.py)
- Added new optional input:
  - `--engineering-signals-summary`
- Added new scorecard component:
  - `engineering_signals`
- Added engineering gap handling:
  - `benchmark_ready_with_engineering_gap`

## Validation

Commands:

```bash
python3 -m py_compile \
  src/core/benchmark/engineering_signals.py \
  scripts/export_benchmark_engineering_signals.py \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_engineering_signals.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 \
  src/core/benchmark/engineering_signals.py \
  scripts/export_benchmark_engineering_signals.py \
  scripts/generate_benchmark_scorecard.py \
  tests/unit/test_benchmark_engineering_signals.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_benchmark_engineering_signals.py \
  tests/unit/test_generate_benchmark_scorecard.py
```

Result:

- `9 passed`
- `py_compile` passed
- `flake8` passed

## Notes

- This change does not yet wire engineering signals into `evaluation-report.yml`.
- It deliberately keeps the new scorecard component optional so existing
  benchmark runs do not regress when no engineering artifact is provided.
