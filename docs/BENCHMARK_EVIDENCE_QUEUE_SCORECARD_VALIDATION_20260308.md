# Benchmark Evidence and Queue Scorecard Validation 2026-03-08

## Goal

Extend the unified benchmark scorecard so "benchmark surpass" is not judged only
by recognition accuracy. The scorecard now also evaluates:

- assistant explainability coverage
- active-learning review queue readiness

This makes the benchmark closer to real product competitiveness:

- explainable results
- governable review backlog
- operational rollout readiness

## Design

### New scorecard inputs

- `--assistant-evidence-summary`
- `--review-queue-summary`

Both inputs are optional JSON files.

### New components

- `assistant_explainability`
  - `missing`
  - `weak_coverage`
  - `partial_coverage`
  - `explainability_ready`
- `review_queue`
  - `missing`
  - `under_control`
  - `routine_backlog`
  - `managed_backlog`
  - `critical_backlog`

### New benchmark logic

- If explainability evidence is weak or partial, the overall scorecard reports
  `benchmark_ready_with_explainability_gap`.
- If the review queue still has high or critical backlog, the overall scorecard
  reports `benchmark_ready_with_review_gap`.
- Existing governance, history, and B-Rep readiness logic remains intact.

## Files

- `scripts/generate_benchmark_scorecard.py`
- `tests/unit/test_generate_benchmark_scorecard.py`

## Validation

Commands run:

```bash
python3 -m py_compile scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py

flake8 scripts/generate_benchmark_scorecard.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  --max-line-length=100

pytest -q tests/unit/test_generate_benchmark_scorecard.py
```

## Expected outcome

- scorecard JSON contains `assistant_explainability` and `review_queue`
- markdown contains both new component rows
- existing missing-input behavior remains stable

