# Feedback Flywheel Benchmark Validation 20260308

## Goal

Extend the unified benchmark scorecard so "benchmark surpass" also measures whether
feedback is actually being converted into retraining signals.

The new scorecard component evaluates three inputs:

- feedback observability from `/api/v1/feedback/stats`
- fine-tune training summary from `scripts/finetune_from_feedback.py --summary-out`
- metric-learning training summary from `scripts/train_metric_model.py --summary-out`

## Design

Added a new scorecard component: `feedback_flywheel`

It reports:

- `feedback_total`
- `correction_count`
- `coarse_correction_count`
- `average_rating`
- `finetune_sample_count`
- `finetune_vector_count`
- `metric_triplet_count`
- `metric_unique_anchor_count`

Status mapping:

- `missing`
- `passive_feedback_only`
- `feedback_collected`
- `partially_closed_loop`
- `closed_loop_ready`

Benchmark effect:

- if the flywheel is not closed, overall status degrades to
  `benchmark_ready_with_feedback_gap`
- recommendations now explicitly call out missing feedback/retraining artifacts

## Implementation

Updated:

- `scripts/generate_benchmark_scorecard.py`
- `scripts/train_metric_model.py`
- `tests/unit/test_generate_benchmark_scorecard.py`
- `tests/unit/test_train_metric_model.py`

New behavior:

- `train_metric_model.py` now supports `--summary-out`
- benchmark scorecard accepts:
  - `--feedback-summary`
  - `--finetune-summary`
  - `--metric-train-summary`
- Markdown scorecard renders a `feedback_flywheel` row

## Validation

Commands:

```bash
python3 -m py_compile \
  scripts/generate_benchmark_scorecard.py \
  scripts/train_metric_model.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_train_metric_model.py

flake8 \
  scripts/generate_benchmark_scorecard.py \
  scripts/train_metric_model.py \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_train_metric_model.py \
  --max-line-length=100

pytest -q \
  tests/unit/test_generate_benchmark_scorecard.py \
  tests/unit/test_train_metric_model.py \
  tests/unit/test_finetune_from_feedback.py \
  tests/test_feedback.py
```

Results:

- `py_compile` passed
- `flake8` passed
- `pytest`: `21 passed`

## Outcome

The benchmark scorecard no longer treats feedback collection as an implicit claim.
It now distinguishes:

- feedback exists but no retraining loop
- partial retraining loop
- closed-loop retraining readiness

This closes another gap between "feature exists" and "benchmark-proven beyond".
