# Feedback / Active Learning Unification Validation

Date: 2026-03-08

## Goal

Unify the feedback flywheel so that these paths can consume each other's outputs
without ad-hoc field translation:

- `POST /api/v1/feedback`
- `ActiveLearner.export_training_data()`
- `scripts/finetune_from_feedback.py`
- `scripts/train_metric_model.py`

## Design

The compatibility layer keeps the newer coarse/fine contract intact while adding
legacy aliases expected by retraining scripts.

### Normalized feedback log aliases

`src/api/v1/feedback.py` now writes:

- `correct_label`
- `correct_fine_label`
- `correct_coarse_label`
- `original_label`
- `original_fine_label`
- `original_coarse_label`

These are emitted alongside the canonical fields:

- `corrected_fine_part_type`
- `corrected_coarse_part_type`
- `original_fine_part_type`
- `original_coarse_part_type`

### Active learning export aliases

`src/core/active_learning.py` export records now include:

- `analysis_id` as an alias of `doc_id`
- `correct_label`
- `correct_fine_label`
- `correct_coarse_label`
- `original_label`
- `original_fine_label`
- `original_coarse_label`

This lets active-learning exports feed both fine-tune and metric-learning
scripts without a separate transform step.

### Retrain readiness contract

`ActiveLearner.check_retrain_threshold()` now returns:

- `remaining_samples`
- `recommendation`

This removes the runtime bug where `scripts/finetune_from_feedback.py` expected
`status["recommendation"]` but the field was not present.

### Legacy script compatibility

`scripts/finetune_from_feedback.py` now:

- accepts `analysis_id` when `doc_id` is absent
- accepts feedback aliases like `correct_label`

`scripts/train_metric_model.py` now:

- loads from `data/feedback/*.jsonl`
- also falls back to `FEEDBACK_LOG_PATH` / `data/feedback_log.jsonl`
- accepts both old and new label field names
- accepts `analysis_id` or `doc_id`

## Files Changed

- `src/api/v1/feedback.py`
- `src/core/active_learning.py`
- `src/api/v1/active_learning.py`
- `scripts/finetune_from_feedback.py`
- `scripts/train_metric_model.py`
- `tests/test_feedback.py`
- `tests/test_active_learning_api.py`
- `tests/unit/test_active_learning_loop.py`
- `tests/unit/test_active_learning_export_context.py`
- `tests/unit/test_finetune_from_feedback.py`
- `tests/unit/test_train_metric_model.py`

## Validation

### Compile

```bash
python3 -m py_compile \
  src/api/v1/feedback.py \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  scripts/finetune_from_feedback.py \
  scripts/train_metric_model.py \
  tests/test_feedback.py \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_finetune_from_feedback.py \
  tests/unit/test_train_metric_model.py
```

Result: pass

### Lint

```bash
flake8 \
  src/api/v1/feedback.py \
  src/core/active_learning.py \
  src/api/v1/active_learning.py \
  scripts/finetune_from_feedback.py \
  scripts/train_metric_model.py \
  tests/test_feedback.py \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_finetune_from_feedback.py \
  tests/unit/test_train_metric_model.py \
  --max-line-length=100
```

Result: pass

### Targeted tests

```bash
pytest -q \
  tests/test_feedback.py \
  tests/test_active_learning_api.py \
  tests/unit/test_active_learning_loop.py \
  tests/unit/test_active_learning_export_context.py \
  tests/unit/test_finetune_from_feedback.py \
  tests/unit/test_train_metric_model.py
```

Result: `32 passed`

## Outcome

The feedback flywheel is now materially more coherent:

- human feedback logs can be consumed by legacy metric-learning scripts
- active-learning exports can be consumed by fine-tuning scripts without reshaping
- retrain readiness now returns an actionable recommendation
- coarse/fine semantics remain explicit while backward-compatible aliases are preserved

## Remaining Work

- feed these compatibility aliases into any future ranked review queue API
- connect feedback summary stats to CI / governance reporting
- replace mock metric-learning scaffolding with a real store-backed retraining path
