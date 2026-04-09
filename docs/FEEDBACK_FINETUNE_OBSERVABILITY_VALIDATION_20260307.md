# Feedback Finetune Observability Validation

## Scope
- Keep `finetune_from_feedback.py` training behavior unchanged
- Add optional JSON summary output for training labels
- Record both exact and coarse label distributions before fine-tuning

## Changed Files
- `scripts/finetune_from_feedback.py`
- `tests/unit/test_finetune_from_feedback.py`

## Validation
```bash
python3 -m py_compile scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback.py

flake8 scripts/finetune_from_feedback.py \
  tests/unit/test_finetune_from_feedback.py \
  --max-line-length=100

pytest -q tests/unit/test_finetune_from_feedback.py
```

## Result
- `py_compile`: pass
- `flake8`: pass
- `pytest`: `5 passed`

## Verified Behavior
- `_build_training_summary()` reports:
  - `sample_count`
  - `vector_count`
  - `unique_label_count`
  - `unique_coarse_label_count`
  - `label_distribution`
  - `coarse_label_distribution`
- `--summary-out` writes the JSON summary to disk
- Fine labels still fall back to legacy `true_type` when needed

## Notes
- This is additive observability for the feedback-to-training pipeline
- No model fitting behavior changed
