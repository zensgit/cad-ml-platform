# Finetune Feedback Label Fields Validation (2026-03-07)

## Goal

Allow `scripts/finetune_from_feedback.py` to train against either fine or coarse labels from exported active-learning feedback, while preserving compatibility with older exports that only contain `true_type`.

## Changes

- Added label resolution helper in `scripts/finetune_from_feedback.py`.
- `_load_samples()` now accepts `label_field` and falls back across:
  - requested field
  - `true_fine_type`
  - `true_type`
  - `true_coarse_type`
- `load_training_data()` now passes through `label_field`.
- CLI now supports `--label-field` with:
  - `true_fine_type`
  - `true_coarse_type`
  - `true_type`
- `scripts/finetune_from_feedback_e2e.py` now exposes the same option.
- Delayed `src.ml.classifier` import in `main()` to keep the module lightweight for unit tests.

## Validation Commands

```bash
python3 -m py_compile scripts/finetune_from_feedback.py \
  scripts/finetune_from_feedback_e2e.py tests/unit/test_finetune_from_feedback.py
flake8 scripts/finetune_from_feedback.py \
  scripts/finetune_from_feedback_e2e.py tests/unit/test_finetune_from_feedback.py \
  --max-line-length=100
pytest -q tests/unit/test_finetune_from_feedback.py
```

## Expected Result

- Fine-label training remains the default.
- Coarse-label training can be selected explicitly.
- Legacy exports without `true_fine_type` still work.
