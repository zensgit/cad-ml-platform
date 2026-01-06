# DEV_FINETUNE_FROM_FEEDBACK_E2E_VALIDATION_20260105

## Scope
Validate the minimal end-to-end demo for fine-tuning from feedback.

## Command
- `python3 scripts/finetune_from_feedback_e2e.py --skip-train`

## Results
- Output:
  - `exported=1 vectors=1 labels=1`
  - `training=skipped`

## Notes
- Training can be enabled by running without `--skip-train` when sklearn is available.
