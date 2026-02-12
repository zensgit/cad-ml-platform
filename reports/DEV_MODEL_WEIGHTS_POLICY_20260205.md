# DEV_MODEL_WEIGHTS_POLICY_20260205

## Summary
Documented the model-weights handling policy and optional Git LFS flow.

## Changes
- Added `docs/MODEL_WEIGHTS.md` describing the default local-weights workflow
  and how to switch to Git LFS if needed.

## Notes
- `.gitignore` already excludes `models/*.pt` to avoid committing large
  checkpoints unintentionally.
