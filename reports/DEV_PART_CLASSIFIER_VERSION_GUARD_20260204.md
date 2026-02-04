# DEV_PART_CLASSIFIER_VERSION_GUARD_20260204

## Summary
- Added version inference when checkpoint metadata is missing to prevent loading the wrong model architecture.
- Clarified PartClassifier support notes and aligned V9 training script default dataset path with V6.

## Changes
- `src/ml/part_classifier.py`
  - Updated support note to indicate historical V7/V8 compatibility only.
  - Added `_infer_version()` based on `input_dim`/`num_classes` when `version` is absent.
- `scripts/train_classifier_v9.py`
  - Default dataset path updated to `data/training_v6`.

## Validation
- Not run (logic change only; covered by existing CI tests).
