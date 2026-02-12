# DEV_INFERENCE_FEATURES_REFACTOR_20260205

## Summary
Refactored DXF feature extraction into a shared utility and tightened inference
handling across classifier entrypoints. This reduces drift, aligns device
selection, and uses `torch.inference_mode()` for lighter inference.

## Changes
- Added shared 48-dim DXF feature extractor: `src/utils/dxf_features.py`.
- Reused shared extractor in `src/inference/classifier_api.py` and
  `src/ml/part_classifier.py`.
- Switched inference to `torch.inference_mode()` and added clearer error
  handling for feature extraction failures.
- Added model file existence checks in `V16Classifier.load()`.
- Aligned PartClassifier device selection to include MPS on macOS.
- Closed DXF render buffers via context managers and logged skipped entities at
  debug level.

## Verification
- No targeted inference tests available; exercised module import locally.

## Notes
- `classifier_api.py` now calls `setup_logging()` only in `__main__` to avoid
  overriding application-wide logging configuration.
