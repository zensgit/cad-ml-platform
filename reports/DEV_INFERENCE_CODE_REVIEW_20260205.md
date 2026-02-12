# DEV_INFERENCE_CODE_REVIEW_20260205

## Summary
Reviewed the updated inference paths for the CAD classifiers to identify maintainability and
runtime risks. Findings focus on consistency between inference entrypoints, error handling, and
resource management.

## Scope
- `src/inference/classifier_api.py`
- `src/ml/part_classifier.py`

## Findings & Recommendations
1. **Logging configuration conflicts** (`src/inference/classifier_api.py`):
   `logging.basicConfig(...)` is executed at import time and can override app-wide logging. Consider
   using the shared logger in `src/utils/logging` or configuring logging in the entrypoint instead.
2. **Model availability checks** (`src/inference/classifier_api.py`):
   `V16Classifier.load()` assumes model files exist. Add explicit `Path.exists()` checks and a
   clearer error message so missing artifacts are surfaced predictably.
3. **Resource cleanup for rendering** (`src/inference/classifier_api.py`, `src/ml/part_classifier.py`):
   `io.BytesIO()` buffers are not explicitly closed. Wrapping buffers in a `with io.BytesIO() as buf:`
   block (or calling `buf.close()`) avoids residual memory usage in long-running services.
4. **Inference context optimization** (both files):
   Use `torch.inference_mode()` in prediction paths for faster inference and lower memory pressure
   instead of `torch.no_grad()`.
5. **Feature extraction duplication** (both files):
   The 48-dim feature logic is implemented three times (`EnhancedFeatureExtractorV4`,
   `PartClassifier._extract_features_v6`, `PartClassifierV16._extract_features`). This risks drift.
   Consider centralizing into a shared helper (e.g., `src/utils/dxf_features.py`).
6. **Bare `except` in V16 feature extraction** (`src/ml/part_classifier.py`):
   `_extract_features()` uses `except:` and silently ignores all exceptions. Swap to
   `except Exception as exc` and optionally log debug details to satisfy linting and avoid masking
   interrupts.
7. **Device selection inconsistency** (`src/ml/part_classifier.py` vs `src/inference/classifier_api.py`):
   `PartClassifier` ignores MPS while other inference paths support it. Align device selection to
   avoid inconsistent behavior on macOS.
8. **Failure handling returns all-zero features** (`src/inference/classifier_api.py`):
   Returning zeros on extraction failure can hide malformed inputs. Consider propagating `None`
   and returning an explicit 4xx/5xx response so clients can detect invalid DXF data.

## Notes
- No code changes were applied in this review; items above are recommendations for follow-up.
