# DEV_ANALYZER_LAZY_MODEL_LOAD_AND_GRAPH2D_SAFE_INIT_20260213

## Summary

Hardening changes to reduce import-time risk and avoid unnecessary heavyweight model loading:

1. `CADAnalyzer` no longer attempts to load V16/V6 classifiers when `CadDocument` does not have a
   usable on-disk `file_path`/`source_path` (common for byte-upload requests).
2. `Graph2DClassifier` model checkpoint loading is now best-effort: if a checkpoint is missing or
   incompatible, initialization will not crash module import.

## Changes

- `src/core/analyzer.py`
  - Reordered `_classify_with_v16` / `_classify_with_ml` so `file_path` is validated **before**
    calling `_get_v16_classifier()` / `_get_ml_classifier()`.
- `src/ml/vision_2d.py`
  - Wrapped `Graph2DClassifier._load_model()` during init in a try/except and degrades to
    `_loaded=False` on failure.
- `tests/unit/test_analyzer_lazy_model_load.py`
  - New regression tests ensuring `_get_v16_classifier` / `_get_ml_classifier` are not invoked
    when no file path is available.
- `tests/unit/test_vision_2d_ensemble_voting.py`
  - Added coverage ensuring `Graph2DClassifier` init is resilient to checkpoint load failures.

## Validation

Executed:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_analyzer_lazy_model_load.py \
  tests/unit/test_vision_2d_ensemble_voting.py -v

make validate-core-fast
```

Results:

- unit tests: `29 passed`
- `make validate-core-fast`: passed

## Notes / Caveats

- V16/V6 part classifiers still require an on-disk path; the analyze endpoint mainly operates on
  uploaded bytes, so these classifiers are typically exercised via the provider framework’s
  shadow-mode path (temp file) instead of the legacy `CADAnalyzer.classify_part(...)` route.
