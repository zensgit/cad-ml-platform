# DEV_GRAPH2D_ENSEMBLE_HEALTH_VISIBILITY_20260210

## Summary
- Exposed Graph2D ensemble configuration in the health payload for operational debugging.
- Aligned HybridClassifier Graph2D lazy-loader with `GRAPH2D_ENSEMBLE_ENABLED`.

## Implementation
- `.env.example`
  - Added:
    - `GRAPH2D_ENSEMBLE_ENABLED`
    - `GRAPH2D_ENSEMBLE_MODELS`
- `src/api/health_models.py`
  - Added optional fields under `config.ml.classification`:
    - `graph2d_ensemble_enabled`
    - `graph2d_ensemble_models_configured`
    - `graph2d_ensemble_models_present`
    - `graph2d_ensemble_models` (basenames only)
- `src/api/health_utils.py`
  - Populates ensemble fields (effective defaults match `src/ml/vision_2d.py`).
- `src/ml/hybrid_classifier.py`
  - If `GRAPH2D_ENSEMBLE_ENABLED=true`, uses `get_ensemble_2d_classifier()`; otherwise uses `get_2d_classifier()`.

## API / Payload Changes
- `GET /health` and `GET /api/v1/health`
  - Adds optional fields under `config.ml.classification` (Graph2D ensemble ops fields).

## Validation
- `pytest -q tests/unit/test_health_utils_coverage.py tests/unit/test_graph2d_temperature_loading.py tests/unit/test_vision_2d_ensemble_voting.py`
  - Result: pass

## Notes
- These fields are additive and should not impact environments without torch/model artifacts.

