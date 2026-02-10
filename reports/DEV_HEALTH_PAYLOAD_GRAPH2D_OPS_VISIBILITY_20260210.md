# DEV_HEALTH_PAYLOAD_GRAPH2D_OPS_VISIBILITY_20260210

## Summary
- Extracted Graph2D temperature scaling load logic into a torch-free helper so it can be reused by health/readiness code paths without importing the full 2D vision stack.
- Extended `/health` and `/api/v1/health` payload schema to surface Graph2D guardrails + calibration metadata and to preserve provider-registry `provider_classes` in the typed health response.
- Documented DXF graph sampling env vars in `.env.example`.

## Implementation
- New helper: `src/ml/graph2d_temperature.py`
  - Precedence: `GRAPH2D_TEMPERATURE` > `GRAPH2D_TEMPERATURE_CALIBRATION_PATH` > default `1.0`
  - Returns `(temperature, source)` where source is `"env"`, `"<path>"`, or `None`.
- Refactor: `src/ml/vision_2d.py`
  - `Graph2DClassifier._load_temperature()` now delegates to the helper (no behavior change intended).
- Health schema + payload:
  - `src/api/health_models.py`:
    - `config.ml.classification.*` now includes optional Graph2D settings:
      - `graph2d_min_confidence`, `graph2d_min_margin`
      - `graph2d_exclude_labels`, `graph2d_allow_labels`
      - `graph2d_temperature`, `graph2d_temperature_source`, `graph2d_temperature_calibration_path`
    - `config.core_providers.provider_classes` is now preserved.
  - `src/api/health_utils.py` now populates the fields above.
- Env docs:
  - `.env.example` now includes `DXF_MAX_NODES`, `DXF_SAMPLING_STRATEGY`, `DXF_SAMPLING_SEED`, `DXF_TEXT_PRIORITY_RATIO`.

## API / Payload Changes
- `GET /health` and `GET /api/v1/health`
  - Adds optional fields under:
    - `config.ml.classification`
    - `config.core_providers.provider_classes`

## Validation
- `pytest -q tests/unit/test_graph2d_temperature_loading.py tests/unit/test_vision_2d_ensemble_voting.py tests/unit/test_health_utils_coverage.py tests/unit/test_provider_registry_bootstrap.py`
  - Result: `48 passed`

## Notes
- Health endpoints remain best-effort; failures in optional config discovery should not impact health availability.

