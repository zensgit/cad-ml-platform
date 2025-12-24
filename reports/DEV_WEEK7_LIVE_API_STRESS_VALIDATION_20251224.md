# DEV_WEEK7_LIVE_API_STRESS_VALIDATION_20251224

## Scope
- Fix model reload when path is omitted.
- Run integration and stress validation against a live local API.

## Changes
- `src/ml/classifier.py`
  - Allow missing reload path by falling back to `CLASSIFICATION_MODEL_PATH` env or current model path.
- `tests/unit/test_model_reload_endpoint.py`
  - Add coverage for missing path reload request.

## Setup
- Created temp model: `/tmp/cad_ml_dummy_model.pkl` (from `src/ml/dummy_model_holder.py`).
- Started API on `http://127.0.0.1:8010` with:
  - `CLASSIFICATION_MODEL_PATH=/tmp/cad_ml_dummy_model.pkl`
  - `MODEL_OPCODE_MODE=audit`

## Validation
- Command: `.venv/bin/python -m pytest tests/unit/test_model_reload_endpoint.py::test_model_reload_missing_path_uses_default -v`
  - Result: `1 passed`.
- Command: `API_BASE_URL=http://127.0.0.1:8010 API_KEY=test ADMIN_TOKEN=test .venv/bin/python -m pytest tests/integration/test_stress_stability.py -v`
  - Result: `3 passed`.
- Command: `.venv/bin/python scripts/stress_concurrency_reload.py --url http://127.0.0.1:8010 --api-key test --admin-token test --threads 4 --iterations 3`
  - Result: `success 12/12`, load sequence monotonic.
- Command: `.venv/bin/python scripts/stress_degradation_flapping.py --url http://127.0.0.1:8010 --api-key test --cycles 20 --interval 0.2`
  - Result: PASS (20/20 OK).

## Notes
- Local API was stopped after tests.
