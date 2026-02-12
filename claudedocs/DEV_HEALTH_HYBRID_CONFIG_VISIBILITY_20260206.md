# DEV_HEALTH_HYBRID_CONFIG_VISIBILITY_20260206

## Scope

- Expose effective hybrid runtime config in health payload.
- Add dedicated endpoint for hybrid config runtime inspection.
- Keep health endpoint resilient if hybrid config loading fails.

## Code Changes

- Updated:
  - `src/api/health_models.py`
  - `src/api/health_utils.py`
  - `src/api/v1/health.py`
  - `docs/HEALTH_ENDPOINT_CONFIG.md`
- Added:
  - `tests/unit/test_health_hybrid_config.py`

## API Additions

- `GET /api/v1/ml/hybrid-config`
- `GET /api/v1/health/ml/hybrid-config`

Response:

```json
{
  "status": "ok",
  "config": { "...effective hybrid config..." }
}
```

## Health Payload Extension

`GET /health` and `GET /api/v1/health` now include:

- `config.ml.classification`
- `config.ml.sampling`

These fields are derived from merged runtime config (`defaults + YAML + env`).

## Verification Commands

```bash
python3 -m black src/api/health_models.py src/api/health_utils.py src/api/v1/health.py tests/unit/test_health_hybrid_config.py
python3 -m pytest tests/unit/test_health_hybrid_config.py tests/unit/test_main_coverage.py tests/unit/test_model_health_uptime.py -q
python3 -m flake8 src/api/health_models.py src/api/health_utils.py src/api/v1/health.py tests/unit/test_health_hybrid_config.py
```

## Results

- Formatting: success.
- Unit tests: `28 passed`.
- Flake8: success.
