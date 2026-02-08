# DEV_READINESS_CORE_PROVIDERS_CHECK_20260207

## Summary

Extended the `/ready` readiness probe to optionally validate "core providers" (via the provider framework health checks) and included a lightweight provider prerequisite summary in `/health` for clearer degraded-mode diagnosis (torch/model files/feature flags).

## Changes

- `src/core/providers/readiness.py`
  - Added helpers to parse `domain/provider` lists and run timeout-bounded readiness checks via `ProviderRegistry`.

- `src/main.py`
  - Enhanced `_run_readiness_check` to support structured check results (`{"ok": bool, "degraded": bool, "detail": str}`).
  - Added a new readiness check key: `core_providers`.
    - Enabled when `READINESS_REQUIRED_PROVIDERS` or `READINESS_OPTIONAL_PROVIDERS` is set.
    - Fails readiness (`503`) when any required provider is down.
    - Marks the check as `degraded=true` when optional providers are down.

- `src/api/health_utils.py` + `src/api/health_models.py`
  - Added `config.ml.readiness` to `/health` payload with:
    - `torch_available`
    - model paths + presence flags
    - `degraded_reasons`
    - readiness provider lists from env vars

- `src/core/providers/classifier.py`
  - Improved classifier provider health checks to return explicit errors (via exceptions) for:
    - `disabled_by_config`
    - `torch_missing`
    - `model_missing: ...`

## Configuration

- `READINESS_REQUIRED_PROVIDERS`:
  - Comma-separated list of required providers, e.g. `classifier/hybrid,ocr/paddle`
- `READINESS_OPTIONAL_PROVIDERS`:
  - Comma-separated list of optional providers, e.g. `classifier/graph2d_ensemble,classifier/v16`

## Validation

Executed:

```bash
python3 -m pytest -q tests/unit/test_main_coverage.py::TestReadinessCheck
python3 -m pytest -q tests/unit/test_provider_readiness.py
python3 -m pytest -q tests/unit/test_provider_health_endpoint.py
python3 -m pytest -q tests/unit/test_main_coverage.py -k health
```

Result: PASS

