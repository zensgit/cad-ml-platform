# DEV_PROVIDER_FRAMEWORK_REVIEW_20260210

## Summary
- Reviewed the current provider framework (`src/core/providers/`) and the local contract helper script (`scripts/test_with_local_api.sh`) to answer whether the provider framework should be kept/extended.
- Conclusion: the provider framework is already wired into runtime health/readiness and the analyze pipeline; it is worth keeping, but further work should remain incremental to avoid merge risk.

## What Exists Today
- Core framework:
  - `src/core/providers/base.py`: `BaseProvider` (async `process`, `health_check`, status snapshot)
  - `src/core/providers/registry.py`: `ProviderRegistry` (type registry + optional instance caching)
  - `src/core/providers/bootstrap.py`: built-in bootstraps + plugin hook (`CORE_PROVIDER_PLUGINS`)
  - `src/core/providers/readiness.py`: readiness probes + metrics emission
- Adapters registered into the registry:
  - `src/core/providers/classifier.py` (Hybrid/Graph2D/V6/V16)
  - `src/core/providers/vision.py`, `src/core/providers/ocr.py`, `src/core/providers/knowledge.py`
- API integration:
  - `/ready` uses provider readiness selection (`READINESS_REQUIRED_PROVIDERS`, `READINESS_OPTIONAL_PROVIDERS`)
  - `/api/v1/health/providers/registry` and `/api/v1/health/providers/health` expose registry and per-provider check results
  - `/api/v1/analyze` can route classification via the provider registry

## Why This Is Worth Keeping
- It reduces merge risk for model iteration:
  - new part-recognition models can be added as separate modules and enabled via `CORE_PROVIDER_PLUGINS` without editing the central bootstrap list
  - rollout can be controlled via env flags and provider readiness configuration
- It centralizes operational behavior:
  - consistent health checks, readiness gates, and check metrics across providers

## Recommended Next Improvements (Incremental)
1) Keep failure visibility without breaking readiness
- `src/core/providers/readiness.py` intentionally treats metrics emission as best-effort (it will not fail readiness).
- If needed, replace the broad `except Exception: pass` with a small helper that logs at debug level to avoid silently hiding instrumentation bugs.

2) Add lifecycle wiring only if/when heavy providers appear
- `BaseProvider` defines `warmup()`/`shutdown()`.
- Only wire these into FastAPI lifespan once we have providers with meaningful startup cost (to avoid slowing dev and CI paths).

3) Treat provider IDs as a stable API surface
- Standardize on `domain/name` strings for env configs and API payloads.
- Keep parsing tolerant (as in `parse_provider_id_list`) to reduce operational foot-guns.

## `scripts/test_with_local_api.sh` Status
- This script is tracked under `scripts/` and supports local contract testing against a running API (with fallbacks for local environments).
- It should be kept as an operator/developer convenience; if desired we can link it from a runbook or `README.md`.

## Validation Notes
- This change is a review only (no runtime behavior changes).
- Existing provider framework validations already cover metrics exposition and provider health endpoints:
  - `pytest -q tests/unit/test_provider_check_metrics_exposed.py`

