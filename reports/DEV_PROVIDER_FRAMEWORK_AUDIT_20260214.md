# DEV_PROVIDER_FRAMEWORK_AUDIT_20260214

## Goal

Confirm whether the `src/core/providers/` provider framework and the local tiered runner `scripts/test_with_local_api.sh` are production-relevant (vs. temporary), and document the current state so future feature branches (e.g. part-classifier/tolerance-knowledge) can integrate safely.

## Findings

### Provider Framework Is Core (Not Temporary)

The provider framework is a first-class runtime integration point used by the API layer:

- `src/api/v1/analyze.py`
  - Uses `ProviderRegistry` for optional classifier overlays (`classifier/graph2d*`, `classifier/hybrid`, `classifier/v16|v6`).
  - Uses `ProviderRegistry` for OCR providers in the optional OCR stage (`ocr/paddle`, `ocr/deepseek_hf`).
- `src/api/v1/health.py`
  - Exposes provider registry snapshots via `/api/v1/providers/registry`.
  - Exposes provider health checks via `/api/v1/providers/health` (best-effort, timeout bounded).
- `src/core/providers/bootstrap.py`
  - Supports plugin loading via `CORE_PROVIDER_PLUGINS` (module import and optional `:bootstrap` function).
  - Tracks plugin state and emits metrics.
- `src/core/providers/readiness.py`
  - Implements readiness selection using `READINESS_REQUIRED_PROVIDERS` / `READINESS_OPTIONAL_PROVIDERS`.

Conclusion: keep and maintain `src/core/providers/` as a stable abstraction boundary. Do not treat it as a scratch folder.

### `scripts/test_with_local_api.sh` Is a Canonical Validation Runner

The script supports:

- Running `unit`, `contract`, `e2e`, or `all` suites.
- Optional local API auto-start (`uvicorn src.main:app`), readiness wait, and cleanup.
- Contract fallback mode (`CONTRACT_INPROCESS_FALLBACK`) when local port binding is not permitted.

This makes it suitable as a default "developer smoke" entrypoint and should not be removed as a temp artifact.

## Recommended Integration Pattern for New Providers

1. Add a provider adapter under `src/core/providers/<domain>.py` with **lazy imports** for optional deps (torch/ocr/etc.).
2. Register via `bootstrap_core_*_providers()` and/or via plugin module + `CORE_PROVIDER_PLUGINS` to reduce merge conflicts.
3. Keep API behavior gated by feature flags (as done in `src/api/v1/analyze.py`) to avoid breaking default runtime.

## References

- Provider docs:
  - `docs/PROVIDER_FRAMEWORK.md`
  - `docs/PROVIDER_PLUGIN_GUIDE.md`
- Key code:
  - `src/core/providers/registry.py`
  - `src/core/providers/bootstrap.py`
  - `src/core/providers/readiness.py`
  - `src/api/v1/health.py`
  - `src/api/v1/analyze.py`

