# Provider Framework (Core Provider Registry)

## What This Is

This repo includes a lightweight provider framework under `src/core/providers/` to standardize:
- Provider discovery/registration (`ProviderRegistry`)
- Health/readiness checks (`BaseProvider.health_check()`)
- Optional dependency isolation (lazy imports for torch-heavy providers)
- Runtime visibility via `/api/v1/health/providers/*` and `/ready`

It is intentionally small and does **not** try to replace domain-specific APIs (for example `src/core/vision/*`).

## Core Concepts

### Domain + Provider Name

Every provider is identified by:
- `domain` (string): e.g. `vision`, `ocr`, `classifier`
- `provider_name` (string): e.g. `stub`, `paddle`, `hybrid`, `v16`

The canonical ID format is `domain/provider_name` (example: `classifier/v16`).

### `BaseProvider`

`src/core/providers/base.py` defines:
- `process(request, **kwargs)` -> calls provider-specific `_process_impl`
- `health_check(timeout_seconds=...)` -> best-effort check with status/latency tracking
- `status_snapshot()` -> JSON-serializable runtime status

### `ProviderRegistry`

`src/core/providers/registry.py` manages:
- Provider class registration: `@ProviderRegistry.register(domain, name)`
- Instance creation: `ProviderRegistry.get(domain, name, *args, **kwargs)`
- Optional singleton caching for default instances:
  - Controlled by `PROVIDER_REGISTRY_CACHE_ENABLED` (default: enabled)
  - Tests should clear instances between cases (see `tests/conftest.py`)

### Bootstrap

`src/core/providers/bootstrap.py` registers built-in adapters (vision/ocr/classifier) and exposes a snapshot:
- `bootstrap_core_provider_registry()`
- `get_core_provider_registry_snapshot()`

## Built-In Core Providers

These providers are registered by default via `bootstrap_core_provider_registry()`:

- `vision/stub`
- `vision/deepseek_stub`
- `ocr/paddle`
- `ocr/deepseek_hf`
- `classifier/hybrid`
- `classifier/graph2d`
- `classifier/graph2d_ensemble`
- `classifier/v16`
- `classifier/v6`

Implementation references:
- `src/core/providers/vision.py`
- `src/core/providers/ocr.py`
- `src/core/providers/classifier.py`

## Runtime Endpoints

These are the main operational touchpoints:

```bash
# Registry snapshot (domains, providers, provider class paths)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/registry
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/providers/registry

# Best-effort health checks for all registered providers
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/providers/health

# Readiness probe (includes provider readiness summary when configured)
curl http://localhost:8000/ready
```

Readiness selection is controlled via:
- `READINESS_REQUIRED_PROVIDERS` (comma/space separated, e.g. `ocr/paddle classifier/hybrid`)
- `READINESS_OPTIONAL_PROVIDERS` (same format)

## Adding A New Provider (Recommended Pattern)

### 1) Define a config dataclass (optional but recommended)

```python
from dataclasses import dataclass
from src.core.providers.base import ProviderConfig


@dataclass
class MyProviderConfig(ProviderConfig):
    # Add provider-specific config fields here
    pass
```

### 2) Implement a provider class

Key rule: **the registry must be able to instantiate your provider with no args**.
Do this by accepting an optional config and providing a default config.

```python
from __future__ import annotations

from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider
from src.core.providers.registry import ProviderRegistry


@ProviderRegistry.register("my_domain", "my_provider")
class MyProvider(BaseProvider[MyProviderConfig, Dict[str, Any]]):
    def __init__(self, config: Optional[MyProviderConfig] = None):
        cfg = config or MyProviderConfig(
            name="my_provider",
            provider_type="my_domain",
        )
        super().__init__(cfg)

    async def _health_check_impl(self) -> bool:
        # Cheap and deterministic checks only (no heavy model loads).
        return True

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "ok"}
```

### 3) Keep optional dependencies lazy

If your provider depends on optional packages (torch, CAD kernels, external SDKs):
- Do **not** import them at module import time.
- Import inside `_process_impl` or behind a `find_spec(...)` check.
- In `_health_check_impl`, validate prerequisites (feature flags, model files, dependency presence) without doing real heavy work.

This pattern is already used in `src/core/providers/classifier.py` for `classifier/v16` and `classifier/graph2d`.

### 4) Wire it into bootstrap (when appropriate)

If your provider should be available in production by default (and appear in `/api/v1/*/providers/*` without extra imports), add it to a bootstrap function and ensure it is called by:
- `src/core/providers/bootstrap.py` (directly), or
- `bootstrap_core_provider_registry()` (indirectly via an existing domain bootstrap)

This repo intentionally centralizes core registration in `bootstrap_core_provider_registry()` so provider availability is explicit and observable.

## When To Use Providers (And When Not To)

Good fits:
- External services (remote OCR/vision/LLM)
- Heavy ML models with optional runtime dependencies
- Anything that benefits from standardized health/readiness + enable/disable flags

Not necessary:
- Pure, deterministic local logic (for example: pure lookup tables or simple parsing),
  unless you specifically want the operational surface (health/readiness/config gating).

## Testing Notes

When writing tests for providers:
- Use `ProviderRegistry.clear()` to remove registrations (careful: affects other tests).
- Prefer `ProviderRegistry.clear_instances()` (keeps registrations, clears cached singletons).
- Keep health checks deterministic and fast; avoid requiring GPU/torch unless explicitly gated.
