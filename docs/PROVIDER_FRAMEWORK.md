# Provider Framework

This repository contains a lightweight provider framework under `src/core/providers/` used to:

- decouple API/business logic from concrete ML/LLM/OCR backends,
- centralize health checks and status snapshots,
- keep optional dependencies lazy (for example `torch`), and
- make it easy to add new backends behind stable interfaces.

## Key Concepts

### `BaseProvider`

`BaseProvider` is the reusable async base class:

- `process(request, **kwargs)` runs provider work.
- `health_check()` runs a provider readiness probe and updates an internal `status` snapshot.
- `status_snapshot()` provides a serializable runtime snapshot (status, last_error, timing).

Implementation guidance:

- Prefer cheap `health_check()` probes. Avoid loading large models or calling external APIs unless needed.
- Keep optional imports lazy inside methods, not at module import time.

### `ProviderRegistry`

`ProviderRegistry` is a global registry keyed by `(domain, provider_name)`.

- `register(domain, provider_name)` decorator registers a provider class.
- `get(domain, provider_name, *args, **kwargs)` instantiates a provider.
- `list_domains() / list_providers(domain)` supports discovery.

Naming guidance:

- Domains are broad areas: `vision`, `ocr`, `classifier`, ...
- Provider names identify the implementation: `paddle`, `deepseek_hf`, `hybrid`, ...

### Bootstrap

Core providers are registered at startup via `bootstrap_core_provider_registry()`:

- `src/core/providers/bootstrap.py`
- Called best-effort in `src/main.py` lifespan

## Built-In Providers (Core)

The following are registered by default:

- `vision/stub`
- `vision/deepseek_stub`
- `ocr/paddle`
- `ocr/deepseek_hf`
- `classifier/hybrid`
- `classifier/graph2d`
- `classifier/graph2d_ensemble`
- `classifier/v16`
- `classifier/v6`

See:

- `src/core/providers/vision.py`
- `src/core/providers/ocr.py`
- `src/core/providers/classifier.py`

## HTTP APIs For Operations

Provider visibility endpoints:

- `GET /api/v1/providers/registry` (alias: `GET /api/v1/health/providers/registry`)
  - returns a snapshot of registry domains and provider names.
- `GET /api/v1/providers/health` (alias: `GET /api/v1/health/providers/health`)
  - runs best-effort, timeout-bounded provider health checks and returns per-provider readiness.

Note:

- Providers can be present in the registry but not "ready" (for example: missing optional deps or model files).

## Adding A New Provider

1. Create a new adapter or provider class under `src/core/providers/<domain>.py`.
2. Register it in a domain bootstrap function (for example `bootstrap_core_<domain>_providers()`).
3. Ensure `src/core/providers/bootstrap.py` calls the bootstrap function.
4. Add unit coverage:
   - registration existence
   - happy-path `process(...)`
   - `health_check()` behavior in both ok and failure modes

Minimal example:

```python
from dataclasses import dataclass
from typing import Dict

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@dataclass
class DemoConfig(ProviderConfig):
    pass


@ProviderRegistry.register("demo", "noop")
class NoopProvider(BaseProvider[DemoConfig, Dict[str, str]]):
    async def _process_impl(self, request, **kwargs):
        return {"status": "ok"}
```

## Testing Tips

- Use patching to avoid optional dependency variance (for example `torch` installed vs missing).
- Keep provider health checks deterministic for unit tests.
- Prefer unit tests that import and instantiate providers without starting the full FastAPI app, unless endpoint behavior is the goal.

