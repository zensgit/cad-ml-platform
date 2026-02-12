# Provider Plugin Guide (CORE_PROVIDER_PLUGINS)

This guide explains how to extend the core provider registry (`src/core/providers/`)
via the plugin hook in `src/core/providers/bootstrap.py` without editing the
built-in bootstrap list.

The primary goal is to reduce merge risk when experimenting with new models or
integrations (for example: a new part classifier).

## When To Use Plugins

Use `CORE_PROVIDER_PLUGINS` when:
- You want to add an experimental provider with minimal conflicts against `main`.
- You want to keep a feature behind env flags and roll it out gradually.
- You want to keep heavy/optional dependencies isolated behind lazy imports.

Prefer direct, built-in registration (editing `src/core/providers/*`) when:
- The provider is stable and should ship enabled by default.
- You need strict behavior changes to API routing.

## Plugin Contract

Plugins are loaded best-effort by default. Configuration:
- `CORE_PROVIDER_PLUGINS`: comma/space separated list of plugin tokens.
- `CORE_PROVIDER_PLUGINS_STRICT`: when `true`, import/call errors raise.

Each plugin token can be:
- `pkg.module`
  - Imports the module. The module is expected to self-register providers.
- `pkg.module:bootstrap`
  - Imports the module and calls a callable `bootstrap()` function (recommended).

## Minimal Example (Recommended)

1) Create a module under `src/` so it is importable.

Example path:
- `src/core/provider_plugins/example_classifier.py`

2) Implement a provider and register it in `bootstrap()`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.classifier import ClassifierRequest
from src.core.providers.registry import ProviderRegistry


@dataclass
class ExampleConfig(ProviderConfig):
    default_label: str = "unknown"


class ExampleRulesProvider(BaseProvider[ExampleConfig, Dict[str, Any]]):
    def __init__(self, config: Optional[ExampleConfig] = None):
        cfg = config or ExampleConfig(name="example_rules", provider_type="classifier")
        super().__init__(cfg)

    async def _process_impl(self, request: Any, **kwargs: Any) -> Dict[str, Any]:
        if not isinstance(request, ClassifierRequest):
            raise TypeError("expects ClassifierRequest")
        return {"status": "ok", "label": self.config.default_label}


def bootstrap() -> None:
    if ProviderRegistry.exists("classifier", "example_rules"):
        return
    ProviderRegistry.register("classifier", "example_rules")(ExampleRulesProvider)
```

Key rule: **your provider must be instantiable with no args**.
`ProviderRegistry.get(domain, name)` will call `provider_cls()` when caching is enabled.

## Enable A Plugin

In `.env` (or runtime env):

```bash
CORE_PROVIDER_PLUGINS="src.core.provider_plugins.example_classifier:bootstrap"
CORE_PROVIDER_PLUGINS_STRICT=false
```

Notes:
- Use `CORE_PROVIDER_PLUGINS_STRICT=true` in CI if you want plugin failures to fail fast.
- Prefer `:bootstrap` tokens so the module import has minimal side effects.

## Verify Runtime Loading

The registry snapshot includes plugin status:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/registry
```

Look for:
- `registry.plugins.configured`
- `registry.plugins.loaded`
- `registry.plugins.errors`

You can also run best-effort health checks:

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/health/providers/health
```

## Testing A Plugin

Recommended unit-test pattern (no real model deps):
- Clear the registry.
- Set env vars with `monkeypatch`.
- Call `bootstrap_core_provider_registry()`.
- Assert the provider exists and can be used.

Example:

```python
def test_plugin_loads(monkeypatch):
    ProviderRegistry.clear()
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS", "src.core.provider_plugins.example_classifier:bootstrap")
    monkeypatch.setenv("CORE_PROVIDER_PLUGINS_STRICT", "true")
    bootstrap_core_provider_registry()
    assert ProviderRegistry.exists("classifier", "example_rules")
```

## Best Practices

- Keep module imports lightweight; do heavy imports inside `_process_impl`.
- Make `_health_check_impl` cheap and deterministic (avoid full model load).
- Add per-provider env flags so you can disable without code changes.
- Use `READINESS_REQUIRED_PROVIDERS` / `READINESS_OPTIONAL_PROVIDERS` for safe rollout.

