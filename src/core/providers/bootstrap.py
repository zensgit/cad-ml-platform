"""Bootstrap helpers for core provider registry."""

from __future__ import annotations

import importlib
import os
import re
import time
from typing import Any, Dict

from src.core.providers.ocr import bootstrap_core_ocr_providers
from src.core.providers.registry import ProviderRegistry
from src.core.providers.vision import bootstrap_core_vision_providers
from src.core.providers.classifier import bootstrap_core_classifier_providers
from src.core.providers.knowledge import bootstrap_core_knowledge_providers

_BOOTSTRAPPED = False
_BOOTSTRAP_TS: float | None = None
_PLUGINS_CONFIG: tuple[str, bool] | None = None
_PLUGINS_STATUS: Dict[str, Any] = {
    "enabled": False,
    "strict": False,
    "configured": [],
    "loaded": [],
    "errors": [],
}


def _parse_plugin_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]


def bootstrap_core_provider_plugins() -> Dict[str, Any]:
    """Best-effort plugin hook for registering additional providers.

    Environment:
    - CORE_PROVIDER_PLUGINS: comma/space separated plugin list
      - "pkg.module" imports the module (expected to self-register providers)
      - "pkg.module:bootstrap" imports the module and calls a bootstrap function
    - CORE_PROVIDER_PLUGINS_STRICT: when true, import/call errors raise

    This is intended to reduce merge risk for experimental providers: new models
    can be added as separate modules and enabled via env without modifying the
    built-in bootstrap list.
    """
    global _PLUGINS_CONFIG, _PLUGINS_STATUS

    raw = os.getenv("CORE_PROVIDER_PLUGINS", "").strip()
    strict = (
        os.getenv("CORE_PROVIDER_PLUGINS_STRICT", "false").strip().lower() == "true"
    )
    config = (raw, bool(strict))
    if _PLUGINS_CONFIG == config:
        return dict(_PLUGINS_STATUS)

    tokens = _parse_plugin_list(raw)
    loaded: list[str] = []
    errors: list[Dict[str, str]] = []

    for token in tokens:
        module_name = token
        func_name = ""
        if ":" in token:
            module_name, func_name = token.split(":", 1)
            module_name = module_name.strip()
            func_name = func_name.strip()
        try:
            module = importlib.import_module(module_name)
            if func_name:
                func = getattr(module, func_name)
                if not callable(func):
                    raise TypeError(
                        f"Plugin bootstrap target not callable: {module_name}:{func_name}"
                    )
                func()
            loaded.append(token)
        except Exception as exc:  # noqa: BLE001
            errors.append({"plugin": token, "error": f"{type(exc).__name__}: {exc}"})
            if strict:
                raise

    _PLUGINS_STATUS = {
        "enabled": bool(tokens),
        "strict": bool(strict),
        "configured": list(tokens),
        "loaded": loaded,
        "errors": errors,
    }
    _PLUGINS_CONFIG = config
    return dict(_PLUGINS_STATUS)


def _build_snapshot() -> Dict[str, Any]:
    domains = ProviderRegistry.list_domains()
    providers = {domain: ProviderRegistry.list_providers(domain) for domain in domains}
    provider_classes: Dict[str, Dict[str, str]] = {}
    for domain, names in providers.items():
        class_map: Dict[str, str] = {}
        for name in names:
            try:
                cls = ProviderRegistry.get_provider_class(domain, name)
                class_map[name] = f"{cls.__module__}.{cls.__qualname__}"
            except Exception:
                # Best-effort only; snapshot should never fail because of metadata.
                class_map[name] = "unknown"
        provider_classes[domain] = class_map
    total_providers = sum(len(items) for items in providers.values())
    return {
        "bootstrapped": _BOOTSTRAPPED,
        "bootstrap_timestamp": _BOOTSTRAP_TS,
        "plugins": dict(_PLUGINS_STATUS),
        "total_domains": len(domains),
        "total_providers": total_providers,
        "domains": domains,
        "providers": providers,
        "provider_classes": provider_classes,
    }


def bootstrap_core_provider_registry() -> Dict[str, Any]:
    """Register built-in provider adapters and return current snapshot."""
    global _BOOTSTRAPPED, _BOOTSTRAP_TS

    bootstrap_core_vision_providers()
    bootstrap_core_ocr_providers()
    bootstrap_core_classifier_providers()
    bootstrap_core_knowledge_providers()
    # Plugin bootstrap is best-effort by default; strict mode is handled inside
    # `bootstrap_core_provider_plugins()` and should raise to callers.
    bootstrap_core_provider_plugins()

    if not _BOOTSTRAPPED:
        _BOOTSTRAPPED = True
        _BOOTSTRAP_TS = time.time()

    return _build_snapshot()


def get_core_provider_registry_snapshot(lazy_bootstrap: bool = True) -> Dict[str, Any]:
    """Return a serializable snapshot of registry status."""
    if lazy_bootstrap and not _BOOTSTRAPPED:
        bootstrap_core_provider_registry()
    return _build_snapshot()
