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
from src.utils.metrics import (
    core_provider_plugin_bootstrap_duration_seconds,
    core_provider_plugin_bootstrap_total,
    core_provider_plugin_configured,
    core_provider_plugin_errors,
    core_provider_plugin_loaded,
)

_BOOTSTRAPPED = False
_BOOTSTRAP_TS: float | None = None
_PLUGINS_CONFIG: tuple[str, bool] | None = None
_PLUGINS_STATUS: Dict[str, Any] = {
    "enabled": False,
    "strict": False,
    "configured": [],
    "loaded": [],
    "errors": [],
    "registered": {},
    "cache": {
        "reused": False,
        "reason": "init",
        "checked_at": None,
        "missing_registered": [],
    },
}


def reset_core_provider_plugins_state() -> None:
    """Reset plugin loader cache.

    This is primarily intended for tests that call `ProviderRegistry.clear()`
    between cases. In production, plugin bootstrap is expected to run once per
    process.
    """
    global _PLUGINS_CONFIG, _PLUGINS_STATUS
    _PLUGINS_CONFIG = None
    _PLUGINS_STATUS = {
        "enabled": False,
        "strict": False,
        "configured": [],
        "loaded": [],
        "errors": [],
        "registered": {},
        "cache": {
            "reused": False,
            "reason": "reset",
            "checked_at": None,
            "missing_registered": [],
        },
    }


def _parse_plugin_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in re.split(r"[,\s]+", raw) if t.strip()]


def _snapshot_provider_ids() -> set[str]:
    ids: set[str] = set()
    for domain in ProviderRegistry.list_domains():
        for name in ProviderRegistry.list_providers(domain):
            ids.add(f"{domain}/{name}")
    return ids


def _plugins_registry_diagnostics(status: Dict[str, Any]) -> tuple[bool, str, list[str]]:
    loaded = status.get("loaded")
    if loaded and "registered" not in status:
        # Backward compatibility: force one refresh if cache schema changed.
        return False, "cache_schema_mismatch", []

    registered = status.get("registered")
    if not isinstance(registered, dict):
        return False, "invalid_registered_map", []

    missing_registered: list[str] = []
    for provider_ids in registered.values():
        if not isinstance(provider_ids, list):
            return False, "invalid_registered_value", []
        for provider_id in provider_ids:
            if not isinstance(provider_id, str) or "/" not in provider_id:
                return False, "invalid_registered_provider_id", []
            domain, provider_name = provider_id.split("/", 1)
            if not ProviderRegistry.exists(domain, provider_name):
                missing_registered.append(provider_id)

    if missing_registered:
        return False, "missing_registered_provider", sorted(set(missing_registered))
    return True, "intact", []


def _observe_plugin_metrics(
    *,
    result: str,
    configured_count: int,
    loaded_count: int,
    error_count: int,
    duration_seconds: float,
) -> None:
    try:
        core_provider_plugin_bootstrap_total.labels(result=result).inc()
        core_provider_plugin_bootstrap_duration_seconds.labels(
            result=result
        ).observe(duration_seconds)
        core_provider_plugin_configured.set(configured_count)
        core_provider_plugin_loaded.set(loaded_count)
        core_provider_plugin_errors.set(error_count)
    except Exception:
        # Metrics should never affect bootstrap behavior.
        pass


def _build_plugin_summary(status: Dict[str, Any]) -> Dict[str, Any]:
    configured = status.get("configured")
    loaded = status.get("loaded")
    errors = status.get("errors")
    cache = status.get("cache")

    configured_count = len(configured) if isinstance(configured, list) else 0
    loaded_count = len(loaded) if isinstance(loaded, list) else 0
    error_count = len(errors) if isinstance(errors, list) else 0

    missing_registered: list[str] = []
    cache_reused = False
    cache_reason = ""
    if isinstance(cache, dict):
        missing_registered = (
            cache.get("missing_registered")
            if isinstance(cache.get("missing_registered"), list)
            else []
        )
        cache_reused = bool(cache.get("reused", False))
        cache_reason = str(cache.get("reason", ""))

    missing_registered_count = len(missing_registered)
    if error_count > 0:
        overall_status = (
            "error" if configured_count > 0 and loaded_count == 0 else "degraded"
        )
    elif missing_registered_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "ok"

    return {
        "configured_count": configured_count,
        "loaded_count": loaded_count,
        "error_count": error_count,
        "missing_registered_count": missing_registered_count,
        "cache_reused": cache_reused,
        "cache_reason": cache_reason,
        "overall_status": overall_status,
    }


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

    started_at = time.perf_counter()
    checked_at = time.time()
    raw = os.getenv("CORE_PROVIDER_PLUGINS", "").strip()
    strict = (
        os.getenv("CORE_PROVIDER_PLUGINS_STRICT", "false").strip().lower() == "true"
    )
    config = (raw, bool(strict))
    intact, cache_reason, missing_registered = _plugins_registry_diagnostics(
        _PLUGINS_STATUS
    )

    if _PLUGINS_CONFIG == config and intact:
        cache = {
            "reused": True,
            "reason": "config_match",
            "checked_at": checked_at,
            "missing_registered": [],
        }
        _PLUGINS_STATUS["cache"] = cache
        _observe_plugin_metrics(
            result="cache_hit",
            configured_count=len(_PLUGINS_STATUS.get("configured", [])),
            loaded_count=len(_PLUGINS_STATUS.get("loaded", [])),
            error_count=len(_PLUGINS_STATUS.get("errors", [])),
            duration_seconds=(time.perf_counter() - started_at),
        )
        return dict(_PLUGINS_STATUS)

    tokens = _parse_plugin_list(raw)
    loaded: list[str] = []
    errors: list[Dict[str, str]] = []
    registered: Dict[str, list[str]] = {}

    if _PLUGINS_CONFIG is None:
        reload_reason = "first_load"
    elif _PLUGINS_CONFIG != config:
        reload_reason = "config_changed"
    elif missing_registered:
        reload_reason = "missing_registered_provider"
    else:
        reload_reason = cache_reason

    for token in tokens:
        module_name = token
        func_name = ""
        if ":" in token:
            module_name, func_name = token.split(":", 1)
            module_name = module_name.strip()
            func_name = func_name.strip()
        try:
            before_ids = _snapshot_provider_ids()
            module = importlib.import_module(module_name)
            if func_name:
                func = getattr(module, func_name)
                if not callable(func):
                    raise TypeError(
                        f"Plugin bootstrap target not callable: {module_name}:{func_name}"
                    )
                func()
            after_ids = _snapshot_provider_ids()
            loaded.append(token)
            registered[token] = sorted(after_ids - before_ids)
        except Exception as exc:  # noqa: BLE001
            errors.append({"plugin": token, "error": f"{type(exc).__name__}: {exc}"})
            if strict:
                _observe_plugin_metrics(
                    result="strict_error",
                    configured_count=len(tokens),
                    loaded_count=len(loaded),
                    error_count=len(errors),
                    duration_seconds=(time.perf_counter() - started_at),
                )
                raise

    _PLUGINS_STATUS = {
        "enabled": bool(tokens),
        "strict": bool(strict),
        "configured": list(tokens),
        "loaded": loaded,
        "errors": errors,
        "registered": registered,
        "cache": {
            "reused": False,
            "reason": reload_reason,
            "checked_at": checked_at,
            "missing_registered": missing_registered,
        },
    }
    _PLUGINS_CONFIG = config

    _observe_plugin_metrics(
        result="reload_partial" if errors else "reload_ok",
        configured_count=len(tokens),
        loaded_count=len(loaded),
        error_count=len(errors),
        duration_seconds=(time.perf_counter() - started_at),
    )
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
    plugins = dict(_PLUGINS_STATUS)
    plugins["summary"] = _build_plugin_summary(plugins)
    return {
        "bootstrapped": _BOOTSTRAPPED,
        "bootstrap_timestamp": _BOOTSTRAP_TS,
        "plugins": plugins,
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
    if lazy_bootstrap and (not _BOOTSTRAPPED or not ProviderRegistry.list_domains()):
        bootstrap_core_provider_registry()
    return _build_snapshot()
