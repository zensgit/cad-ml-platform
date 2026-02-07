"""Bootstrap helpers for core provider registry."""

from __future__ import annotations

import time
from typing import Any, Dict

from src.core.providers.ocr import bootstrap_core_ocr_providers
from src.core.providers.registry import ProviderRegistry
from src.core.providers.vision import bootstrap_core_vision_providers
from src.core.providers.classifier import bootstrap_core_classifier_providers

_BOOTSTRAPPED = False
_BOOTSTRAP_TS: float | None = None


def _build_snapshot() -> Dict[str, Any]:
    domains = ProviderRegistry.list_domains()
    providers = {domain: ProviderRegistry.list_providers(domain) for domain in domains}
    total_providers = sum(len(items) for items in providers.values())
    return {
        "bootstrapped": _BOOTSTRAPPED,
        "bootstrap_timestamp": _BOOTSTRAP_TS,
        "total_domains": len(domains),
        "total_providers": total_providers,
        "domains": domains,
        "providers": providers,
    }


def bootstrap_core_provider_registry() -> Dict[str, Any]:
    """Register built-in provider adapters and return current snapshot."""
    global _BOOTSTRAPPED, _BOOTSTRAP_TS

    bootstrap_core_vision_providers()
    bootstrap_core_ocr_providers()
    bootstrap_core_classifier_providers()

    if not _BOOTSTRAPPED:
        _BOOTSTRAPPED = True
        _BOOTSTRAP_TS = time.time()

    return _build_snapshot()


def get_core_provider_registry_snapshot(lazy_bootstrap: bool = True) -> Dict[str, Any]:
    """Return a serializable snapshot of registry status."""
    if lazy_bootstrap and not _BOOTSTRAPPED:
        bootstrap_core_provider_registry()
    return _build_snapshot()
