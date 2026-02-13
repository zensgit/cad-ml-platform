"""Provider registry for discovery and instance creation."""

from __future__ import annotations

import os
from threading import RLock
from typing import Dict, List, Type

from src.core.providers.base import BaseProvider


class ProviderRegistry:
    """Simple type-safe registry for provider classes."""

    _providers: Dict[str, Dict[str, Type[BaseProvider]]] = {}
    _instances: Dict[str, Dict[str, BaseProvider]] = {}
    _lock = RLock()

    @staticmethod
    def _normalize_token(value: str, field_name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")
        token = value.strip()
        if not token:
            raise ValueError(f"{field_name} must be a non-empty string")
        if "/" in token or ":" in token:
            raise ValueError(f"{field_name} cannot contain '/' or ':'")
        return token

    @staticmethod
    def _validate_provider_class(provider_cls: Type[BaseProvider]) -> None:
        if not isinstance(provider_cls, type) or not issubclass(provider_cls, BaseProvider):
            raise TypeError(
                "Registered provider class must inherit BaseProvider: "
                f"{provider_cls!r}"
            )

    @classmethod
    def normalize_provider_id(cls, domain: str, provider_name: str) -> tuple[str, str]:
        """Normalize and validate provider identifiers."""
        return (
            cls._normalize_token(domain, "domain"),
            cls._normalize_token(provider_name, "provider_name"),
        )

    @classmethod
    def register(cls, domain: str, provider_name: str):
        """Decorator to register a provider class.

        Example:
            @ProviderRegistry.register("vision", "deepseek")
            class DeepSeekProvider(...):
                ...
        """
        normalized_domain, normalized_provider_name = cls.normalize_provider_id(
            domain, provider_name
        )

        def _decorator(provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
            cls._validate_provider_class(provider_cls)
            with cls._lock:
                domain_map = cls._providers.setdefault(normalized_domain, {})
                if normalized_provider_name in domain_map:
                    raise ValueError(
                        "Provider already registered: "
                        f"{normalized_domain}/{normalized_provider_name}"
                    )
                domain_map[normalized_provider_name] = provider_cls
            return provider_cls

        return _decorator

    @classmethod
    def _cache_enabled(cls) -> bool:
        raw = os.getenv("PROVIDER_REGISTRY_CACHE_ENABLED", "true").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    @classmethod
    def get_provider_class(cls, domain: str, provider_name: str) -> Type[BaseProvider]:
        normalized_domain, normalized_provider_name = cls.normalize_provider_id(
            domain, provider_name
        )
        with cls._lock:
            domain_map = cls._providers.get(normalized_domain, {})
            provider_cls = domain_map.get(normalized_provider_name)
        if provider_cls is None:
            raise KeyError(
                f"Provider not found: {normalized_domain}/{normalized_provider_name}"
            )
        return provider_cls

    @classmethod
    def get(cls, domain: str, provider_name: str, *args, **kwargs) -> BaseProvider:
        """Create or fetch a provider instance from registry.

        Behavior:
        - If *args/**kwargs are provided, always create a new instance (no caching).
        - If no *args/**kwargs are provided, return a cached singleton instance by
          default (controlled via PROVIDER_REGISTRY_CACHE_ENABLED).
        """
        normalized_domain, normalized_provider_name = cls.normalize_provider_id(
            domain, provider_name
        )
        provider_cls = cls.get_provider_class(normalized_domain, normalized_provider_name)
        if args or kwargs or not cls._cache_enabled():
            return provider_cls(*args, **kwargs)

        with cls._lock:
            domain_instances = cls._instances.setdefault(normalized_domain, {})
            inst = domain_instances.get(normalized_provider_name)
            if inst is None or type(inst) is not provider_cls:
                inst = provider_cls()
                domain_instances[normalized_provider_name] = inst
            return inst

    @classmethod
    def list_domains(cls) -> List[str]:
        with cls._lock:
            return sorted(cls._providers.keys())

    @classmethod
    def list_providers(cls, domain: str) -> List[str]:
        normalized_domain = cls._normalize_token(domain, "domain")
        with cls._lock:
            return sorted(cls._providers.get(normalized_domain, {}).keys())

    @classmethod
    def exists(cls, domain: str, provider_name: str) -> bool:
        normalized_domain, normalized_provider_name = cls.normalize_provider_id(
            domain, provider_name
        )
        with cls._lock:
            return normalized_provider_name in cls._providers.get(normalized_domain, {})

    @classmethod
    def unregister(cls, domain: str, provider_name: str) -> bool:
        normalized_domain, normalized_provider_name = cls.normalize_provider_id(
            domain, provider_name
        )
        with cls._lock:
            domain_map = cls._providers.get(normalized_domain)
            if not domain_map or normalized_provider_name not in domain_map:
                return False
            del domain_map[normalized_provider_name]
            if not domain_map:
                del cls._providers[normalized_domain]
            inst_map = cls._instances.get(normalized_domain)
            if inst_map and normalized_provider_name in inst_map:
                del inst_map[normalized_provider_name]
            if inst_map is not None and not inst_map:
                cls._instances.pop(normalized_domain, None)
            return True

    @classmethod
    def clear_instances(cls) -> None:
        """Clear cached provider instances (keeps provider class registrations)."""
        with cls._lock:
            cls._instances.clear()

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._providers.clear()
            cls._instances.clear()
