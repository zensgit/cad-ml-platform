"""Provider registry for discovery and instance creation."""

from __future__ import annotations

from threading import RLock
from typing import Dict, List, Type

from src.core.providers.base import BaseProvider


class ProviderRegistry:
    """Simple type-safe registry for provider classes."""

    _providers: Dict[str, Dict[str, Type[BaseProvider]]] = {}
    _lock = RLock()

    @classmethod
    def register(cls, domain: str, provider_name: str):
        """Decorator to register a provider class.

        Example:
            @ProviderRegistry.register("vision", "deepseek")
            class DeepSeekProvider(...):
                ...
        """

        def _decorator(provider_cls: Type[BaseProvider]) -> Type[BaseProvider]:
            with cls._lock:
                domain_map = cls._providers.setdefault(domain, {})
                if provider_name in domain_map:
                    raise ValueError(
                        f"Provider already registered: {domain}/{provider_name}"
                    )
                domain_map[provider_name] = provider_cls
            return provider_cls

        return _decorator

    @classmethod
    def get_provider_class(cls, domain: str, provider_name: str) -> Type[BaseProvider]:
        with cls._lock:
            domain_map = cls._providers.get(domain, {})
            provider_cls = domain_map.get(provider_name)
        if provider_cls is None:
            raise KeyError(f"Provider not found: {domain}/{provider_name}")
        return provider_cls

    @classmethod
    def get(cls, domain: str, provider_name: str, *args, **kwargs) -> BaseProvider:
        """Create a provider instance from registry."""
        provider_cls = cls.get_provider_class(domain, provider_name)
        return provider_cls(*args, **kwargs)

    @classmethod
    def list_domains(cls) -> List[str]:
        with cls._lock:
            return sorted(cls._providers.keys())

    @classmethod
    def list_providers(cls, domain: str) -> List[str]:
        with cls._lock:
            return sorted(cls._providers.get(domain, {}).keys())

    @classmethod
    def exists(cls, domain: str, provider_name: str) -> bool:
        with cls._lock:
            return provider_name in cls._providers.get(domain, {})

    @classmethod
    def unregister(cls, domain: str, provider_name: str) -> bool:
        with cls._lock:
            domain_map = cls._providers.get(domain)
            if not domain_map or provider_name not in domain_map:
                return False
            del domain_map[provider_name]
            if not domain_map:
                del cls._providers[domain]
            return True

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._providers.clear()
