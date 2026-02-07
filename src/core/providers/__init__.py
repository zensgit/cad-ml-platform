"""Provider Framework.

Provides unified base classes and registry for all providers:
- BaseProvider: Common functionality for health checks, metrics, lifecycle
- ProviderRegistry: Factory and discovery for provider instances
- Bootstrap helpers: Register built-in provider adapters used by the core app

Example usage:
    from src.core.providers import BaseProvider, ProviderRegistry

    @ProviderRegistry.register("vision", "deepseek")
    class DeepSeekProvider(BaseProvider[VisionConfig, VisionResult]):
        ...

    # Get provider
    provider = ProviderRegistry.get("vision", "deepseek")
    if await provider.health_check():
        result = await provider.process(...)
"""

from src.core.providers.base import BaseProvider, ProviderConfig, ProviderStatus
from src.core.providers.bootstrap import (
    bootstrap_core_provider_registry,
    get_core_provider_registry_snapshot,
)
from src.core.providers.ocr import (
    OcrProviderAdapter,
    OcrProviderConfig,
    bootstrap_core_ocr_providers,
)
from src.core.providers.registry import ProviderRegistry
from src.core.providers.vision import (
    VisionProviderAdapter,
    VisionProviderConfig,
    bootstrap_core_vision_providers,
)

__all__ = [
    "BaseProvider",
    "ProviderConfig",
    "ProviderStatus",
    "ProviderRegistry",
    "OcrProviderConfig",
    "OcrProviderAdapter",
    "bootstrap_core_ocr_providers",
    "VisionProviderConfig",
    "VisionProviderAdapter",
    "bootstrap_core_vision_providers",
    "bootstrap_core_provider_registry",
    "get_core_provider_registry_snapshot",
]
