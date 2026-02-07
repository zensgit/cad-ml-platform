"""Provider Framework.

Provides unified base classes and registry for all providers:
- BaseProvider: Common functionality for health checks, metrics, lifecycle
- ProviderRegistry: Factory and discovery for provider instances
- ConfigurableProvider: Integration with ConfigManager

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
from src.core.providers.registry import ProviderRegistry

__all__ = [
    "BaseProvider",
    "ProviderConfig",
    "ProviderStatus",
    "ProviderRegistry",
]
