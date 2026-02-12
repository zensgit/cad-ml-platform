"""Vision bridge for the core provider framework.

This module adapts existing ``src.core.vision`` providers into the generic
``src.core.providers`` abstractions with minimal behavior changes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry
from src.core.vision.base import VisionDescription, VisionProvider
from src.core.vision.factory import create_vision_provider
from src.core.vision.providers.deepseek_stub import create_stub_provider


@dataclass
class VisionProviderConfig(ProviderConfig):
    """Configuration for vision adapter providers."""

    provider_name: str = "stub"
    include_description_default: bool = True
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)


class VisionProviderAdapter(BaseProvider[VisionProviderConfig, VisionDescription]):
    """Adapter that exposes a vision provider through ``BaseProvider``."""

    def __init__(
        self,
        config: VisionProviderConfig,
        wrapped_provider: Optional[VisionProvider] = None,
    ):
        super().__init__(config)
        self._wrapped_provider = wrapped_provider or create_vision_provider(
            provider_type=config.provider_name,
            fallback_to_stub=True,
            **config.provider_kwargs,
        )

    async def _process_impl(self, request: Any, **kwargs: Any) -> VisionDescription:
        if not isinstance(request, (bytes, bytearray)):
            raise TypeError("VisionProviderAdapter expects raw image bytes as request")
        image_bytes = bytes(request)
        if not image_bytes:
            raise ValueError("image bytes cannot be empty")
        include_description = kwargs.get(
            "include_description",
            self.config.include_description_default,
        )
        return await self._wrapped_provider.analyze_image(
            image_data=image_bytes,
            include_description=bool(include_description),
        )

    async def _health_check_impl(self) -> bool:
        maybe_health = getattr(self._wrapped_provider, "health_check", None)
        if callable(maybe_health):
            result = maybe_health()
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        probe = await self._wrapped_provider.analyze_image(
            image_data=b"provider-health-check",
            include_description=False,
        )
        return isinstance(probe, VisionDescription)


def bootstrap_core_vision_providers() -> None:
    """Register minimal built-in vision providers in ``ProviderRegistry``.

    Registers:
    - ``vision/stub``
    - ``vision/deepseek_stub``
    """

    if not ProviderRegistry.exists("vision", "stub"):

        @ProviderRegistry.register("vision", "stub")
        class StubVisionCoreProvider(VisionProviderAdapter):
            def __init__(self, config: Optional[VisionProviderConfig] = None):
                cfg = config or VisionProviderConfig(
                    name="stub",
                    provider_type="vision",
                    provider_name="stub",
                )
                super().__init__(
                    config=cfg,
                    wrapped_provider=create_stub_provider(simulate_latency_ms=0),
                )

    if not ProviderRegistry.exists("vision", "deepseek_stub"):

        @ProviderRegistry.register("vision", "deepseek_stub")
        class DeepSeekStubVisionCoreProvider(VisionProviderAdapter):
            def __init__(self, config: Optional[VisionProviderConfig] = None):
                cfg = config or VisionProviderConfig(
                    name="deepseek_stub",
                    provider_type="vision",
                    provider_name="deepseek_stub",
                )
                super().__init__(
                    config=cfg,
                    wrapped_provider=create_stub_provider(simulate_latency_ms=0),
                )
