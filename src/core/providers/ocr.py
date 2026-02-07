"""OCR bridge for the core provider framework."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.core.ocr.base import OcrClient, OcrResult
from src.core.providers.base import BaseProvider, ProviderConfig
from src.core.providers.registry import ProviderRegistry


@dataclass
class OcrProviderConfig(ProviderConfig):
    """Configuration for OCR adapter providers."""

    provider_name: str = "paddle"
    trace_id: Optional[str] = None
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)


class OcrProviderAdapter(BaseProvider[OcrProviderConfig, OcrResult]):
    """Adapter that exposes ``OcrClient`` through ``BaseProvider``."""

    def __init__(
        self,
        config: OcrProviderConfig,
        wrapped_provider: Optional[OcrClient] = None,
    ):
        super().__init__(config)
        self._wrapped_provider = wrapped_provider or self._build_default_provider(
            config
        )

    def _build_default_provider(self, config: OcrProviderConfig) -> OcrClient:
        provider_name = config.provider_name.lower()
        kwargs = dict(config.provider_kwargs)
        if provider_name == "deepseek_hf":
            from src.core.ocr.providers.deepseek_hf import DeepSeekHfProvider

            return DeepSeekHfProvider(**kwargs)
        if provider_name == "paddle":
            from src.core.ocr.providers.paddle import PaddleOcrProvider

            return PaddleOcrProvider(**kwargs)
        raise ValueError(
            f"Unsupported OCR provider for adapter: {config.provider_name}"
        )

    async def _process_impl(self, request: Any, **kwargs: Any) -> OcrResult:
        if not isinstance(request, (bytes, bytearray)):
            raise TypeError("OcrProviderAdapter expects raw image bytes as request")
        image_bytes = bytes(request)
        if not image_bytes:
            raise ValueError("image bytes cannot be empty")
        trace_id = kwargs.get("trace_id", self.config.trace_id)
        return await self._wrapped_provider.extract(
            image_bytes=image_bytes, trace_id=trace_id
        )

    async def _health_check_impl(self) -> bool:
        maybe_health = getattr(self._wrapped_provider, "health_check", None)
        if callable(maybe_health):
            result = maybe_health()
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        return True


def bootstrap_core_ocr_providers() -> None:
    """Register built-in OCR providers in ``ProviderRegistry``."""

    if not ProviderRegistry.exists("ocr", "paddle"):

        @ProviderRegistry.register("ocr", "paddle")
        class PaddleCoreProvider(OcrProviderAdapter):
            def __init__(self, config: Optional[OcrProviderConfig] = None):
                cfg = config or OcrProviderConfig(
                    name="paddle",
                    provider_type="ocr",
                    provider_name="paddle",
                )
                super().__init__(config=cfg)

    if not ProviderRegistry.exists("ocr", "deepseek_hf"):

        @ProviderRegistry.register("ocr", "deepseek_hf")
        class DeepSeekHfCoreProvider(OcrProviderAdapter):
            def __init__(self, config: Optional[OcrProviderConfig] = None):
                cfg = config or OcrProviderConfig(
                    name="deepseek_hf",
                    provider_type="ocr",
                    provider_name="deepseek_hf",
                )
                super().__init__(config=cfg)
