from __future__ import annotations

import pytest

from src.core.providers import ProviderRegistry
from src.core.providers.vision import (
    VisionProviderAdapter,
    VisionProviderConfig,
    bootstrap_core_vision_providers,
)
from src.core.vision.providers.deepseek_stub import create_stub_provider


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


@pytest.mark.asyncio
async def test_vision_adapter_process_and_health_check():
    provider = VisionProviderAdapter(
        config=VisionProviderConfig(
            name="vision_stub_adapter",
            provider_type="vision",
            provider_name="stub",
            include_description_default=False,
        ),
        wrapped_provider=create_stub_provider(simulate_latency_ms=0),
    )
    result = await provider.process(b"fake-image-bytes")
    assert result.summary == "Image processed (OCR-only mode)"
    ok = await provider.health_check()
    assert ok is True
    assert provider.status.value == "healthy"


@pytest.mark.asyncio
async def test_vision_adapter_rejects_invalid_request_type():
    provider = VisionProviderAdapter(
        config=VisionProviderConfig(
            name="vision_stub_adapter",
            provider_type="vision",
            provider_name="stub",
        ),
        wrapped_provider=create_stub_provider(simulate_latency_ms=0),
    )
    with pytest.raises(TypeError, match="expects raw image bytes"):
        await provider.process({"not": "bytes"})


@pytest.mark.asyncio
async def test_bootstrap_core_vision_providers_registers_stub_aliases():
    bootstrap_core_vision_providers()
    assert ProviderRegistry.exists("vision", "stub") is True
    assert ProviderRegistry.exists("vision", "deepseek_stub") is True

    provider = ProviderRegistry.get("vision", "deepseek_stub")
    result = await provider.process(b"img", include_description=True)
    assert "mechanical engineering drawing" in result.summary
