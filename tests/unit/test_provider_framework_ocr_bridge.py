from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.core.ocr.base import OcrClient, OcrResult, TitleBlock
from src.core.providers import ProviderRegistry
from src.core.providers.ocr import (
    OcrProviderAdapter,
    OcrProviderConfig,
    bootstrap_core_ocr_providers,
)


@dataclass
class FakeOcrProvider(OcrClient):
    name: str = "fake"

    async def warmup(self) -> None:
        return None

    async def extract(
        self, image_bytes: bytes, trace_id: str | None = None
    ) -> OcrResult:
        return OcrResult(
            text=f"len={len(image_bytes)}",
            dimensions=[],
            symbols=[],
            title_block=TitleBlock(),
            trace_id=trace_id,
            confidence=0.99,
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _clear_registry():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


@pytest.mark.asyncio
async def test_ocr_adapter_process_and_health_check():
    provider = OcrProviderAdapter(
        config=OcrProviderConfig(
            name="ocr_fake",
            provider_type="ocr",
            provider_name="fake",
            trace_id="trace-default",
        ),
        wrapped_provider=FakeOcrProvider(),
    )
    result = await provider.process(b"abc")
    assert result.text == "len=3"
    assert result.trace_id == "trace-default"
    ok = await provider.health_check()
    assert ok is True
    assert provider.status.value == "healthy"


@pytest.mark.asyncio
async def test_ocr_adapter_trace_override():
    provider = OcrProviderAdapter(
        config=OcrProviderConfig(
            name="ocr_fake",
            provider_type="ocr",
            provider_name="fake",
            trace_id="trace-default",
        ),
        wrapped_provider=FakeOcrProvider(),
    )
    result = await provider.process(b"abc", trace_id="trace-override")
    assert result.trace_id == "trace-override"


@pytest.mark.asyncio
async def test_ocr_adapter_rejects_invalid_request_type():
    provider = OcrProviderAdapter(
        config=OcrProviderConfig(
            name="ocr_fake",
            provider_type="ocr",
            provider_name="fake",
        ),
        wrapped_provider=FakeOcrProvider(),
    )
    with pytest.raises(TypeError, match="expects raw image bytes"):
        await provider.process({"not": "bytes"})


def test_bootstrap_core_ocr_providers_registers_expected_entries():
    bootstrap_core_ocr_providers()
    assert ProviderRegistry.exists("ocr", "paddle") is True
    assert ProviderRegistry.exists("ocr", "deepseek_hf") is True
