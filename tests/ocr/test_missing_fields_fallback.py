"""Test missing-fields fallback trigger in OcrManager."""

import pytest
from src.core.ocr.manager import OcrManager
from src.core.ocr.base import OcrResult


class StubProvider:
    name = "paddle"
    async def warmup(self):
        pass
    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
        # raw text contains dimension tokens but returns empty parsed lists
        return OcrResult(text="Φ20 R5 M10 Ra3.2", confidence=0.9)
    async def health_check(self):
        return True


class DeepSeekStub:
    name = "deepseek_hf"
    async def warmup(self):
        pass
    async def extract(self, image_bytes: bytes, trace_id: str | None = None) -> OcrResult:
        # simulate parsed extraction
        from src.core.ocr.base import DimensionInfo, DimensionType, SymbolInfo, SymbolType
        return OcrResult(
            text="Φ20 R5 M10 Ra3.2",
            confidence=0.95,
            dimensions=[
                DimensionInfo(type=DimensionType.diameter, value=20.0),
                DimensionInfo(type=DimensionType.radius, value=5.0),
                DimensionInfo(type=DimensionType.thread, value=10.0, pitch=1.5),
            ],
            symbols=[SymbolInfo(type=SymbolType.surface_roughness, value="3.2")],
        )
    async def health_check(self):
        return True


@pytest.mark.asyncio
async def test_missing_fields_triggers_fallback(monkeypatch):
    mgr = OcrManager(confidence_fallback=0.85)
    mgr.register_provider("paddle", StubProvider())
    mgr.register_provider("deepseek_hf", DeepSeekStub())
    result = await mgr.extract(b"img", strategy="auto")
    # Because paddle returned no parsed dimensions/symbols -> fallback to deepseek expected
    assert result.provider == "deepseek_hf"
    assert result.fallback_level == "missing_fields"
    assert result.completeness == 1.0  # deepseek stub completeness
