"""Tests for dynamic confidence fallback threshold adaptation (EMA-based)."""

from src.core.ocr.base import OcrResult
from src.core.ocr.manager import OcrManager


class _DummyProvider:
    name = "dummy"

    def __init__(self, confidences):
        self._conf = list(confidences)

    async def warmup(self):
        return None

    async def extract(self, image_bytes: bytes, trace_id: str | None = None):
        # pop or reuse last
        c = self._conf.pop(0) if self._conf else 0.8
        return OcrResult(text="Φ10 R5", confidence=c, dimensions=[], symbols=[])

    async def health_check(self) -> bool:
        return True


async def _run(manager: OcrManager, data):
    for _ in data:
        await manager.extract(b"img", strategy="paddle")


def test_threshold_adapts_downward():
    provider = _DummyProvider([0.9, 0.85, 0.8, 0.75])
    m = OcrManager({"paddle": provider}, confidence_fallback=0.9)
    import asyncio

    asyncio.get_event_loop().run_until_complete(m.extract(b"img", strategy="paddle"))
    thr1 = m.confidence_fallback
    asyncio.get_event_loop().run_until_complete(m.extract(b"img", strategy="paddle"))
    thr2 = m.confidence_fallback
    assert thr2 <= thr1  # EMA下降后，阈值不应上升


def test_threshold_bounds():
    provider = _DummyProvider([0.3, 0.2, 0.1])
    m = OcrManager({"paddle": provider}, confidence_fallback=0.85)
    import asyncio

    asyncio.get_event_loop().run_until_complete(m.extract(b"img", strategy="paddle"))
    assert 0.6 <= m.confidence_fallback <= 0.95
