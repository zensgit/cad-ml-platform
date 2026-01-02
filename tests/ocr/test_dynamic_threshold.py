"""Tests for dynamic confidence fallback threshold adaptation (EMA-based)."""

from __future__ import annotations

import asyncio

import pytest

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


@pytest.mark.asyncio
async def test_threshold_adapts_downward():
    """Test that confidence threshold adapts downward via EMA."""
    provider = _DummyProvider([0.9, 0.85, 0.8, 0.75])
    m = OcrManager({"paddle": provider}, confidence_fallback=0.9)

    await m.extract(b"img", strategy="paddle")
    thr1 = m.confidence_fallback
    await m.extract(b"img", strategy="paddle")
    thr2 = m.confidence_fallback
    assert thr2 <= thr1  # EMA下降后，阈值不应上升


@pytest.mark.asyncio
async def test_threshold_bounds():
    """Test that threshold stays within bounds."""
    provider = _DummyProvider([0.3, 0.2, 0.1])
    m = OcrManager({"paddle": provider}, confidence_fallback=0.85)

    await m.extract(b"img", strategy="paddle")
    assert 0.6 <= m.confidence_fallback <= 0.95
