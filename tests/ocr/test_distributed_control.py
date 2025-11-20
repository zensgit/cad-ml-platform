"""Tests for rate limiter and circuit breaker integrated via OcrManager."""

import asyncio

import pytest

from src.core.ocr.base import OcrResult
from src.core.ocr.manager import OcrManager


class _SlowFailProvider:
    name = "paddle"

    def __init__(self, fail_times=3):
        self.fail = fail_times

    async def warmup(self):
        return None

    async def extract(self, image_bytes: bytes, trace_id: str | None = None):
        if self.fail > 0:
            self.fail -= 1
            raise RuntimeError("simulated failure")
        return OcrResult(text="Φ10", confidence=0.9, dimensions=[], symbols=[])

    async def health_check(self) -> bool:
        return True


def test_rate_limit_blocks(monkeypatch):
    # Patch RateLimiter to a tiny burst/qps
    from src.utils.rate_limiter import RateLimiter

    class _TinyRL(RateLimiter):
        def __init__(self, key, qps=1.0, burst=1):
            super().__init__(key, qps, burst)

    provider = _SlowFailProvider(fail_times=0)
    m = OcrManager({"paddle": provider})
    m._rate_limiters = {"paddle": _TinyRL("paddle", qps=0.0001, burst=1)}
    # First allowed, second likely blocked
    loop = asyncio.get_event_loop()
    loop.run_until_complete(m.extract(b"img", strategy="paddle"))
    with pytest.raises(Exception):
        loop.run_until_complete(m.extract(b"img", strategy="paddle"))


def test_circuit_opens_and_blocks():
    provider = _SlowFailProvider(fail_times=2)
    m = OcrManager({"paddle": provider})
    loop = asyncio.get_event_loop()
    # First call fails and opens breaker
    with pytest.raises(Exception):
        loop.run_until_complete(m.extract(b"img", strategy="paddle"))
    # Immediately second call should be blocked by circuit (open)
    with pytest.raises(Exception):
        loop.run_until_complete(m.extract(b"img", strategy="paddle"))
