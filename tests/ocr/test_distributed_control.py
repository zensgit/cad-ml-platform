"""Tests for rate limiter and circuit breaker integrated via OcrManager."""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_rate_limit_blocks(monkeypatch):
    # Patch RateLimiter to a tiny burst/qps
    # Use unique key to avoid polluting Redis state for other tests
    import uuid
    from src.utils.rate_limiter import RateLimiter

    unique_key = f"paddle_test_{uuid.uuid4().hex[:8]}"

    class _TinyRL(RateLimiter):
        def __init__(self, key, qps=1.0, burst=1):
            super().__init__(key, qps, burst)

    provider = _SlowFailProvider(fail_times=0)
    m = OcrManager({"paddle": provider})
    m._rate_limiters = {"paddle": _TinyRL(unique_key, qps=0.0001, burst=1)}
    # First allowed, second likely blocked
    await m.extract(b"img", strategy="paddle")
    with pytest.raises(Exception):
        await m.extract(b"img", strategy="paddle")


@pytest.mark.asyncio
async def test_circuit_opens_and_blocks():
    import uuid
    # Use unique image bytes to avoid cache hits from other tests
    unique_img = f"circuit_test_{uuid.uuid4()}".encode()
    provider = _SlowFailProvider(fail_times=2)
    m = OcrManager({"paddle": provider})
    # First call fails and opens breaker
    with pytest.raises(Exception):
        await m.extract(unique_img, strategy="paddle")
    # Immediately second call should be blocked by circuit (open)
    with pytest.raises(Exception):
        await m.extract(unique_img, strategy="paddle")
