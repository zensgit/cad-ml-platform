"""Simple in-memory token bucket rate limiter (per IP + endpoint)."""
from __future__ import annotations

import time
from typing import Dict, Tuple

from fastapi import HTTPException, Request

RATE_LIMIT_QPS = float(int(__import__("os").getenv("RATE_LIMIT_QPS", "10")))
BURST = int(__import__("os").getenv("RATE_LIMIT_BURST", "20"))

_buckets: Dict[Tuple[str, str], Dict[str, float]] = {}


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    key = (ip, request.url.path)
    now = time.time()
    bucket = _buckets.get(key)
    if not bucket:
        bucket = {"tokens": BURST, "timestamp": now}
        _buckets[key] = bucket
    # refill
    elapsed = now - bucket["timestamp"]
    bucket["timestamp"] = now
    refill = elapsed * RATE_LIMIT_QPS
    bucket["tokens"] = min(BURST, bucket["tokens"] + refill)
    if bucket["tokens"] < 1:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket["tokens"] -= 1
