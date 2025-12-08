"""Distributed rate limiter using Redis + Lua (token bucket).

Falls back to in-process leaky bucket when Redis unavailable.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional  # noqa: F401 (placeholder for optional parameters)

from src.utils.cache import get_client
from src.utils.metrics import ocr_rate_limited_total

_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local refill = tonumber(ARGV[3]) -- tokens per second
local cost = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'timestamp')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then tokens = capacity end
if ts == nil then ts = now end

-- refill
local delta = math.max(0, now - ts)
tokens = math.min(capacity, tokens + delta * refill)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  allowed = 0
end

redis.call('HMSET', key, 'tokens', tokens, 'timestamp', now)
redis.call('EXPIRE', key, 60)
return allowed
"""


class RateLimiter:
    def __init__(self, key: str, qps: float = 10.0, burst: int = 10):
        self.key = f"ocr:rl:{key}"
        self.qps = qps
        self.burst = max(1, burst)
        self._local_tokens = float(self.burst)
        self._local_ts = time.time()
        self._lock = asyncio.Lock()
        self._script = None

    async def allow(self, cost: int = 1) -> bool:
        client = get_client()
        now = time.time()
        if client is None:  # fallback local
            async with self._lock:
                elapsed = max(0.0, now - self._local_ts)
                refilled = self._local_tokens + elapsed * self.qps
                self._local_tokens = min(self.burst, refilled)
                self._local_ts = now
                if self._local_tokens >= cost:
                    self._local_tokens -= cost
                    return True
                else:
                    ocr_rate_limited_total.inc()
                    return False
        # Redis path
        try:
            if self._script is None:
                self._script = client.register_script(_LUA_TOKEN_BUCKET)
            allowed = await self._script(keys=[self.key], args=[now, self.burst, self.qps, cost])
            if int(allowed) == 1:
                return True
            else:
                ocr_rate_limited_total.inc()
                return False
        except Exception:
            # fail-open to local limiter
            async with self._lock:
                elapsed = max(0.0, now - self._local_ts)
                self._local_tokens = min(self.burst, self._local_tokens + elapsed * self.qps)
                self._local_ts = now
                if self._local_tokens >= cost:
                    self._local_tokens -= cost
                    return True
                else:
                    ocr_rate_limited_total.inc()
                    return False
