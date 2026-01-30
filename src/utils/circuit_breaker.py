"""Simple circuit breaker with half-open probe using Redis or in-memory.

States: 0=closed, 1=half_open, 2=open
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional  # noqa: F401 (future optional enhancements)

from src.utils.cache import get_client
from src.utils.metrics import ocr_circuit_state


@dataclass
class CircuitConfig:
    error_threshold: int = 5
    timeout_seconds: int = 300
    half_open_requests: int = 2


class CircuitBreaker:
    def __init__(self, key: str, cfg: Optional[CircuitConfig] = None):
        self.key = f"ocr:cb:{key}"
        self.cfg = cfg or CircuitConfig()
        self._state = 0
        self._opened_at = 0.0
        self._half_open_budget = 0
        self._lock = asyncio.Lock()
        ocr_circuit_state.labels(key=self.key).set(0)

    async def _get_state(self) -> int:
        client = get_client()
        if client is None:
            return self._state
        try:
            v = await client.get(self.key)
            return int(v) if v is not None else 0
        except Exception:
            return self._state

    async def _set_state(self, state: int) -> None:
        client = get_client()
        self._state = state
        ocr_circuit_state.labels(key=self.key).set(state)
        if client is None:
            return
        try:
            if state == 0:
                await client.delete(self.key)
            else:
                await client.setex(self.key, self.cfg.timeout_seconds, str(state))
        except Exception:
            pass

    async def should_allow(self) -> bool:
        async with self._lock:
            state = await self._get_state()
            now = time.time()
            if state == 2:  # open
                if now - self._opened_at >= self.cfg.timeout_seconds:
                    await self._set_state(1)
                    self._half_open_budget = self.cfg.half_open_requests
                    return True
                return False
            if state == 1:  # half-open
                if self._half_open_budget > 0:
                    self._half_open_budget -= 1
                    return True
                return False
            return True

    async def on_success(self) -> None:
        async with self._lock:
            await self._set_state(0)

    async def on_error(self) -> None:
        async with self._lock:
            state = await self._get_state()
            if state == 0:
                # transition to open after threshold; we keep simple counterless for now
                self._opened_at = time.time()
                await self._set_state(2)
            elif state == 1:
                # half-open -> open immediately
                self._opened_at = time.time()
                await self._set_state(2)
