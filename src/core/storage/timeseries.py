"""
Time-series storage abstractions for telemetry ingestion.

Provides a simple in-memory ring buffer implementation suitable for tests
and early-stage hardening; can be swapped with InfluxDB/Timescale adapters later.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Deque, Dict, List, Protocol

from src.core.twin.connectivity import TelemetryFrame


class TimeSeriesStore(Protocol):
    async def append(self, frame: TelemetryFrame) -> None: ...

    async def history(self, device_id: str, limit: int = 100) -> List[TelemetryFrame]: ...


class InMemoryTimeSeriesStore:
    """Per-device ring buffer with async-friendly locks."""

    def __init__(self, max_per_device: int = 1000):
        self._max_per_device = max_per_device
        self._store: Dict[str, Deque[TelemetryFrame]] = defaultdict(deque)
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock to avoid issues with missing event loop at import time."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def append(self, frame: TelemetryFrame) -> None:
        async with self._get_lock():
            buf = self._store[frame.device_id]
            buf.append(frame)
            if len(buf) > self._max_per_device:
                buf.popleft()

    async def history(self, device_id: str, limit: int = 100) -> List[TelemetryFrame]:
        async with self._get_lock():
            buf = self._store.get(device_id, deque())
            if not buf:
                return []
            # Return newest first for quick dashboards
            return list(buf)[-limit:][::-1]


class NullTimeSeriesStore:
    """No-op store used when backend is explicitly disabled.

    Useful for running API surfaces without persisting telemetry.
    """

    async def append(self, frame: TelemetryFrame) -> None:  # pragma: no cover - trivial
        return

    async def history(  # pragma: no cover - trivial
        self, device_id: str, limit: int = 100
    ) -> List[TelemetryFrame]:
        return []


__all__ = ["TimeSeriesStore", "InMemoryTimeSeriesStore", "NullTimeSeriesStore"]
