"""
Telemetry ingestion pipeline with backpressure handling.

Consumes telemetry frames (bytes/dicts/models), enqueues them, and writes to a
time-series store via a background worker. Drops are tracked when the queue is full.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.core.storage.timeseries import TimeSeriesStore
from src.core.twin.connectivity import TelemetryFrame
from src.core.config import get_settings
from src.core.storage.timeseries import InMemoryTimeSeriesStore, NullTimeSeriesStore

logger = logging.getLogger(__name__)


class TelemetryIngestor:
    def __init__(self, store: TimeSeriesStore, max_queue: int = 1000) -> None:
        self.store = store
        self._max_queue = max_queue
        self._queue: Optional[asyncio.Queue[TelemetryFrame]] = None
        self.drop_count = 0
        self._worker: Optional[asyncio.Task[None]] = None
        self._started = False
        self._start_lock: Optional[asyncio.Lock] = None

    @property
    def queue(self) -> asyncio.Queue[TelemetryFrame]:
        """Lazily create queue to avoid issues with missing event loop."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self._max_queue)
        return self._queue

    def _get_start_lock(self) -> asyncio.Lock:
        """Lazily create lock to avoid issues with missing event loop."""
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        return self._start_lock

    async def ensure_started(self) -> None:
        """Start background worker if not already running."""
        async with self._get_start_lock():
            if self._started:
                return
            self._started = True
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop worker task."""
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._worker = None
        self._started = False

    async def handle_payload(self, payload: Any, topic: Optional[str] = None) -> Dict[str, Any]:
        """Decode payload and enqueue for persistence."""
        try:
            frame = self._coerce_frame(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to decode telemetry payload", exc_info=True)
            return {"status": "rejected", "reason": str(exc)}

        try:
            self.queue.put_nowait(frame)
            return {"status": "queued", "device_id": frame.device_id, "topic": topic}
        except asyncio.QueueFull:
            self.drop_count += 1
            return {"status": "dropped", "reason": "backpressure", "device_id": frame.device_id}

    async def _run(self) -> None:
        """Worker loop that drains the queue to the time-series store."""
        while True:
            frame = await self.queue.get()
            try:
                await self.store.append(frame)
            except Exception:
                logger.exception("Failed to persist telemetry frame")
            finally:
                self.queue.task_done()

    def _coerce_frame(self, payload: Any) -> TelemetryFrame:
        """Normalize incoming payload to TelemetryFrame."""
        if isinstance(payload, TelemetryFrame):
            return payload
        if isinstance(payload, (bytes, bytearray)):
            return TelemetryFrame.from_bytes(bytes(payload))
        if isinstance(payload, dict):
            return TelemetryFrame(**payload)
        raise TypeError(f"Unsupported telemetry payload type: {type(payload)}")


# Shared in-memory store and ingestor for routers/tests
_store: Optional[TimeSeriesStore] = None
_ingestor: Optional[TelemetryIngestor] = None


def get_ingestor() -> TelemetryIngestor:
    global _ingestor, _store
    if _ingestor is None:
        settings = get_settings()
        backend = settings.TELEMETRY_STORE_BACKEND.lower()
        if backend == "memory":
            _store = InMemoryTimeSeriesStore()
        elif backend in {"influx", "timescale"}:
            # Placeholders until full adapters are added
            logger.warning("Telemetry store backend '%s' not implemented; using memory", backend)
            _store = InMemoryTimeSeriesStore()
        elif backend == "none":
            _store = NullTimeSeriesStore()
        else:
            logger.warning("Unknown TELEMETRY_STORE_BACKEND=%s; falling back to memory", backend)
            _store = InMemoryTimeSeriesStore()
        _ingestor = TelemetryIngestor(_store)
    return _ingestor


def get_store() -> TimeSeriesStore:
    get_ingestor()  # ensure initialization
    assert _store is not None
    return _store


async def reset_ingestor_for_tests() -> None:
    """Reset singleton for tests (clears queue and store)."""
    global _ingestor, _store
    if _ingestor:
        await _ingestor.stop()
    _ingestor = None
    _store = None
