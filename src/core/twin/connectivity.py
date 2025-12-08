"""
Telemetry connectivity and serialization helpers for Digital Twin ingestion.

Provides:
- TelemetryFrame Pydantic model
- MsgPack (preferred) / JSON serialization helpers with graceful fallback
- MQTT client scaffold (uses aiomqtt when available; degrades to no-op)
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

try:
    import msgpack  # type: ignore[import-untyped]

    _msgpack_available = True
except Exception:
    _msgpack_available = False

try:
    import aiomqtt

    _aiomqtt_available = True
except Exception:
    _aiomqtt_available = False


class TelemetryFrame(BaseModel):
    """Canonical telemetry envelope."""

    timestamp: float = Field(..., description="Unix timestamp seconds")
    device_id: str = Field(..., description="Source device / asset identifier")
    sensors: Dict[str, float] = Field(default_factory=dict, description="Raw sensor readings")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Derived metrics/health signals"
    )
    status: Dict[str, Any] = Field(default_factory=dict, description="Additional status/labels")

    @field_validator("timestamp")
    @classmethod
    def _timestamp_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    def to_bytes(self) -> bytes:
        """Serialize to bytes (MsgPack preferred, JSON fallback)."""
        payload = self.model_dump()
        if _msgpack_available:
            try:
                result: bytes = msgpack.packb(payload, use_bin_type=True)
                return result
            except Exception:
                logger.debug("MsgPack serialization failed; falling back to JSON", exc_info=True)
        return json.dumps(payload, ensure_ascii=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "TelemetryFrame":
        """Deserialize from MsgPack or JSON bytes."""
        if not data:
            raise ValueError("empty telemetry payload")
        if _msgpack_available:
            try:
                raw = msgpack.unpackb(data, raw=False)
                if isinstance(raw, dict):
                    return cls(**raw)
            except Exception:
                logger.debug("MsgPack decode failed; trying JSON", exc_info=True)
        return cls(**json.loads(data.decode("utf-8")))


@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    cafile: Optional[str] = None
    qos: int = 1
    client_id: str = "cad-ml-platform"


class MqttTelemetryClient:
    """Lightweight MQTT client wrapper with graceful degradation when aiomqtt is absent."""

    def __init__(self, config: Optional[MqttConfig] = None) -> None:
        self.config = config or MqttConfig()
        self._client: Optional[Any] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._stopped = asyncio.Event()
        self._subscribed = asyncio.Event()

    async def start(
        self,
        topics: list[str],
        handler: Callable[[str, bytes], Awaitable[None]],
    ) -> None:
        """Start subscription loop; no-op if aiomqtt is unavailable."""
        if not _aiomqtt_available:
            logger.warning("aiomqtt not installed; MQTT client is disabled")
            return
        if self._client is not None:
            return

        tls_context = None
        if self.config.cafile:
            tls_context = ssl.create_default_context(cafile=self.config.cafile)

        # aiomqtt Client signature varies by version; pass only supported args
        client_kwargs = {
            "hostname": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "password": self.config.password,
            "tls_context": tls_context,
        }
        # Filter out None values to avoid unexpected keyword errors
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
        self._client = aiomqtt.Client(**client_kwargs)  # type: ignore[arg-type]

        async def _runner() -> None:
            assert self._client is not None
            async with self._client:
                for topic in topics:
                    await self._client.subscribe(topic, qos=self.config.qos)
                self._subscribed.set()
                messages = self._client.messages  # MessagesIterator (async iterator)
                async for message in messages:
                    try:
                        await handler(message.topic, message.payload)
                    except Exception:
                        logger.exception("Telemetry handler failed for topic %s", message.topic)
                    if self._stopped.is_set():
                        break

        self._task = asyncio.create_task(_runner())

    async def wait_subscribed(self, timeout: float = 2.0) -> bool:
        """Wait until subscription is established (best-effort)."""
        try:
            await asyncio.wait_for(self._subscribed.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def stop(self) -> None:
        """Stop subscription loop."""
        self._stopped.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._client = None


__all__ = [
    "TelemetryFrame",
    "MqttTelemetryClient",
    "MqttConfig",
    "_msgpack_available",
    "_aiomqtt_available",
]
