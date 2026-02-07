"""Core provider abstractions.

This module defines a lightweight, reusable provider base class used by
cross-domain integrations (vision/ocr/knowledge/etc.) without coupling
directly to domain-specific models.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, Optional, TypeVar

ConfigT = TypeVar("ConfigT")
ResultT = TypeVar("ResultT")


class ProviderStatus(str, Enum):
    """Provider runtime status."""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class ProviderConfig:
    """Generic provider configuration."""

    name: str
    provider_type: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProvider(ABC, Generic[ConfigT, ResultT]):
    """Reusable async provider base class.

    Subclasses implement `_process_impl`. `health_check` behavior can be
    customized by overriding `_health_check_impl`.
    """

    def __init__(self, config: ConfigT):
        self.config = config
        self._status = ProviderStatus.UNKNOWN
        self._last_error: Optional[str] = None
        self._last_health_check_at: Optional[float] = None
        self._last_health_check_latency_ms: Optional[float] = None

    @property
    def status(self) -> ProviderStatus:
        return self._status

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def name(self) -> str:
        cfg_name = getattr(self.config, "name", None)
        if isinstance(cfg_name, str) and cfg_name:
            return cfg_name
        return self.__class__.__name__

    @property
    def provider_type(self) -> Optional[str]:
        cfg_type = getattr(self.config, "provider_type", None)
        if isinstance(cfg_type, str) and cfg_type:
            return cfg_type
        return None

    @property
    def last_health_check_at(self) -> Optional[float]:
        return self._last_health_check_at

    @property
    def last_health_check_latency_ms(self) -> Optional[float]:
        return self._last_health_check_latency_ms

    async def warmup(self) -> None:
        """Optional initialization hook."""

    async def shutdown(self) -> None:
        """Optional cleanup hook."""

    async def process(self, request: Any, **kwargs: Any) -> ResultT:
        return await self._process_impl(request, **kwargs)

    async def health_check(self) -> bool:
        """Run provider health check and update runtime status."""
        started_at = time.perf_counter()
        ok = False
        try:
            ok = await self._health_check_impl()
            self._status = ProviderStatus.HEALTHY if ok else ProviderStatus.DOWN
            if ok:
                self._last_error = None
            return ok
        except Exception as exc:  # noqa: BLE001
            self._status = ProviderStatus.DOWN
            self._last_error = str(exc)
            return False
        finally:
            self._last_health_check_at = time.time()
            self._last_health_check_latency_ms = (
                time.perf_counter() - started_at
            ) * 1000.0

    def mark_degraded(self, reason: str) -> None:
        self._status = ProviderStatus.DEGRADED
        self._last_error = reason

    def mark_healthy(self) -> None:
        self._status = ProviderStatus.HEALTHY
        self._last_error = None

    def status_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider_type": self.provider_type,
            "status": self.status.value,
            "last_error": self.last_error,
            "last_health_check_at": self.last_health_check_at,
            "last_health_check_latency_ms": self.last_health_check_latency_ms,
        }

    async def _health_check_impl(self) -> bool:
        return True

    @abstractmethod
    async def _process_impl(self, request: Any, **kwargs: Any) -> ResultT:
        """Provider-specific processing implementation."""
