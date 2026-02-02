"""Health Check Module.

Provides health checking infrastructure:
- Liveness and readiness probes
- Dependency health checks
- Health aggregation
- Kubernetes-compatible endpoints
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"  # Is the app running?
    READINESS = "readiness"  # Is the app ready to serve?
    STARTUP = "startup"  # Has the app started?


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    check_type: CheckType = CheckType.READINESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class AggregatedHealth:
    """Aggregated health status."""

    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [c.to_dict() for c in self.checks],
        }


class HealthCheck(ABC):
    """Abstract health check."""

    def __init__(
        self,
        name: str,
        check_type: CheckType = CheckType.READINESS,
        timeout: float = 5.0,
        critical: bool = True,
    ):
        self._name = name
        self._check_type = check_type
        self._timeout = timeout
        self._critical = critical

    @property
    def name(self) -> str:
        return self._name

    @property
    def check_type(self) -> CheckType:
        return self._check_type

    @property
    def is_critical(self) -> bool:
        return self._critical

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        pass

    async def execute(self) -> HealthCheckResult:
        """Execute check with timeout."""
        start = time.time()
        try:
            result = await asyncio.wait_for(self.check(), timeout=self._timeout)
            result.duration_ms = (time.time() - start) * 1000
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {self._timeout}s",
                duration_ms=(time.time() - start) * 1000,
                check_type=self._check_type,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
                check_type=self._check_type,
            )


class FunctionHealthCheck(HealthCheck):
    """Health check from a function."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        check_type: CheckType = CheckType.READINESS,
        **kwargs,
    ):
        super().__init__(name, check_type, **kwargs)
        self._check_func = check_func

    async def check(self) -> HealthCheckResult:
        result = self._check_func()
        if asyncio.iscoroutine(result):
            result = await result

        if result:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="Check passed",
                check_type=self._check_type,
            )
        else:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message="Check failed",
                check_type=self._check_type,
            )


class HTTPHealthCheck(HealthCheck):
    """HTTP endpoint health check."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._url = url
        self._expected_status = expected_status

    async def check(self) -> HealthCheckResult:
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(self._url, timeout=aiohttp.ClientTimeout(total=self._timeout)) as response:
                    if response.status == self._expected_status:
                        return HealthCheckResult(
                            name=self._name,
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP {response.status}",
                            details={"url": self._url, "status_code": response.status},
                            check_type=self._check_type,
                        )
                    else:
                        return HealthCheckResult(
                            name=self._name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Expected {self._expected_status}, got {response.status}",
                            details={"url": self._url, "status_code": response.status},
                            check_type=self._check_type,
                        )
        except ImportError:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="aiohttp not available",
                check_type=self._check_type,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                details={"url": self._url},
                check_type=self._check_type,
            )


class TCPHealthCheck(HealthCheck):
    """TCP connection health check."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._host = host
        self._port = port

    async def check(self) -> HealthCheckResult:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout,
            )
            writer.close()
            await writer.wait_closed()

            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="TCP connection successful",
                details={"host": self._host, "port": self._port},
                check_type=self._check_type,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                details={"host": self._host, "port": self._port},
                check_type=self._check_type,
            )


class DatabaseHealthCheck(HealthCheck):
    """Database connection health check."""

    def __init__(
        self,
        name: str,
        connection_func: Callable[[], Any],
        query: str = "SELECT 1",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._connection_func = connection_func
        self._query = query

    async def check(self) -> HealthCheckResult:
        try:
            conn = self._connection_func()
            if asyncio.iscoroutine(conn):
                conn = await conn

            # Execute simple query
            if hasattr(conn, "execute"):
                result = conn.execute(self._query)
                if asyncio.iscoroutine(result):
                    await result

            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.HEALTHY,
                message="Database connection healthy",
                check_type=self._check_type,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                check_type=self._check_type,
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Disk space health check."""

    def __init__(
        self,
        name: str,
        path: str = "/",
        min_free_percent: float = 10.0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._path = path
        self._min_free_percent = min_free_percent

    async def check(self) -> HealthCheckResult:
        try:
            import shutil

            total, used, free = shutil.disk_usage(self._path)
            free_percent = (free / total) * 100

            details = {
                "path": self._path,
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percent": free_percent,
            }

            if free_percent >= self._min_free_percent:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                    message=f"{free_percent:.1f}% free",
                    details=details,
                    check_type=self._check_type,
                )
            else:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Low disk space: {free_percent:.1f}% free",
                    details=details,
                    check_type=self._check_type,
                )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                check_type=self._check_type,
            )


class MemoryHealthCheck(HealthCheck):
    """Memory usage health check."""

    def __init__(
        self,
        name: str,
        max_used_percent: float = 90.0,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._max_used_percent = max_used_percent

    async def check(self) -> HealthCheckResult:
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_percent = memory.percent

            details = {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": used_percent,
            }

            if used_percent <= self._max_used_percent:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.HEALTHY,
                    message=f"{used_percent:.1f}% used",
                    details=details,
                    check_type=self._check_type,
                )
            else:
                return HealthCheckResult(
                    name=self._name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"High memory usage: {used_percent:.1f}%",
                    details=details,
                    check_type=self._check_type,
                )
        except ImportError:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available",
                check_type=self._check_type,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self._name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                check_type=self._check_type,
            )


class HealthChecker:
    """Manages and executes health checks."""

    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._cache: Dict[str, HealthCheckResult] = {}
        self._cache_ttl: float = 5.0  # Cache results for 5 seconds

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self._checks[check.name] = check

    def remove_check(self, name: str) -> None:
        """Remove a health check."""
        self._checks.pop(name, None)
        self._cache.pop(name, None)

    def list_checks(self) -> List[str]:
        """List all check names."""
        return list(self._checks.keys())

    async def check(self, name: str, use_cache: bool = True) -> HealthCheckResult:
        """Execute a single health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not found",
            )

        # Check cache
        if use_cache and name in self._cache:
            cached = self._cache[name]
            if (datetime.utcnow() - cached.timestamp).total_seconds() < self._cache_ttl:
                return cached

        result = await self._checks[name].execute()
        self._cache[name] = result
        return result

    async def check_all(
        self,
        check_type: Optional[CheckType] = None,
        use_cache: bool = True,
    ) -> AggregatedHealth:
        """Execute all health checks."""
        checks_to_run = [
            c for c in self._checks.values()
            if check_type is None or c.check_type == check_type
        ]

        results = await asyncio.gather(*[
            self.check(c.name, use_cache) for c in checks_to_run
        ])

        # Aggregate status
        status = self._aggregate_status(results, checks_to_run)

        return AggregatedHealth(
            status=status,
            checks=list(results),
        )

    def _aggregate_status(
        self,
        results: List[HealthCheckResult],
        checks: List[HealthCheck],
    ) -> HealthStatus:
        """Aggregate health status from individual checks."""
        if not results:
            return HealthStatus.HEALTHY

        # Map check names to their critical status
        critical_map = {c.name: c.is_critical for c in checks}

        unhealthy_critical = False
        unhealthy_non_critical = False

        for result in results:
            if result.status == HealthStatus.UNHEALTHY:
                if critical_map.get(result.name, True):
                    unhealthy_critical = True
                else:
                    unhealthy_non_critical = True

        if unhealthy_critical:
            return HealthStatus.UNHEALTHY
        elif unhealthy_non_critical:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def liveness(self) -> AggregatedHealth:
        """Get liveness status."""
        return await self.check_all(CheckType.LIVENESS)

    async def readiness(self) -> AggregatedHealth:
        """Get readiness status."""
        return await self.check_all(CheckType.READINESS)

    async def startup(self) -> AggregatedHealth:
        """Get startup status."""
        return await self.check_all(CheckType.STARTUP)


class HealthEndpoint:
    """HTTP endpoint handler for health checks."""

    def __init__(self, checker: HealthChecker):
        self._checker = checker

    async def health(self) -> Dict[str, Any]:
        """Full health check endpoint."""
        result = await self._checker.check_all()
        return result.to_dict()

    async def live(self) -> Dict[str, Any]:
        """Liveness probe endpoint."""
        result = await self._checker.liveness()
        return {
            "status": result.status.value,
            "timestamp": result.timestamp.isoformat(),
        }

    async def ready(self) -> Dict[str, Any]:
        """Readiness probe endpoint."""
        result = await self._checker.readiness()
        return {
            "status": result.status.value,
            "timestamp": result.timestamp.isoformat(),
        }

    def get_http_status(self, health: AggregatedHealth) -> int:
        """Get HTTP status code for health result."""
        if health.status == HealthStatus.HEALTHY:
            return 200
        elif health.status == HealthStatus.DEGRADED:
            return 200  # Still serving, just degraded
        else:
            return 503


def create_health_checker(
    checks: Optional[List[Dict[str, Any]]] = None,
) -> HealthChecker:
    """Factory to create configured health checker."""
    checker = HealthChecker()

    if checks:
        for check_config in checks:
            check_type_str = check_config.get("type", "function")

            if check_type_str == "http":
                check = HTTPHealthCheck(
                    name=check_config["name"],
                    url=check_config["url"],
                    expected_status=check_config.get("expected_status", 200),
                    timeout=check_config.get("timeout", 5.0),
                    critical=check_config.get("critical", True),
                )
            elif check_type_str == "tcp":
                check = TCPHealthCheck(
                    name=check_config["name"],
                    host=check_config["host"],
                    port=check_config["port"],
                    timeout=check_config.get("timeout", 5.0),
                    critical=check_config.get("critical", True),
                )
            elif check_type_str == "disk":
                check = DiskSpaceHealthCheck(
                    name=check_config["name"],
                    path=check_config.get("path", "/"),
                    min_free_percent=check_config.get("min_free_percent", 10.0),
                    critical=check_config.get("critical", False),
                )
            elif check_type_str == "memory":
                check = MemoryHealthCheck(
                    name=check_config["name"],
                    max_used_percent=check_config.get("max_used_percent", 90.0),
                    critical=check_config.get("critical", False),
                )
            else:
                continue

            checker.add_check(check)

    return checker


__all__ = [
    "HealthStatus",
    "CheckType",
    "HealthCheckResult",
    "AggregatedHealth",
    "HealthCheck",
    "FunctionHealthCheck",
    "HTTPHealthCheck",
    "TCPHealthCheck",
    "DatabaseHealthCheck",
    "DiskSpaceHealthCheck",
    "MemoryHealthCheck",
    "HealthChecker",
    "HealthEndpoint",
    "create_health_checker",
]
