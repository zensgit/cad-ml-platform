"""Health Check Implementation.

Provides comprehensive health checking for all system dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DependencyHealth:
    """Health status of a dependency."""

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_checked: datetime = field(default_factory=datetime.utcnow)
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthCheckResult:
    """Overall health check result."""

    status: HealthStatus
    timestamp: datetime
    version: str
    uptime_seconds: float
    dependencies: Dict[str, DependencyHealth]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "dependencies": {
                name: {
                    "status": dep.status.value,
                    "latency_ms": dep.latency_ms,
                    "message": dep.message,
                    "details": dep.details,
                    "consecutive_failures": dep.consecutive_failures,
                }
                for name, dep in self.dependencies.items()
            },
            "metadata": self.metadata,
        }


class HealthCheck(ABC):
    """Base class for health checks."""

    def __init__(
        self,
        name: str,
        critical: bool = True,
        timeout: float = 5.0,
    ):
        self.name = name
        self.critical = critical  # If critical, failure makes overall status unhealthy
        self.timeout = timeout
        self._last_result: Optional[DependencyHealth] = None

    @abstractmethod
    async def check(self) -> DependencyHealth:
        """Perform health check."""
        pass

    async def execute(self) -> DependencyHealth:
        """Execute health check with timeout."""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout,
            )
            result.latency_ms = (time.time() - start_time) * 1000

            # Update consecutive counters
            if result.is_healthy:
                result.consecutive_successes = (
                    (self._last_result.consecutive_successes + 1)
                    if self._last_result else 1
                )
                result.consecutive_failures = 0
            else:
                result.consecutive_failures = (
                    (self._last_result.consecutive_failures + 1)
                    if self._last_result else 1
                )
                result.consecutive_successes = 0

            self._last_result = result
            return result

        except asyncio.TimeoutError:
            result = DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Health check timed out after {self.timeout}s",
            )
            if self._last_result:
                result.consecutive_failures = self._last_result.consecutive_failures + 1
            self._last_result = result
            return result

        except Exception as e:
            result = DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"Health check failed: {str(e)}",
            )
            if self._last_result:
                result.consecutive_failures = self._last_result.consecutive_failures + 1
            self._last_result = result
            return result


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections."""

    def __init__(
        self,
        name: str = "database",
        connection_string: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.connection_string = connection_string or os.getenv("DATABASE_URL", "")

    async def check(self) -> DependencyHealth:
        if not self.connection_string:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="No database connection configured",
            )

        try:
            # Try to import and test connection
            import asyncpg
            conn = await asyncpg.connect(self.connection_string, timeout=self.timeout)
            result = await conn.fetchval("SELECT 1")
            await conn.close()

            if result == 1:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Unexpected query result",
                )
        except ImportError:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="asyncpg not installed",
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connections."""

    def __init__(
        self,
        name: str = "redis",
        host: str = "localhost",
        port: int = 6379,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))

    async def check(self) -> DependencyHealth:
        try:
            import redis.asyncio as redis
            client = redis.Redis(host=self.host, port=self.port)
            await client.ping()
            info = await client.info("server")
            await client.close()

            return DependencyHealth(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                details={
                    "version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                },
            )
        except ImportError:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="redis package not installed",
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.url = url
        self.expected_status = expected_status

    async def check(self) -> DependencyHealth:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == self.expected_status:
                        return DependencyHealth(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP {response.status}",
                            details={"url": self.url},
                        )
                    else:
                        return DependencyHealth(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Unexpected status: {response.status}",
                            details={"url": self.url, "status": response.status},
                        )
        except ImportError:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="aiohttp not installed",
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                details={"url": self.url},
            )


class TCPHealthCheck(HealthCheck):
    """Health check for TCP connectivity."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.host = host
        self.port = port

    async def check(self) -> DependencyHealth:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )
            writer.close()
            await writer.wait_closed()

            return DependencyHealth(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"TCP connection to {self.host}:{self.port} successful",
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"TCP connection failed: {str(e)}",
            )


class DiskSpaceHealthCheck(HealthCheck):
    """Health check for disk space."""

    def __init__(
        self,
        name: str = "disk",
        path: str = "/",
        warning_threshold: float = 0.8,  # 80%
        critical_threshold: float = 0.95,  # 95%
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> DependencyHealth:
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.path)
            usage_ratio = used / total

            details = {
                "path": self.path,
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round(usage_ratio * 100, 1),
            }

            if usage_ratio >= self.critical_threshold:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk usage critical: {details['usage_percent']}%",
                    details=details,
                )
            elif usage_ratio >= self.warning_threshold:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Disk usage high: {details['usage_percent']}%",
                    details=details,
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Disk usage OK: {details['usage_percent']}%",
                    details=details,
                )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )


class MemoryHealthCheck(HealthCheck):
    """Health check for memory usage."""

    def __init__(
        self,
        name: str = "memory",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> DependencyHealth:
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_ratio = memory.percent / 100

            details = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
            }

            if usage_ratio >= self.critical_threshold:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory usage critical: {memory.percent}%",
                    details=details,
                )
            elif usage_ratio >= self.warning_threshold:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Memory usage high: {memory.percent}%",
                    details=details,
                )
            else:
                return DependencyHealth(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"Memory usage OK: {memory.percent}%",
                    details=details,
                )
        except ImportError:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return DependencyHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )


class CustomHealthCheck(HealthCheck):
    """Custom health check using a user-provided function."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], DependencyHealth],
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.check_func = check_func

    async def check(self) -> DependencyHealth:
        if asyncio.iscoroutinefunction(self.check_func):
            return await self.check_func()
        else:
            return self.check_func()


class HealthChecker:
    """Central health checker managing all health checks."""

    def __init__(
        self,
        service_name: str = "cad-ml-platform",
        version: str = "1.0.0",
    ):
        self.service_name = service_name
        self.version = version
        self._start_time = time.time()
        self._checks: Dict[str, HealthCheck] = {}
        self._listeners: List[Callable[[HealthCheckResult], None]] = []
        self._last_result: Optional[HealthCheckResult] = None

    def register(self, check: HealthCheck) -> None:
        """Register a health check."""
        self._checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)

    def add_listener(self, listener: Callable[[HealthCheckResult], None]) -> None:
        """Add listener for health check results."""
        self._listeners.append(listener)

    async def check_all(self) -> HealthCheckResult:
        """Run all health checks."""
        dependencies: Dict[str, DependencyHealth] = {}

        # Run all checks in parallel
        tasks = {
            name: asyncio.create_task(check.execute())
            for name, check in self._checks.items()
        }

        for name, task in tasks.items():
            try:
                dependencies[name] = await task
            except Exception as e:
                dependencies[name] = DependencyHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )

        # Determine overall status
        overall_status = self._compute_overall_status(dependencies)

        result = HealthCheckResult(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=self.version,
            uptime_seconds=time.time() - self._start_time,
            dependencies=dependencies,
            metadata={
                "service": self.service_name,
                "hostname": socket.gethostname(),
            },
        )

        self._last_result = result

        # Notify listeners
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(result)
                else:
                    listener(result)
            except Exception as e:
                logger.error(f"Health listener error: {e}")

        return result

    def _compute_overall_status(
        self,
        dependencies: Dict[str, DependencyHealth],
    ) -> HealthStatus:
        """Compute overall status from dependencies."""
        has_unhealthy_critical = False
        has_degraded = False

        for name, dep in dependencies.items():
            check = self._checks.get(name)
            is_critical = check.critical if check else True

            if dep.status == HealthStatus.UNHEALTHY:
                if is_critical:
                    has_unhealthy_critical = True
                else:
                    has_degraded = True
            elif dep.status == HealthStatus.DEGRADED:
                has_degraded = True

        if has_unhealthy_critical:
            return HealthStatus.UNHEALTHY
        elif has_degraded:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def check_liveness(self) -> bool:
        """Simple liveness check (is the service running?)."""
        return True

    async def check_readiness(self) -> bool:
        """Readiness check (can the service handle requests?)."""
        result = await self.check_all()
        return result.status != HealthStatus.UNHEALTHY

    @property
    def last_result(self) -> Optional[HealthCheckResult]:
        """Get last health check result."""
        return self._last_result


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def setup_default_checks(checker: Optional[HealthChecker] = None) -> HealthChecker:
    """Setup default health checks."""
    hc = checker or get_health_checker()

    # System checks (non-critical)
    hc.register(DiskSpaceHealthCheck(critical=False))
    hc.register(MemoryHealthCheck(critical=False))

    # Redis (if configured)
    if os.getenv("REDIS_HOST") or os.getenv("REDIS_URL"):
        hc.register(RedisHealthCheck())

    # Database (if configured)
    if os.getenv("DATABASE_URL"):
        hc.register(DatabaseHealthCheck())

    return hc
