"""Health Aggregation.

Provides health check and aggregation:
- Health check execution
- Health status aggregation
- Dependency health tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AggregatedHealth:
    """Aggregated health status."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, critical: bool = True, timeout: float = 5.0):
        self.name = name
        self.critical = critical
        self.timeout = timeout

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Execute health check."""
        pass


class HTTPHealthCheck(HealthCheck):
    """HTTP-based health check."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        critical: bool = True,
        timeout: float = 5.0,
    ):
        super().__init__(name, critical, timeout)
        self.url = url
        self.expected_status = expected_status

    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            # Use aiohttp if available, otherwise simulate
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    latency = (time.time() - start) * 1000
                    if response.status == self.expected_status:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message=f"HTTP {response.status}",
                            latency_ms=latency,
                            details={"url": self.url},
                        )
                    else:
                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"HTTP {response.status} (expected {self.expected_status})",
                            latency_ms=latency,
                            details={"url": self.url, "status": response.status},
                        )
        except ImportError:
            # Simulate for testing without aiohttp
            await asyncio.sleep(0.01)
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Simulated (aiohttp not installed)",
                latency_ms=latency,
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.timeout}s",
                latency_ms=self.timeout * 1000,
                details={"url": self.url},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
                details={"url": self.url, "error": type(e).__name__},
            )


class TCPHealthCheck(HealthCheck):
    """TCP connection health check."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        critical: bool = True,
        timeout: float = 5.0,
    ):
        super().__init__(name, critical, timeout)
        self.host = host
        self.port = port

    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.timeout,
            )
            latency = (time.time() - start) * 1000
            writer.close()
            await writer.wait_closed()

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"TCP connection successful",
                latency_ms=latency,
                details={"host": self.host, "port": self.port},
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Connection timeout after {self.timeout}s",
                latency_ms=self.timeout * 1000,
                details={"host": self.host, "port": self.port},
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
                details={"host": self.host, "port": self.port, "error": type(e).__name__},
            )


class FunctionHealthCheck(HealthCheck):
    """Custom function-based health check."""

    def __init__(
        self,
        name: str,
        check_fn: Callable[[], Any],
        critical: bool = True,
        timeout: float = 5.0,
    ):
        super().__init__(name, critical, timeout)
        self.check_fn = check_fn

    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.check_fn):
                result = await asyncio.wait_for(
                    self.check_fn(),
                    timeout=self.timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self.check_fn),
                    timeout=self.timeout,
                )

            latency = (time.time() - start) * 1000

            # Interpret result
            if isinstance(result, HealthCheckResult):
                return result
            elif isinstance(result, bool):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    latency_ms=latency,
                )
            elif isinstance(result, dict):
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details=result,
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    latency_ms=latency,
                )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Timeout after {self.timeout}s",
                latency_ms=self.timeout * 1000,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=latency,
                details={"error": type(e).__name__},
            )


class HealthAggregator:
    """Aggregates health check results."""

    def __init__(self):
        self._checks: List[HealthCheck] = []
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._check_interval = 30.0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self._checks.append(check)

    def remove_check(self, name: str) -> None:
        """Remove a health check by name."""
        self._checks = [c for c in self._checks if c.name != name]
        if name in self._last_results:
            del self._last_results[name]

    async def check_all(self) -> AggregatedHealth:
        """Run all health checks and aggregate results."""
        results = await asyncio.gather(
            *[c.check() for c in self._checks],
            return_exceptions=True,
        )

        check_results: List[HealthCheckResult] = []
        for i, result in enumerate(results):
            check = self._checks[i]
            if isinstance(result, Exception):
                check_results.append(HealthCheckResult(
                    name=check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                    details={"error": type(result).__name__},
                ))
            else:
                check_results.append(result)

        # Cache results
        for result in check_results:
            self._last_results[result.name] = result

        # Aggregate status
        overall_status = self._aggregate_status(check_results)

        # Generate summary
        healthy_count = sum(1 for r in check_results if r.status == HealthStatus.HEALTHY)
        total_count = len(check_results)
        summary = f"{healthy_count}/{total_count} checks passing"

        return AggregatedHealth(
            status=overall_status,
            checks=check_results,
            summary=summary,
        )

    def _aggregate_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall health status."""
        if not results:
            return HealthStatus.UNKNOWN

        has_critical_failure = False
        has_any_failure = False
        all_healthy = True

        for i, result in enumerate(results):
            check = self._checks[i]
            if result.status == HealthStatus.UNHEALTHY:
                all_healthy = False
                has_any_failure = True
                if check.critical:
                    has_critical_failure = True
            elif result.status == HealthStatus.DEGRADED:
                all_healthy = False
            elif result.status == HealthStatus.UNKNOWN:
                all_healthy = False

        if has_critical_failure:
            return HealthStatus.UNHEALTHY
        elif has_any_failure:
            return HealthStatus.DEGRADED
        elif all_healthy:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED

    def get_last_result(self, name: str) -> Optional[HealthCheckResult]:
        """Get last result for a specific check."""
        return self._last_results.get(name)

    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get all last results."""
        return self._last_results.copy()

    async def start_background_checks(self, interval: float = 30.0) -> None:
        """Start background health checking."""
        self._check_interval = interval
        self._running = True
        self._task = asyncio.create_task(self._background_loop())

    async def stop_background_checks(self) -> None:
        """Stop background health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _background_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"Background health check error: {e}")

            await asyncio.sleep(self._check_interval)


class DependencyHealthTracker:
    """Tracks health of service dependencies."""

    def __init__(self):
        self._dependencies: Dict[str, HealthAggregator] = {}

    def add_dependency(
        self,
        name: str,
        checks: List[HealthCheck],
    ) -> HealthAggregator:
        """Add a dependency with its health checks."""
        aggregator = HealthAggregator()
        for check in checks:
            aggregator.add_check(check)
        self._dependencies[name] = aggregator
        return aggregator

    async def check_dependency(self, name: str) -> Optional[AggregatedHealth]:
        """Check health of a specific dependency."""
        aggregator = self._dependencies.get(name)
        if aggregator:
            return await aggregator.check_all()
        return None

    async def check_all_dependencies(self) -> Dict[str, AggregatedHealth]:
        """Check health of all dependencies."""
        results = {}
        for name, aggregator in self._dependencies.items():
            results[name] = await aggregator.check_all()
        return results

    def get_overall_status(self, results: Dict[str, AggregatedHealth]) -> HealthStatus:
        """Get overall status from dependency results."""
        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN
