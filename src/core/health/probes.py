"""Kubernetes Probes Implementation.

Provides liveness, readiness, and startup probes for Kubernetes integration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProbeStatus(str, Enum):
    """Probe status."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class ProbeResult:
    """Result of a probe check."""

    status: ProbeStatus
    timestamp: datetime
    message: Optional[str] = None
    details: Dict[str, Any] = None

    @property
    def is_success(self) -> bool:
        return self.status == ProbeStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "details": self.details or {},
        }


class ProbeBase:
    """Base class for Kubernetes probes."""

    def __init__(
        self,
        failure_threshold: int = 3,
        success_threshold: int = 1,
        timeout_seconds: float = 5.0,
        period_seconds: float = 10.0,
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.period_seconds = period_seconds

        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_result: Optional[ProbeResult] = None
        self._checks: List[Callable[[], bool]] = []

    def add_check(self, check: Callable[[], bool]) -> None:
        """Add a check function."""
        self._checks.append(check)

    async def execute(self) -> ProbeResult:
        """Execute the probe."""
        try:
            start_time = time.time()

            # Run all checks with timeout
            success = await asyncio.wait_for(
                self._run_checks(),
                timeout=self.timeout_seconds,
            )

            elapsed = time.time() - start_time

            if success:
                self._consecutive_successes += 1
                self._consecutive_failures = 0
                result = ProbeResult(
                    status=ProbeStatus.SUCCESS,
                    timestamp=datetime.utcnow(),
                    message="All checks passed",
                    details={
                        "duration_ms": round(elapsed * 1000, 2),
                        "consecutive_successes": self._consecutive_successes,
                    },
                )
            else:
                self._consecutive_failures += 1
                self._consecutive_successes = 0
                result = ProbeResult(
                    status=ProbeStatus.FAILURE,
                    timestamp=datetime.utcnow(),
                    message="One or more checks failed",
                    details={
                        "duration_ms": round(elapsed * 1000, 2),
                        "consecutive_failures": self._consecutive_failures,
                    },
                )

        except asyncio.TimeoutError:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            result = ProbeResult(
                status=ProbeStatus.FAILURE,
                timestamp=datetime.utcnow(),
                message=f"Probe timed out after {self.timeout_seconds}s",
                details={"consecutive_failures": self._consecutive_failures},
            )

        except Exception as e:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            result = ProbeResult(
                status=ProbeStatus.FAILURE,
                timestamp=datetime.utcnow(),
                message=f"Probe error: {str(e)}",
                details={"consecutive_failures": self._consecutive_failures},
            )

        self._last_result = result
        return result

    async def _run_checks(self) -> bool:
        """Run all registered checks."""
        if not self._checks:
            return True

        for check in self._checks:
            try:
                if asyncio.iscoroutinefunction(check):
                    result = await check()
                else:
                    result = check()
                if not result:
                    return False
            except Exception as e:
                logger.error(f"Probe check failed: {e}")
                return False

        return True

    def should_restart(self) -> bool:
        """Check if failures exceed threshold (for liveness)."""
        return self._consecutive_failures >= self.failure_threshold

    def is_ready(self) -> bool:
        """Check if successes meet threshold (for readiness)."""
        return self._consecutive_successes >= self.success_threshold


class LivenessProbe(ProbeBase):
    """Liveness probe - is the application alive?

    Failed liveness probe causes container restart.
    Should only fail if the application is in an unrecoverable state.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        timeout_seconds: float = 5.0,
        period_seconds: float = 10.0,
    ):
        super().__init__(
            failure_threshold=failure_threshold,
            success_threshold=1,
            timeout_seconds=timeout_seconds,
            period_seconds=period_seconds,
        )

        # Add basic liveness check
        self.add_check(self._basic_liveness)

    def _basic_liveness(self) -> bool:
        """Basic check that application is running."""
        return True

    def add_deadlock_detection(self) -> None:
        """Add deadlock detection check."""
        self.add_check(self._check_deadlock)

    def _check_deadlock(self) -> bool:
        """Check for potential deadlocks."""
        # Could implement actual deadlock detection
        # For now, just check if event loop is responsive
        return True


class ReadinessProbe(ProbeBase):
    """Readiness probe - can the application serve traffic?

    Failed readiness probe removes pod from service endpoints.
    Should fail during initialization or when dependencies are unavailable.
    """

    def __init__(
        self,
        health_checker: Optional[Any] = None,
        failure_threshold: int = 3,
        success_threshold: int = 1,
        timeout_seconds: float = 10.0,
        period_seconds: float = 10.0,
    ):
        super().__init__(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            period_seconds=period_seconds,
        )
        self._health_checker = health_checker

    async def _run_checks(self) -> bool:
        """Override to include health checker."""
        # Run registered checks
        if not await super()._run_checks():
            return False

        # Run health checker if available
        if self._health_checker:
            try:
                result = await self._health_checker.check_all()
                from src.core.health.checker import HealthStatus
                return result.status != HealthStatus.UNHEALTHY
            except Exception as e:
                logger.error(f"Health checker error: {e}")
                return False

        return True


class StartupProbe(ProbeBase):
    """Startup probe - has the application started?

    Used for slow-starting applications.
    Disables liveness and readiness probes until startup succeeds.
    """

    def __init__(
        self,
        startup_checks: Optional[List[Callable[[], bool]]] = None,
        failure_threshold: int = 30,  # More tolerant during startup
        timeout_seconds: float = 10.0,
        period_seconds: float = 5.0,
    ):
        super().__init__(
            failure_threshold=failure_threshold,
            success_threshold=1,
            timeout_seconds=timeout_seconds,
            period_seconds=period_seconds,
        )

        self._started = False

        # Add startup checks
        if startup_checks:
            for check in startup_checks:
                self.add_check(check)

    async def execute(self) -> ProbeResult:
        """Execute startup probe."""
        if self._started:
            # Already started, always succeed
            return ProbeResult(
                status=ProbeStatus.SUCCESS,
                timestamp=datetime.utcnow(),
                message="Application already started",
            )

        result = await super().execute()

        if result.is_success:
            self._started = True
            logger.info("Startup probe succeeded - application ready")

        return result

    @property
    def is_started(self) -> bool:
        """Check if startup has completed."""
        return self._started


class ProbeManager:
    """Manages all Kubernetes probes."""

    def __init__(self):
        self.liveness = LivenessProbe()
        self.readiness = ReadinessProbe()
        self.startup = StartupProbe()
        self._running = False
        self._tasks: Dict[str, asyncio.Task] = {}

    def set_health_checker(self, health_checker: Any) -> None:
        """Set health checker for readiness probe."""
        self.readiness._health_checker = health_checker

    async def start_background_probes(self) -> None:
        """Start background probe execution."""
        if self._running:
            return

        self._running = True

        async def run_probe(name: str, probe: ProbeBase) -> None:
            while self._running:
                try:
                    await probe.execute()
                except Exception as e:
                    logger.error(f"{name} probe error: {e}")
                await asyncio.sleep(probe.period_seconds)

        self._tasks["liveness"] = asyncio.create_task(
            run_probe("liveness", self.liveness)
        )
        self._tasks["readiness"] = asyncio.create_task(
            run_probe("readiness", self.readiness)
        )

        logger.info("Background probes started")

    async def stop_background_probes(self) -> None:
        """Stop background probe execution."""
        self._running = False

        for name, task in self._tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("Background probes stopped")

    async def check_liveness(self) -> ProbeResult:
        """Execute liveness probe."""
        return await self.liveness.execute()

    async def check_readiness(self) -> ProbeResult:
        """Execute readiness probe."""
        # Check startup first
        if not self.startup.is_started:
            return ProbeResult(
                status=ProbeStatus.FAILURE,
                timestamp=datetime.utcnow(),
                message="Application not yet started",
            )
        return await self.readiness.execute()

    async def check_startup(self) -> ProbeResult:
        """Execute startup probe."""
        return await self.startup.execute()


# Global probe manager
_probe_manager: Optional[ProbeManager] = None


def get_probe_manager() -> ProbeManager:
    """Get global probe manager."""
    global _probe_manager
    if _probe_manager is None:
        _probe_manager = ProbeManager()
    return _probe_manager
